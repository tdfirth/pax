# Training

Pax owns your model's *state*; **optax owns optimization.** There is no `Trainer`,
no `.fit()`, no `apply` — a training loop is plain JAX and optax operating on the
plain pytrees `model.state()` hands you. This page builds that loop end to end,
then covers the three things real models add: stateful layers (BatchNorm), eval
mode, and freezing part of the model.

## The pieces

```python
import jax
import jax.numpy as jnp
import optax
import pax
from pax.layers import Linear

with pax.seed(0):
    model = pax.sequential(Linear(4, 8), jax.nn.relu, Linear(8, 2))

state = model.state()          # {'params': {...}} — a plain nested dict of arrays
```

Two facts (both from earlier pages) drive everything below:

- `forward(state, x) -> (new_state, y)` is a **pure** function of `(state, x)`;
  the model holds no traced arrays between calls, so you `jit` and `grad` it
  directly (see [transforms](transforms.md)).
- Gradients are taken **with respect to `params` only**. You make `params` the
  differentiated argument and splice it back into the full state with
  `{**state, "params": p}`. Buffers and flags ride along in `state` but are held
  constant, because they aren't what's being differentiated.

## A jitted training step

```python
def loss_fn(params, state, x, targets):
    _, pred = model.forward({**state, "params": params}, x)
    return jnp.mean((pred - targets) ** 2)

opt = optax.adam(1e-3)
opt_state = opt.init(state["params"])

@jax.jit
def train_step(params, opt_state, x, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, state, x, targets)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

params = state["params"]
x, targets = jnp.ones((8, 4)), jnp.zeros((8, 2))
for _ in range(100):
    params, opt_state, loss = train_step(params, opt_state, x, targets)
```

`grads` has exactly the structure of `state["params"]`, so optax consumes it with
no adaptation. Here `state` (its buffers, flags) is captured as a constant, which
is fine because this model writes no buffers. The next section handles the case
where it does. A complete runnable version of this is
[`examples/mlp.py`](../../examples/mlp.py).

## Stateful layers: threading `new_state`

A layer that writes state during `forward` — a `BatchNorm` updating its running
statistics — returns the updated values in `new_state["buffers"]`. Buffers are
**not** differentiated, but they **must be carried forward** from step to step,
or your running stats never move. The pattern: pull them off `new_state` (via
`has_aux`) and pass them back in on the next step.

```python
from pax.layers import BatchNorm

class Net(pax.Module):
    def __init__(self, d_in, d_out):
        self.fc = Linear(d_in, d_out)
        self.bn = BatchNorm(d_out)
    def __call__(self, x):
        return self.bn(self.fc(x))

with pax.seed(0):
    model = Net(4, 3)
state = model.state()          # {'params': {...}, 'buffers': {...}, 'flags': Static({'training': True})}

def loss_fn(params, buffers, x, targets):
    new_state, pred = model.forward(
        {**state, "params": params, "buffers": buffers}, x
    )
    return jnp.mean((pred - targets) ** 2), new_state["buffers"]

opt = optax.adam(1e-3)
opt_state = opt.init(state["params"])

@jax.jit
def train_step(params, buffers, opt_state, x, targets):
    (loss, buffers), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, buffers, x, targets
    )
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, buffers, opt_state, loss

params, buffers = state["params"], state["buffers"]
params, buffers, opt_state, loss = train_step(params, buffers, opt_state, x, targets)
```

`buffers` now carries the running statistics forward across steps; `grads` still
covers `params` alone. `flags` stays constant in `state` — `training` is `True`,
so BatchNorm takes its train branch and writes the buffers. See
[`examples/batchnorm.py`](../../examples/batchnorm.py) for the full loop plus a
check that it recompiles exactly once.

## Train vs. eval — the `flags` namespace

BatchNorm's `training` lives in the static, global `flags` namespace: it rides in
the treedef, so inside a trace it's a concrete `bool` and it's part of the `jit`
compile-cache key (see [namespaces](namespaces.md#why-static-flags-survive-jit)).
You flip it for evaluation by swapping in a new `Static`:

```python
eval_state = {
    **state,
    "params": params,
    "buffers": buffers,
    "flags": pax.Static({"training": False}),
}
_, preds = model.forward(eval_state, x)     # uses running stats; writes no buffers
```

Under `jit`, the two flag values produce **two compilations** — one train program,
one eval program — cached and reused after that. In the eval branch BatchNorm
reads the running statistics instead of the batch and records no buffer write, so
`new_state["buffers"]` comes back equal to what you passed in.

And because `training` is a **global** flag, one switch drives *every* layer that
reads it. Add a `pax.layers.Dropout` to the model and the same
`Static({"training": False})` puts BatchNorm into eval mode *and* turns Dropout
into the identity — no per-layer bookkeeping. (Dropout draws its mask from a
forward-time `rng` leaf, which — like buffers — lives in `new_state` and must be
threaded to advance the mask; see
[transforms → forward-time randomness](transforms.md#forward-time-randomness--selfrng-and-dropout).)

## Selective training — `freeze` and `select`

To train only part of a model, mark the rest **frozen**. `pax.freeze` returns a
boolean-mask pytree of the same shape as `state["params"]` — `True` at the leaves
you matched — which drops straight into optax:

```python
with pax.seed(1):
    model = pax.sequential(Linear(4, 8), jax.nn.relu, Linear(8, 2))
params = model.state()["params"]

mask = pax.freeze(params, "Linear_0.*")
# mask == {'Linear_0': {'W': True,  'b': True},
#          'Linear_2': {'W': False, 'b': False}}

# zero the updates of the frozen leaves; everything else trains normally
tx = optax.masked(optax.set_to_zero(), mask)
opt = optax.chain(tx, optax.adam(1e-3))
opt_state = opt.init(params)
```

One thing to know about the patterns: they're **whole-path globs** where `*`
matches *exactly one* path segment. There's no prefix matching, so you match at
the exact leaf depth — `Linear_0.*` (matching `Linear_0.W` and `Linear_0.b`), not
`Linear_0` on its own. Equivalently, you can partition with `multi_transform`:

```python
labels = jax.tree_util.tree_map(lambda frozen: "frozen" if frozen else "train", mask)
tx = optax.multi_transform(
    {"train": optax.adam(1e-3), "frozen": optax.set_to_zero()}, labels
)
```

`pax.select` is the companion for **reading** a subtree — it returns just the
matching leaves, with non-matching branches pruned. Handy for inspecting or
checkpointing one part of a model:

```python
pax.select(params, "Linear_2.*")   # -> {'Linear_2': {'W': ..., 'b': ...}}
```

Both operate on a single namespace's dict (usually `state["params"]`) and share
the same matcher, so a mask and a selection always agree on which leaves match.

## Reproducibility

Initialization is a deterministic function of *(root seed, construction order)* —
rename-safe but reorder-sensitive, exactly like PyTorch's `manual_seed`. For a
reproducible run, construct the whole model under `with pax.seed(n):`.

`model.state(key=...)` is an escape hatch that **re-materializes** params from a
fresh key by replaying construction — useful for re-drawing a hand-written module
without rebuilding it:

```python
with pax.seed(0):
    model = Net(4, 3)
state_a = model.state()
state_b = model.state(key=jax.random.key(42))   # same model, freshly drawn params
```

It has one limitation worth knowing: it **raises** for any model built from
pre-existing child instances — every combinator (`sequential` / `repeat` /
`parallel`), and any module that takes a `Module` as a constructor argument —
because replaying construction would reuse those same child objects and leave
their params at the original draw. For those models, reseed by simply rebuilding
under a fresh context:

```python
with pax.seed(42):
    model = pax.sequential(Linear(4, 8), jax.nn.relu, Linear(8, 2))
state = model.state()
```
