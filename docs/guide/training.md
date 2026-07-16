# Training

Pax owns *state*; **optax owns optimization**. A training loop is plain JAX and
optax operating on the plain pytrees `model.state()` returns — Pax adds no
`Trainer`, no `apply`, no wrapper. This page builds the loop end to end, then
covers stateful layers (BatchNorm), eval mode, and freezing part of the model.

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

Two facts drive everything below:

- `forward(state, x) -> (new_state, y)` is a **pure** function of `(state, x)`;
  the module holds no traced arrays between calls, so you `jit`/`grad` it directly
  (see [transforms](transforms.md)).
- Gradients are taken **with respect to `params` only**. Make `params` the
  differentiated argument and splice it back into the full state with
  `{**state, "params": p}` (contract §4). Buffers and flags are held constant
  because they aren't the differentiated argument.

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
no adaptation. `state` (buffers, flags) is captured as a constant here because this
model writes no buffers — the next section threads them when it must.

## Stateful layers: threading `new_state`

A layer that writes state during `forward` — a `BatchNorm` updating running
statistics — returns the new values in `new_state["buffers"]`. Buffers are **not**
differentiated, but they **must be threaded** from step to step. Take them off
`new_state` (via `has_aux`) and pass them back in on the next step:

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

`buffers` carries the running statistics forward across steps; `grads` still
covers `params` alone. The `flags` namespace stays constant in `state` — the
`training` flag is `True`, so BatchNorm takes its train branch and writes the
buffers.

## Train vs. eval — the `flags` namespace

BatchNorm's `training` lives in the static, global `flags` namespace: it rides in
the treedef (`pax.Static`), so it is a concrete Python `bool` inside the trace and
part of the `jit` compile-cache key (see [namespaces](namespaces.md) and
[transforms — static under jit](transforms.md#static-namespaces-under-jit)). Flip
it for evaluation by swapping in a new `Static`:

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
one eval program — cached and reused thereafter. In the eval branch BatchNorm reads
the running statistics instead of the batch and records no buffer write, so
`new_state["buffers"]` equals what you passed in.

Because `training` is the **global** `flags` namespace, one switch drives every
layer that reads it. Add a `pax.layers.Dropout` and the same `Static({"training":
False})` puts BatchNorm into eval *and* turns Dropout into the identity — no
per-layer bookkeeping. Dropout draws its mask from a forward-time `rng` leaf, so
like buffers it lives in `new_state` (under `state["rng"]`) and must be threaded
to advance the mask; see [transforms — forward-time randomness](transforms.md#forward-time-randomness--selfrng-and-dropout).

## Selective training — `freeze` and `select`

To train only part of a model, mark the rest **frozen** with `pax.freeze`, which
returns a boolean-mask pytree of the same shape as `state["params"]` (`True` at
matched leaves). It composes directly with `optax.masked` and
`optax.multi_transform` (contract §9):

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

Patterns are whole-path globs where `*` matches **exactly one segment**; there is
no prefix matching, so match at the exact leaf depth (`Linear_0.*`, not
`Linear_0`). Equivalently, partition with `multi_transform`:

```python
labels = jax.tree_util.tree_map(lambda frozen: "frozen" if frozen else "train", mask)
tx = optax.multi_transform(
    {"train": optax.adam(1e-3), "frozen": optax.set_to_zero()}, labels
)
```

`pax.select` is the companion for **reading** a subtree — it returns just the
matching leaves (non-matching branches pruned), handy for inspecting or
checkpointing one part of the model:

```python
pax.select(params, "Linear_2.*")   # -> {'Linear_2': {'W': ..., 'b': ...}}
```

Both operate on a single namespace's dict (typically `state["params"]`) and share
one `match_path` primitive, so a mask and a selection always agree on which leaves
match.

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

It has one limitation to know: it **raises** for any model built from pre-existing
child instances — every combinator (`sequential` / `repeat` / `parallel`) and any
module taking a `Module` argument — because replaying construction would reuse
those children and leave their params at the original draw (contract §4). For those
models, reseed by rebuilding under a fresh context instead:

```python
with pax.seed(42):
    model = pax.sequential(Linear(4, 8), jax.nn.relu, Linear(8, 2))
state = model.state()
```
