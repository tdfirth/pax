# Quickstart

This page takes a hand-written `Linear` layer all the way from definition to a
single training step. By the end you'll have seen every core piece of Pax:
defining a module, seeding it, extracting state, running `forward`, compiling
with `jit`, and taking a gradient step with optax.

Every snippet runs as written.

## 1. Define a module

A `pax.Module` is a plain Python class. You write the layer exactly as you would
in PyTorch — but you never call `super().__init__()`. Pax sets up its bookkeeping
in `__new__`, before your `__init__` body runs, so there is nothing to
initialize.

```python
import jax
import jax.numpy as jnp
import pax


class Linear(pax.Module):
    def __init__(self, in_f, out_f):
        self.W = jax.random.normal(self.key(), (in_f, out_f)) * 0.01
        self.b = jnp.zeros(out_f)

    def __call__(self, x):
        return x @ self.W + self.b
```

**Each assignment in `__init__` is classified by the type of its value.** This is
the heart of how Pax turns an object into a state pytree:

- a bare `jax.Array` → a **param** (a trainable weight),
- a child `Module` → a **subtree** (a nested part of the model),
- a *tagged* value like `pax.buffer(...)` or `pax.flags(...)` → that namespace
  (covered in [namespaces](namespaces.md)),
- anything else — an `int`, `float`, `str`, `tuple` → a plain config attribute
  that stays on the object and never enters the state pytree.

So `self.W` and `self.b` become params; had you written `self.eps = 1e-5`, that
would just be config.

**`__call__` takes only `x`.** It reads `self.W` and `self.b` back through
`self`. There is no state argument threaded by hand — during a forward pass those
attribute reads resolve against the bound state (more on that in step 3), and
outside a forward pass they resolve to the values you initialized. If the module
had children, you'd invoke them the same natural way: `self.child(x)`.

## 2. Construct under a seed, then extract state

Weight initialization needs randomness, and `self.key()` supplies it — it pulls a
fresh PRNG subkey each time it's called. Where does that randomness come from?
From an ambient **seed context**:

```python
with pax.seed(0):
    model = Linear(4, 8)

state = model.state()
# state == {'params': {'W': f32[4, 8], 'b': f32[8]}}
```

`pax.seed(n)` makes initialization deterministic. Construct the same model under
the same seed and you get the same weights — every time. Initialization depends
on *(the root seed, the order modules are constructed)*, which means it is
**rename-safe but reorder-sensitive**: renaming `self.W` to `self.weight` changes
nothing, but swapping the order of two layers reshuffles which subkey each gets.
This is exactly how PyTorch's `manual_seed` behaves.

`model.state()` walks the module and collects its weights into a **plain nested
dict of arrays** — one top-level key per *namespace*. For a simple layer that's
just `params`. This dict is the single source of truth for your model's state
from here on. It's a fresh copy each time you call `state()`, and it's an
ordinary JAX pytree — nothing Pax-specific about it.

## 3. Run the forward pass

`forward(state, x) -> (new_state, y)` is the pure entry point. Give it a state
pytree and an input, and it:

1. **binds** the state onto the module tree (so `self.W` now reads from
   `state["params"]["W"]`),
2. runs your `__call__(x)`,
3. **collects** any writes made during the call into a fresh `new_state`,
4. unbinds, leaving the model object clean and reusable.

```python
x = jnp.ones((3, 4))
new_state, y = model.forward(state, x)   # y: f32[3, 8]
```

For a stateless layer like `Linear`, nothing was written, so `new_state` is
structurally identical to `state`. Layers that *own* mutable state — a
BatchNorm's running statistics, say — return their updated values in
`new_state`. That's why `forward` returns state as well as output: it's the same
`(carry, x) -> (carry, y)` shape you'd write for a `scan` (not a coincidence —
see [transforms](transforms.md)).

## 4. Compile it with `jit`

Because `forward` is a pure function of `(state, x)`, and the model holds no
traced arrays between calls, you `jit` it directly:

```python
forward = jax.jit(model.forward)
new_state, y = forward(state, x)
```

No wrapper, no `apply`, no manual static/traced partition. The whole `state`
pytree is traced as an argument, and its structure does the partitioning for you.
`vmap`, `scan`, `grad`, and `remat` work the same way — see
[transforms](transforms.md).

## 5. One training step

Here is the one idea that takes a moment to click. Gradients should be taken with
respect to **params only** — not buffers, not flags. Pax achieves this without
any special API by making `params` a *separate argument* to the loss, and
splicing it back into the full state with `{**state, "params": p}`:

```python
import optax

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

targets = jnp.zeros((3, 8))
params = state["params"]
params, opt_state, loss = train_step(params, opt_state, x, targets)
```

`jax.grad` differentiates its *first* argument, so making `params` first means
you differentiate params alone; everything else in `state` is held constant. And
because `grads` comes back with the exact structure of `state["params"]`, optax
consumes it with no adaptation — no framework glue in sight.

If a layer also writes buffers, you thread `new_state["buffers"]` back into the
`state` you pass on the next step; the params-only gradient path above is
unchanged. The [training guide](training.md) builds the full loop — stateful
buffers, train/eval mode, and freezing part of a model. To compose layers into
bigger models, see [combinators](combinators.md); to understand `buffers` /
`flags` and the rest of the state pytree, read [namespaces](namespaces.md) next.
