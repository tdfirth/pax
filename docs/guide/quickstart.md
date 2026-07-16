# Quickstart

This page takes a hand-written `Linear` from definition to a single training
step. Every snippet runs against the committed core.

## 1. Define a module

A `pax.Module` is a plain class. In `__init__` you assign the module's state to
`self`; the assignment is classified by value type (contract §5.2):

- a bare `jax.Array` → a **param** (the default namespace),
- a child `Module` → a **subtree**,
- a `Tagged` value (`pax.buffer(...)`, `pax.flags(...)`) → its namespace,
- anything else (int, float, str, tuple) → a plain config attribute.

You never call `super().__init__()` — `Module.__new__` sets up all bookkeeping
before `__init__` runs.

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

`__call__` takes **only** `x`. It reads `self.W` / `self.b` back through `self`,
which resolves to the bound state during a `forward` (and to the init values
otherwise). Child modules, if any, are invoked as `self.child(x)` — no state
argument is threaded by hand.

## 2. Construct under a seed and extract state

Initialization is deterministic in *(root seed, construction order)* — rename-safe
but reorder-sensitive, exactly like PyTorch's `manual_seed`. Construct inside
`with pax.seed(n):` for reproducibility. `self.key()` pulls a fresh subkey per
parameter from that ambient seed.

```python
with pax.seed(0):
    model = Linear(4, 8)

state = model.state()
# state == {'params': {'W': f32[4, 8], 'b': f32[8]}}
```

`model.state()` returns a **fresh** pytree each call — a plain nested dict of
arrays with one top-level key per namespace. `params` is always present.

## 3. Run the forward pass

`forward(state, x) -> (new_state, y)` is the pure entry point. It binds the state
onto the module tree, runs `__call__`, collects any writes, and returns the new
state alongside the output.

```python
x = jnp.ones((3, 4))
new_state, y = model.forward(state, x)   # y: f32[3, 8]
```

For a stateless layer like `Linear`, `new_state` is structurally identical to
`state` (nothing was written). Layers that own buffers — e.g. a BatchNorm's
running statistics — return the updated values in `new_state`.

## 4. Wrap in `jit`

Because `forward` is a pure function of `(state, x)` and the module holds no
traced arrays between calls, you `jit` it directly:

```python
forward = jax.jit(model.forward)
new_state, y = forward(state, x)
```

No wrapper, no `apply`, no manual static/traced partition. See
[transforms](transforms.md) for `vmap` / `scan` / `grad` / `remat`.

## 5. One training step

Gradients are taken **with respect to params only**. The trick (contract §4) is
to make `params` the differentiated argument and splice it back into the full
state with `{**state, 'params': p}`:

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

`grads` has exactly the structure of `state["params"]`, so `optax` consumes it
directly. If a layer also writes buffers, thread `new_state["buffers"]` back into
the `state` you pass on the next step; the params-only gradient path is unchanged.
See [training](training.md) for the full loop — stateful buffers, train/eval
flags, and freezing part of the model. To compose layers, see
[combinators](combinators.md).
