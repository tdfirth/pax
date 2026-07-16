# Transforms

The design goal of Pax is that **`forward` needs no wrappers**. It is a pure
function

```
forward(state, x) -> (new_state, y)
```

of two pytree arguments, and between calls the module object holds only Python
config, child references, its registry, and its init PRNG key — **no traced
arrays**. The init values kept on the instance (so you can re-`state()`) are never
read during `forward`, so they never leak into a trace. Every JAX transform
therefore composes with `model.forward` directly (contract §4):

```python
jax.jit(model.forward)
jax.vmap(model.forward, in_axes=(None, 0))
jax.lax.scan(model.forward, state, xs)
jax.grad(lambda p, s, x: loss(model.forward({**s, "params": p}, x)))
jax.remat(model.forward)
```

Setup used throughout this page:

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


with pax.seed(0):
    model = Linear(4, 8)

state = model.state()
```

## `jit`

Just wrap it:

```python
forward = jax.jit(model.forward)
new_state, y = forward(state, jnp.ones((3, 4)))
```

The whole `state` pytree is traced as an argument, so there is no separate
"partition into static and traced" step — the pytree structure does the
partitioning for you (see [static under jit](#static-namespaces-under-jit)).

## `vmap`

`forward` takes state first, input second, so batch over the input axis and leave
the shared state unbatched with `in_axes=(None, 0)`:

```python
xs = jnp.ones((32, 4))                                  # a batch of 32
batched = jax.vmap(model.forward, in_axes=(None, 0))
new_state, ys = batched(state, xs)                      # ys: f32[32, 8]
```

`None` broadcasts `state` (one shared copy of the weights); `0` maps over the
leading axis of `xs`. For a layer that does **not** update state per example,
prefer `out_axes=(None, 0)` so the returned `new_state` stays a single unbatched
copy rather than 32 identical stacks:

```python
batched = jax.vmap(model.forward, in_axes=(None, 0), out_axes=(None, 0))
```

(For a layer that *does* write per-example state, keep the default `out_axes=0`.)

## `scan`

`forward` was given the signature `(carry, x) -> (carry, y)` precisely so it *is*
a `scan` body — no adapter needed. Thread `state` as the carry to run the same
module over a sequence, carrying state (e.g. a growing cache or evolving buffers)
from step to step:

```python
xs = jnp.ones((10, 4))                                  # 10 timesteps
final_state, ys = jax.lax.scan(model.forward, state, xs)  # ys: f32[10, 8]
```

## `grad` (params-only)

Differentiate with respect to `params` alone by making it the differentiated
argument and splicing it back into the full state with `{**state, "params": p}`:

```python
def loss(state, x, targets):
    _, pred = model.forward(state, x)
    return jnp.mean((pred - targets) ** 2)

x, targets = jnp.ones((3, 4)), jnp.zeros((3, 8))
grads = jax.grad(lambda p: loss({**state, "params": p}, x, targets))(state["params"])
```

`grads` has exactly the structure of `state["params"]`, ready for `optax`. Buffers
and flags are held constant because they are not the differentiated argument.

## `remat`

Rematerialization (gradient checkpointing) wraps `forward` like any other
transform, trading compute for memory during the backward pass:

```python
checkpointed = jax.remat(model.forward)
new_state, y = checkpointed(state, jnp.ones((3, 4)))
```

## Static namespaces under `jit`

Static namespaces (`flags`) ride in the treedef via `pax.Static`, not as leaves,
so their values are part of the `jit` compile-cache key. JAX compiles **one program
per unique flag combination** and reuses it thereafter; inside the trace the flag is
a concrete Python value, so `if self.flags.training:` is a real branch.

```python
class Dropoutish(pax.Module):
    def __init__(self):
        self.training = pax.flags(True)

    def __call__(self, x):
        return x * 0.5 if self.flags.training else x


with pax.seed(0):
    m = Dropoutish()

train_state = m.state()                          # flags: Static({'training': True})
eval_state = {**train_state, "flags": pax.Static({"training": False})}

f = jax.jit(m.forward)
f(train_state, jnp.ones(4))   # compiles the training program
f(eval_state, jnp.ones(4))    # different treedef -> compiles the eval program once
f(train_state, jnp.ones(4))   # reuses the first compilation
```

Two flag combinations, two compilations, cached forever after. This is why flag
values must be **hashable** (they live in `aux_data`): `bool`, `int`, `str`,
`tuple` — the natural shape of a small, finite set of modes.
