# Transforms

The whole point of Pax is this: **`forward` needs no wrappers.** It's a pure
function

```
forward(state, x) -> (new_state, y)
```

of two pytree arguments, so every JAX transform composes with it *directly* — no
`apply`, no partitioning step, no framework adapter.

## Why it's actually pure

It's worth being precise about why this works, because it's the property
everything below relies on.

Between calls, a Pax module holds only Python config, references to its children,
its init-time bookkeeping, and its init PRNG key — **no traced arrays**. The
weights you initialized are kept on the instance so you can call `state()` again,
but they are *never read during `forward`* (a forward reads from the bound state
you pass in, not from the instance). So nothing on the object can leak into a
trace. `forward(state, x)` genuinely depends only on `state` and `x`.

That's the entire trick. Everything a JAX transform needs — purity, all inputs as
explicit pytree arguments — `forward` already has.

```python
jax.jit(model.forward)
jax.vmap(model.forward, in_axes=(None, 0))
jax.lax.scan(model.forward, state, xs)
jax.grad(lambda p, s, x: loss(model.forward({**s, "params": p}, x)))
jax.remat(model.forward)
```

The rest of this page walks through each one. Shared setup:

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
partitioning for you (see [static namespaces under jit](#static-namespaces-under-jit)).

## `vmap`

`forward` takes state first and input second. To run a batch of inputs against
one shared set of weights, batch the input axis and broadcast the state with
`in_axes=(None, 0)`:

```python
xs = jnp.ones((32, 4))                                  # a batch of 32
batched = jax.vmap(model.forward, in_axes=(None, 0))
new_state, ys = batched(state, xs)                      # ys: f32[32, 8]
```

`None` says "don't map this argument — every example shares the same weights";
`0` maps over the leading axis of `xs`. For a layer that doesn't write per-example
state, add `out_axes=(None, 0)` so the returned `new_state` stays a single
unbatched copy rather than 32 identical stacks:

```python
batched = jax.vmap(model.forward, in_axes=(None, 0), out_axes=(None, 0))
```

(For a layer that *does* write per-example state, keep the default `out_axes=0`.)

## `scan`

This is why `forward` has the signature it does. `jax.lax.scan` wants a body of
shape `(carry, x) -> (carry, y)` — which is *exactly* `forward`'s signature. So
running a module over a sequence, carrying its state from step to step (a growing
cache, evolving buffers), needs no adapter at all:

```python
xs = jnp.ones((10, 4))                                  # 10 timesteps
final_state, ys = jax.lax.scan(model.forward, state, xs)  # ys: f32[10, 8]
```

The state pytree threads through as the carry; the output is stacked across
steps.

## `grad` (params-only)

You almost always want gradients with respect to `params` alone. There's no
special API for it — you make `params` the differentiated argument and splice it
back into the full state:

```python
def loss(state, x, targets):
    _, pred = model.forward(state, x)
    return jnp.mean((pred - targets) ** 2)

x, targets = jnp.ones((3, 4)), jnp.zeros((3, 8))
grads = jax.grad(lambda p: loss({**state, "params": p}, x, targets))(state["params"])
```

`jax.grad` differentiates its first argument, so passing `state["params"]` as that
argument differentiates params alone. Buffers and flags come along inside `state`
but are held constant, because they're not what's being differentiated. `grads`
has exactly the structure of `state["params"]`, ready for optax.

## `remat`

Rematerialization (gradient checkpointing) trades compute for memory on the
backward pass, and wraps `forward` like anything else:

```python
checkpointed = jax.remat(model.forward)
new_state, y = checkpointed(state, jnp.ones((3, 4)))
```

## Static namespaces under `jit`

Recall from [namespaces](namespaces.md#why-static-flags-survive-jit) that `flags`
ride in the treedef via `pax.Static`, not as leaves. The consequence under `jit`
is worth seeing directly: **flag values are part of the compile-cache key**, so
JAX compiles one program per unique flag combination and reuses it.

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

Two flag combinations, two compilations, cached forever after. This is the reason
flag values must be **hashable** (they live in the treedef): `bool`, `int`,
`str`, `tuple` — the natural shape of a small, finite set of modes. To flip the
flag you build a new `Static`, as shown; the [training guide](training.md#train-vs-eval--the-flags-namespace)
does this for real eval-mode inference.

## Forward-time randomness — `self.rng()` and Dropout

`self.key()` is the init-time seed source. It's a thread-local, so it only exists
during construction — using it inside a trace would break purity. Randomness
*during* `forward` therefore has to come from somewhere explicit: a **pytree leaf
that threads through the state**, split fresh on each call.

That's what `pax.rng` and `self.rng()` are. You declare an `rng` leaf at init
(seeded from the init PRNG), and split it during `forward` with `self.rng()`:

```python
class Dropout(pax.Module):
    def __init__(self, p=0.5):
        self.p = p
        self.rng_key = pax.rng(self.key())   # a forward-RNG leaf (any name but `rng`)
        self.training = pax.flags(True)      # shares BatchNorm's global flag

    def __call__(self, x):
        if not self.flags.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = jax.random.bernoulli(self.rng(), keep, x.shape)
        return x * mask / keep               # inverted dropout: no eval-time scaling
```

`self.rng()` splits the leaf, writes the advanced half back into
`new_state["rng"]` (just like a buffer write), and returns a fresh subkey. So the
mask is a **pure function of the input state**, and two useful properties follow:

- **Reproducible.** The same input `state` always gives the same mask. To draw a
  *different* mask next call, **thread `new_state`** forward — the advanced key
  rides along in the carry.
- **Composes with every transform.** Under `jit`, one state gives one mask. Under
  `scan`, the mask differs per step because the key threads through the carry.
  Under `grad`, it's differentiable w.r.t. `x` (gradient flows through the kept
  units). Under `vmap`, map a batch of keys for a per-row mask
  (`in_axes=({"params": None, "rng": 0}, None)`); a single broadcast leaf gives
  the same mask on every row.

Two rules to remember: declare **exactly one** `rng` leaf per module (`self.rng()`
finds it by namespace — zero or two both raise), and **don't name it `rng`** — a
bare `rng` attribute would shadow the `self.rng()` method. Any other name works
(`self.rng_key`, `self.dropout_key`, …).
