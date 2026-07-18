# Pax

A minimal neural network library for JAX. **PyTorch's surface, JAX's soul.**

You write models the way you would in PyTorch — a class, weights assigned to
`self` in `__init__`, a `__call__` that uses them. But there is no hidden state
and no framework runtime. Your model compiles down to a single **pure function**
over a **plain pytree of arrays**, and from there it's all JAX: `jit`, `vmap`,
`scan`, `grad`, `optax`. Pax gets out of the way completely.

```python
import jax, jax.numpy as jnp
import pax


class Linear(pax.Module):
    def __init__(self, in_f, out_f):              # no super().__init__()
        self.W = jax.random.normal(self.key(), (in_f, out_f)) * 0.01
        self.b = jnp.zeros(out_f)

    def __call__(self, x):                         # takes only x; reads W/b via self
        return x @ self.W + self.b


with pax.seed(0):
    model = Linear(4, 8)

state = model.state()                              # {'params': {'W': ..., 'b': ...}}
new_state, y = jax.jit(model.forward)(state, jnp.ones((3, 4)))
```

## The core idea

Most JAX frameworks make you choose between a familiar object-oriented surface
and JAX's functional purity. Pax refuses the trade-off with one move: **`self`
doubles as a scope into a state pytree.**

- **`__init__` declares and initializes** the weights. Each assignment to `self`
  is sorted by type — a bare array becomes a trainable **param**, a child module
  becomes a **subtree**, plain config (ints, strings) stays an ordinary
  attribute.
- **`__call__(self, x)` routes state** during the forward pass. It reads
  `self.W`, calls `self.child(x)`, and writes `self.running_mean = ...` — but
  behind the scenes those resolve against a *bound* state pytree, not the object.
- **`forward(state, x) -> (new_state, y)`** is the pure function that ties it
  together. Give it a state pytree and an input; get back the updated state and
  the output. Nothing is stored on the model between calls.

That last point is what makes everything else free. Because `forward` is a pure
function of two pytree arguments — and the model holds no traced arrays between
calls — every JAX transform composes with it *directly*:

```python
jax.jit(model.forward)
jax.vmap(model.forward, in_axes=(None, 0))
jax.lax.scan(model.forward, state, xs)             # forward already has scan's shape
jax.grad(lambda p, s, x: loss(model.forward({**s, "params": p}, x)))
```

No `apply`, no wrapper, no "partition the state into static and traced" step, no
custom optimizer. `state` is an ordinary nested dict of arrays; hand it to
`jax.grad`, `optax`, `jax.tree_util`, or a checkpoint with zero Pax involvement.

**Pax owns exactly two jobs:** declare/initialize state, and route state during
the forward pass. Backprop, optimizer updates, `jit`, `vmap`, `scan` — that's
all plain JAX operating on a plain pytree.

## Install

```bash
uv add pax
```

For local development (editable install with the dev tooling):

```bash
git clone https://github.com/…/pax && cd pax
uv sync
```

The only runtime dependency is `jax`. `optax` is used in the training examples
but Pax never depends on it — you bring your own optimizer.

## Batteries included

A handful of standard layers ship in `pax.layers`, each a small, readable
`Module` you can copy from:

| Layer | What it is |
|-------|------------|
| `Linear(in_f, out_f)` | affine map `x @ W + b` |
| `Embedding(vocab, d)` | lookup table `E[x]` |
| `LayerNorm(d)` | normalize over the last axis, learned scale/shift |
| `BatchNorm(d)` | batch norm with running stats and a train/eval flag |
| `Dropout(p)` | inverted dropout with forward-time randomness |
| `Attention(d, heads)` | multi-head self-attention with an optional KV cache |

And three combinators for the common wiring patterns:

```python
pax.sequential(*layers)   # thread x through each layer in order
pax.repeat(layer, n)      # apply ONE layer n times (weight-tied), via scan
pax.parallel(*layers)     # apply each layer to the input, return a tuple
```

## Guide

Read these in order the first time through — each builds on the last.

1. [**Quickstart**](docs/guide/quickstart.md) — define a module, extract state,
   run `forward`, `jit` it, and write a one-step training loop.
2. [**Namespaces**](docs/guide/namespaces.md) — how state is organized: params
   vs. buffers vs. flags, the traced/static and scoped/global axes, custom
   namespaces, and tied weights.
3. [**Transforms**](docs/guide/transforms.md) — why `forward` needs no wrappers
   under `jit` / `vmap` / `scan` / `grad` / `remat`, plus forward-time
   randomness.
4. [**Combinators**](docs/guide/combinators.md) — `sequential` / `repeat` /
   `parallel`, when to use them, and when plain Python in `__call__` is better.
5. [**Training**](docs/guide/training.md) — the end-to-end optax loop: params-only
   gradients, threading buffers, train/eval mode, and freezing part of a model.

Runnable end-to-end examples live in [`examples/`](examples/): an MLP, a
BatchNorm net, and a small transformer with tied embeddings and a KV cache.
