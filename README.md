# Pax

A minimal neural network library for JAX. **PyTorch's surface, JAX's soul.**

`self` on a `Module` doubles as a scope into the state pytree: `__init__`
declares and initializes weights, `__call__` routes state during the forward
pass, and `forward(state, x) -> (new_state, y)` is a pure function that every JAX
transform composes with directly — no wrappers, no `apply`, no partitioning step.

Pax owns exactly two jobs: **declare/initialize state** and **route state during
the forward pass**. Everything else — backprop, optimizer updates, `jit`, `vmap`,
`scan` — is plain JAX operating on a plain pytree of arrays.

## Install

```bash
uv add pax
```

For local development (editable install with dev tooling):

```bash
git clone https://github.com/…/pax && cd pax
uv sync
```

Runtime dependency is `jax` only.

## Quickstart

A `Module` is a normal Python class. Assignments in `__init__` are classified by
value type: a bare `jax.Array` becomes a **param**, a child `Module` becomes a
subtree, and config (ints, strings) stays a plain attribute. `__call__(self, x)`
reads those params back through `self`.

```python
import jax
import jax.numpy as jnp
import pax


class Linear(pax.Module):
    def __init__(self, in_f, out_f):          # no super().__init__() needed
        self.W = jax.random.normal(self.key(), (in_f, out_f)) * 0.01
        self.b = jnp.zeros(out_f)

    def __call__(self, x):                     # takes only x; reads state via self
        return x @ self.W + self.b


with pax.seed(0):                              # reproducible: (root seed, ctor order)
    model = Linear(4, 8)

state = model.state()                          # a plain pytree: {'params': {'W', 'b'}}

# forward is pure (state, x) -> (new_state, y); jit it directly, no wrapper.
forward = jax.jit(model.forward)
new_state, y = forward(state, jnp.ones((3, 4)))
```

`state` is an ordinary nested dict of arrays. Pass it to `jax.grad`, `optax`,
`jax.tree_util`, or serialization with zero Pax involvement.

## Guide

- [Quickstart](docs/guide/quickstart.md) — define a module, run `forward`, `jit`
  it, and a one-step training loop.
- [Namespaces](docs/guide/namespaces.md) — `params` / `buffers` / `flags`, the
  static×scoped grid, custom namespaces, and tied weights.
- [Transforms](docs/guide/transforms.md) — why `forward` needs no wrappers under
  `jit` / `vmap` / `scan` / `grad` / `remat`.

The full interface is specified in [`docs/api-contract.md`](docs/api-contract.md)
(frozen v1).
