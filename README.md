# Pax

A minimal neural network library for JAX. PyTorch's surface, JAX's soul.

`self` on a `Module` doubles as a scope into the state pytree: `__init__`
declares and initializes weights, `__call__` routes state during the forward
pass, and `forward(state, x) -> (new_state, y)` is a pure function that every JAX
transform composes with directly — no wrappers.

```python
import jax, jax.numpy as jnp
import pax

class Linear(pax.Module):
    def __init__(self, i, o):
        self.W = jax.random.normal(self.key(), (i, o)) * 0.01
        self.b = jnp.zeros(o)

    def __call__(self, x):
        return x @ self.W + self.b

with pax.seed(0):
    model = Linear(4, 8)

state = model.state()
new_state, y = jax.jit(model.forward)(state, jnp.ones((3, 4)))
```

> **Status:** v1 in development. See `docs/api-contract.md` (frozen) and
> `docs/implementation-plan.md`. Full documentation is forthcoming (Task E).
