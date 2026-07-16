"""Linear layer (contract §3, Task A)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..module import Module


class Linear(Module):
    """Affine map `x @ W + b` (contract §3)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        self.W = jax.random.normal(self.key(), (in_features, out_features)) * 0.01
        self.b = jnp.zeros(out_features)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.W + self.b
