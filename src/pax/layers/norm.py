"""Normalization layers (contract §3, Task A)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..module import Module
from ..namespaces import buffer, flags

_EPS = 1e-5


class LayerNorm(Module):
    """Normalize over the last axis with learned scale/shift (contract §3)."""

    def __init__(self, d: int) -> None:
        self.g = jnp.ones(d)
        self.b = jnp.zeros(d)

    def __call__(self, x: jax.Array) -> jax.Array:
        mean = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return self.g * (x - mean) / jnp.sqrt(var + _EPS) + self.b


class BatchNorm(Module):
    """Batch normalization over axis 0 with running statistics (contract §3).

    The richest layer: params (`g`, `b`), buffers (running stats), a static
    `training` flag, and forward-time buffer writes in the train branch.
    """

    def __init__(self, d: int, momentum: float = 0.9) -> None:
        self.momentum = momentum
        self.g = jnp.ones(d)
        self.b = jnp.zeros(d)
        self.running_mean = buffer(jnp.zeros(d))
        self.running_var = buffer(jnp.ones(d))
        self.training = flags(True)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.flags.training:
            mean = x.mean(0)
            var = x.var(0)
            m = self.momentum
            self.running_mean = m * self.running_mean + (1 - m) * mean
            self.running_var = m * self.running_var + (1 - m) * var
        else:
            mean = self.running_mean
            var = self.running_var
        return self.g * (x - mean) / jnp.sqrt(var + _EPS) + self.b
