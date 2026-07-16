"""Dropout layer (contract §7): forward-time RNG via `self.rng()`."""

from __future__ import annotations

import jax

from ..module import Module
from ..namespaces import flags, rng


class Dropout(Module):
    """Inverted dropout: zero a fraction `p` in train, identity in eval (§7).

    Randomness is a forward-time `rng` leaf split by `self.rng()`, so the mask is
    a pure function of the threaded state. Shares the global `training` flag with
    `BatchNorm`, so one switch drives both in a composed model.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        self.rng_key = rng(self.key())
        self.training = flags(True)

    def __call__(self, x: jax.Array) -> jax.Array:
        if not self.flags.training or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = jax.random.bernoulli(self.rng(), keep, x.shape)
        return x * mask / keep
