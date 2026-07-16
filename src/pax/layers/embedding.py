"""Embedding layer (contract §3, Task A)."""

from __future__ import annotations

import jax

from ..module import Module


class Embedding(Module):
    """Lookup table `E[x]` over a `(vocab, d)` matrix (contract §3)."""

    def __init__(self, vocab: int, d: int) -> None:
        self.E = jax.random.normal(self.key(), (vocab, d)) * 0.01

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.E[x]
