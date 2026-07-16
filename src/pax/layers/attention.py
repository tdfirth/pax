"""Multi-head self-attention with an optional KV cache (contract §3, Task A)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..module import Module
from ..namespaces import flags, namespace

# Traced, scoped namespace holding the growing key/value cache (contract §2).
cache = namespace("cache")


def _split_heads(x: jax.Array, heads: int) -> jax.Array:
    """`(seq, d) -> (heads, seq, d // heads)`."""
    seq, d = x.shape
    return x.reshape(seq, heads, d // heads).transpose(1, 0, 2)


def _merge_heads(x: jax.Array) -> jax.Array:
    """`(heads, seq, dh) -> (seq, heads * dh)`."""
    heads, seq, dh = x.shape
    return x.transpose(1, 0, 2).reshape(seq, heads * dh)


class Attention(Module):
    """Scaled dot-product self-attention over a `(seq, d)` sequence (contract §3).

    Exercises a user traced-scoped namespace (`cache`) plus a static flag: when
    `flags.use_cache` is set, new keys/values are concatenated onto the cached
    `k`/`v` and written back, so the cache grows across successive calls.
    """

    def __init__(self, d: int, heads: int) -> None:
        if d % heads != 0:
            raise ValueError(f"d={d} must be divisible by heads={heads}")
        self.heads = heads
        self.Wq = jax.random.normal(self.key(), (d, d)) * 0.01
        self.Wk = jax.random.normal(self.key(), (d, d)) * 0.01
        self.Wv = jax.random.normal(self.key(), (d, d)) * 0.01
        self.Wo = jax.random.normal(self.key(), (d, d)) * 0.01
        self.k = cache(jnp.zeros((heads, 0, d // heads)))
        self.v = cache(jnp.zeros((heads, 0, d // heads)))
        self.use_cache = flags(False)

    def __call__(self, x: jax.Array) -> jax.Array:
        q = _split_heads(x @ self.Wq, self.heads)
        k = _split_heads(x @ self.Wk, self.heads)
        v = _split_heads(x @ self.Wv, self.heads)
        if self.flags.use_cache:
            k = jnp.concatenate([self.k, k], axis=1)
            v = jnp.concatenate([self.v, v], axis=1)
            self.k = k
            self.v = v
        dh = q.shape[-1]
        scores = q @ k.swapaxes(-1, -2) / jnp.sqrt(dh)
        weights = jax.nn.softmax(scores, axis=-1)
        out = _merge_heads(weights @ v)
        return out @ self.Wo
