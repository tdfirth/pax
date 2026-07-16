"""Initialization-time PRNG (contract §6, Option A: scoped seed context).

Eager init with a thread-local seed context. There is no key threading in
constructors: `self.key()` reads from an ambient source scoped to construction
(`pax.seed(n)`), not a process-wide persistent global.

Reproducibility is a deterministic function of *(root seed, construction order)*:
rename-safe but reorder-sensitive, identical to PyTorch's `manual_seed`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import jax

from ._types import PRNGKey

_DEFAULT_SEED = 0

_local = threading.local()


def _current_key() -> PRNGKey:
    """The thread-local init key, lazily initialized from the default seed."""
    key = getattr(_local, "key", None)
    if key is None:
        key = jax.random.key(_DEFAULT_SEED)
        _local.key = key
    return key


@contextmanager
def seed(n: int) -> Iterator[None]:
    """Set the thread-local init key to `random.key(n)` for the duration.

    Nestable: the previous key is restored on exit, so unrelated constructions
    do not leak into one another (contract §6).
    """
    prev = getattr(_local, "key", None)
    _local.key = jax.random.key(n)
    try:
        yield
    finally:
        _local.key = prev


@contextmanager
def _use_key(key: PRNGKey) -> Iterator[None]:
    """Like `seed`, but seeded from an existing key (the `state(key=...)` path)."""
    prev = getattr(_local, "key", None)
    _local.key = key
    try:
        yield
    finally:
        _local.key = prev


def _next_module_key() -> PRNGKey:
    """Split the thread-local key, advancing it so sibling modules differ.

    Called by `Module.__new__`. Outside any `seed` context this lazily
    initializes from the default seed and advances thread-locally.
    """
    key, sub = jax.random.split(_current_key())
    _local.key = key
    return sub
