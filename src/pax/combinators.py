"""Combinators: sequential, repeat, parallel (contract §8).

STUB — implemented in Phase 2, Task B. Signatures are frozen here so the package
imports and downstream code can reference them.
"""

from __future__ import annotations

from collections.abc import Callable

from .module import Module


def sequential(*layers: Module | Callable) -> Module:
    """Thread `x` through each layer in order (contract §8)."""
    raise NotImplementedError("combinators.sequential — Phase 2, Task B")


def repeat(layer: Module, n: int) -> Module:
    """`n` weight-tied applications of `layer` via `jax.lax.scan` (contract §8)."""
    raise NotImplementedError("combinators.repeat — Phase 2, Task B")


def parallel(*layers: Module | Callable) -> Module:
    """Apply each layer to the input; return a tuple of outputs (contract §8)."""
    raise NotImplementedError("combinators.parallel — Phase 2, Task B")
