"""Path / selective-operation utilities over namespace dicts (contract §9).

STUB — implemented in Phase 2, Task C. Signatures are frozen here so the package
imports and downstream code can reference them.
"""

from __future__ import annotations


def freeze(tree: dict, pattern: str) -> dict:
    """Freeze (stop-gradient) the subtree of `tree` matching a glob (contract §9)."""
    raise NotImplementedError("paths.freeze — Phase 2, Task C")


def select(tree: dict, pattern: str) -> dict:
    """Return the subtree of `tree` matching a dot-path glob (contract §9)."""
    raise NotImplementedError("paths.select — Phase 2, Task C")
