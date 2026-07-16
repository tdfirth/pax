"""Normalization layers (contract §3, Task A). STUB — implemented in Phase 2."""

from __future__ import annotations

from typing import Any

from ..module import Module


class LayerNorm(Module):
    """Normalize over the last axis. STUB — implemented in Phase 2, Task A."""

    def __init__(self, d: int) -> None:
        raise NotImplementedError("layers.LayerNorm — Phase 2, Task A")

    def __call__(self, x: Any) -> Any:
        raise NotImplementedError("layers.LayerNorm — Phase 2, Task A")


class BatchNorm(Module):
    """Batch normalization with running stats. STUB — implemented in Phase 2, Task A."""

    def __init__(self, d: int) -> None:
        raise NotImplementedError("layers.BatchNorm — Phase 2, Task A")

    def __call__(self, x: Any) -> Any:
        raise NotImplementedError("layers.BatchNorm — Phase 2, Task A")
