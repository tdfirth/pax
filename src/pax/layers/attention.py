"""Multi-head attention (contract §3, Task A). STUB — implemented in Phase 2."""

from __future__ import annotations

from typing import Any

from ..module import Module


class Attention(Module):
    """Multi-head attention with optional KV cache. STUB — Phase 2, Task A."""

    def __init__(self, d: int, heads: int) -> None:
        raise NotImplementedError("layers.Attention — Phase 2, Task A")

    def __call__(self, x: Any) -> Any:
        raise NotImplementedError("layers.Attention — Phase 2, Task A")
