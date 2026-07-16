"""Embedding layer (contract §3, Task A). STUB — implemented in Phase 2."""

from __future__ import annotations

from typing import Any

from ..module import Module


class Embedding(Module):
    """`E[x]` over an `(vocab, d)` table. STUB — implemented in Phase 2, Task A."""

    def __init__(self, vocab: int, d: int) -> None:
        raise NotImplementedError("layers.Embedding — Phase 2, Task A")

    def __call__(self, x: Any) -> Any:
        raise NotImplementedError("layers.Embedding — Phase 2, Task A")
