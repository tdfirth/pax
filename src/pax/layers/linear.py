"""Linear layer (contract §3, Task A). STUB — implemented in Phase 2."""

from __future__ import annotations

from typing import Any

from ..module import Module


class Linear(Module):
    """`x @ W + b`. STUB — implemented in Phase 2, Task A."""

    def __init__(self, in_features: int, out_features: int) -> None:
        raise NotImplementedError("layers.Linear — Phase 2, Task A")

    def __call__(self, x: Any) -> Any:
        raise NotImplementedError("layers.Linear — Phase 2, Task A")
