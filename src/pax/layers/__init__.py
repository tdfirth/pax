"""Standard layers (contract §3 conventions).

STUB — implemented in Phase 2, Task A. Each layer lives in its own module and is
re-exported here so `from pax.layers import Linear` works once implemented.
"""

from __future__ import annotations

from .attention import Attention
from .dropout import Dropout
from .embedding import Embedding
from .linear import Linear
from .norm import BatchNorm, LayerNorm

__all__ = ["Attention", "BatchNorm", "Dropout", "Embedding", "LayerNorm", "Linear"]
