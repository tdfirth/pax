"""Pax — a minimal neural network library for JAX.

PyTorch's surface, JAX's soul. `self` on a `Module` doubles as a scope into the
state pytree: `__init__` declares/initializes weights, `__call__` routes state
during the forward pass, and `forward(state, x) -> (new_state, y)` is a pure
function that every JAX transform composes with directly — no wrappers.
"""

from __future__ import annotations

from . import layers
from .combinators import parallel, repeat, sequential
from .module import Module
from .namespaces import Static, buffer, flags, namespace, rng
from .paths import freeze, select
from .random import seed

__all__ = [
    "Module",
    "Static",
    "buffer",
    "flags",
    "freeze",
    "layers",
    "namespace",
    "parallel",
    "repeat",
    "rng",
    "seed",
    "select",
    "sequential",
]
