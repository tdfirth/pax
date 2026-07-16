"""Shared type aliases. Leaf module — imports nothing from pax."""

from __future__ import annotations

import jax

PRNGKey = jax.Array
"""A JAX typed PRNG key (jax.random.key), carried as a jax.Array."""
