"""Init-time PRNG: scoped seed context (contract §6, Option A)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import pax
from pax.module import Module


class Linear(Module):
    def __init__(self, i: int, o: int) -> None:
        self.W = jax.random.normal(self.key(), (i, o)) * 0.01
        self.b = jnp.zeros(o)

    def __call__(self, x):
        return x @ self.W + self.b


def _params(model: Module) -> dict:
    return model.state()["params"]


def test_same_seed_is_reproducible():
    with pax.seed(0):
        a = _params(Linear(4, 8))
    with pax.seed(0):
        b = _params(Linear(4, 8))
    assert jnp.array_equal(a["W"], b["W"])


def test_different_seeds_differ():
    with pax.seed(0):
        a = _params(Linear(4, 8))
    with pax.seed(1):
        b = _params(Linear(4, 8))
    assert not jnp.array_equal(a["W"], b["W"])


def test_construction_order_matters_siblings_differ():
    with pax.seed(0):
        first = _params(Linear(4, 8))
        second = _params(Linear(4, 8))
    assert not jnp.array_equal(first["W"], second["W"])


def test_seed_context_is_nestable_and_restores():
    # A nested context saves and restores the outer key, so the outer's draws
    # are unaffected by whatever the inner context drew.
    with pax.seed(0):
        _params(Linear(4, 8))  # advance the outer key once
        with pax.seed(99):
            _params(Linear(4, 8))  # inner draw — must not perturb the outer
        nested_second = _params(Linear(4, 8))
    with pax.seed(0):
        _params(Linear(4, 8))
        plain_second = _params(Linear(4, 8))
    assert jnp.array_equal(nested_second["W"], plain_second["W"])


def test_state_key_rematerializes_deterministically():
    with pax.seed(0):
        model = Linear(4, 8)
    k = jax.random.key(42)
    a = model.state(key=k)["params"]
    b = model.state(key=k)["params"]
    assert jnp.array_equal(a["W"], b["W"])
    # Different from the construction-time values.
    assert not jnp.array_equal(a["W"], model.state()["params"]["W"])
