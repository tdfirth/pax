"""Forward-time RNG: `self.rng()` and `Dropout` (contract §7).

`self.rng()` splits this module's single `rng` leaf, writes the advanced half
back into state (collected into `new_state['rng']` like a buffer write), and
returns a fresh subkey. Because the randomness is an explicit pytree leaf, the
mask is a pure function of the threaded state: identical input state gives an
identical mask, and threading `new_state` advances it.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

import pax
from pax.layers import BatchNorm, Dropout, Linear
from pax.module import Module
from pax.namespaces import Static, rng


class Noise(Module):
    """Adds standard-normal noise drawn from a forward-RNG leaf."""

    def __init__(self) -> None:
        self.k = rng(self.key())

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + jax.random.normal(self.rng(), x.shape)


def build(cls: type, *args: Any) -> Any:
    with pax.seed(0):
        return cls(*args)


def _key_eq(a: jax.Array, b: jax.Array) -> bool:
    return bool(jnp.array_equal(jax.random.key_data(a), jax.random.key_data(b)))


# -- self.rng(): split, write-back, purity -------------------------------------


def test_rng_leaf_in_state():
    state = build(Noise).state()
    assert "rng" in state
    assert "k" in state["rng"]


def test_same_input_state_gives_identical_output():
    model = build(Noise)
    state = model.state()
    x = jnp.zeros(64)
    _, ya = model.forward(state, x)
    _, yb = model.forward(state, x)
    assert jnp.array_equal(ya, yb)


def test_threading_new_state_advances_the_key():
    model = build(Noise)
    state = model.state()
    x = jnp.zeros(64)
    s1, y1 = model.forward(state, x)
    _, y2 = model.forward(s1, x)
    assert not jnp.allclose(y1, y2)


def test_advanced_key_lands_in_new_state_and_differs():
    model = build(Noise)
    state = model.state()
    new_state, _ = model.forward(state, jnp.zeros(4))
    assert not _key_eq(new_state["rng"]["k"], state["rng"]["k"])


def test_forward_leaves_module_pure():
    model = build(Noise)
    model.forward(model.state(), jnp.zeros(4))
    assert model._bound == []
    assert model._writes == []


# -- self.rng() error paths ----------------------------------------------------


def test_rng_outside_forward_raises():
    model = build(Noise)
    with pytest.raises(RuntimeError, match="forward-time only"):
        model.rng()


class NoLeaf(Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.rng()


def test_rng_with_zero_leaves_raises():
    model = build(NoLeaf)
    with pytest.raises(RuntimeError, match="declare one"):
        model.forward(model.state(), jnp.zeros(4))


class TwoLeaves(Module):
    def __init__(self) -> None:
        self.a = rng(self.key())
        self.b = rng(self.key())

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.rng()


def test_rng_with_two_leaves_raises():
    model = build(TwoLeaves)
    with pytest.raises(RuntimeError, match="ambiguous"):
        model.forward(model.state(), jnp.zeros(4))


# -- transforms ----------------------------------------------------------------


def test_rng_under_jit_matches_eager():
    model = build(Noise)
    state = model.state()
    x = jnp.zeros(64)
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.array_equal(eager, jitted)


def test_rng_under_vmap_different_mask_per_key():
    model = build(Noise)
    base = model.state()
    keys = jax.random.split(jax.random.key(0), 5)
    state = {**base, "rng": {"k": keys}}
    in_axes = ({"params": None, "rng": 0}, None)
    _, ys = jax.vmap(model.forward, in_axes=in_axes)(state, jnp.zeros(8))
    assert ys.shape == (5, 8)
    assert not jnp.allclose(ys[0], ys[1])


def test_rng_under_scan_mask_differs_per_step():
    model = build(Noise)
    state = model.state()
    xs = jnp.zeros((6, 8))
    _, ys = jax.lax.scan(model.forward, state, xs)
    assert ys.shape == (6, 8)
    assert not jnp.allclose(ys[0], ys[1])


def test_rng_is_differentiable_wrt_x():
    model = build(Dropout, 0.5)
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (256,))

    def f(x: jax.Array) -> jax.Array:
        _, y = model.forward(state, x)
        return jnp.sum(y**2)

    g = jax.grad(f)(x)
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0)


# -- Dropout -------------------------------------------------------------------


def test_dropout_eval_is_exact_identity():
    model = build(Dropout, 0.5)
    state: dict[str, Any] = {
        **model.state(),
        "flags": Static({"training": False}),
    }
    x = jax.random.normal(jax.random.key(2), (100,))
    _, y = model.forward(state, x)
    assert jnp.array_equal(y, x)


def test_dropout_p_zero_is_identity():
    model = build(Dropout, 0.0)
    x = jax.random.normal(jax.random.key(3), (100,))
    _, y = model.forward(model.state(), x)
    assert jnp.array_equal(y, x)


def test_dropout_train_zeros_and_preserves_expectation():
    p = 0.3
    model = build(Dropout, p)
    x = jnp.ones(20000)
    _, y = model.forward(model.state(), x)
    frac_zero = jnp.mean(y == 0.0)
    assert jnp.isclose(frac_zero, p, atol=0.02)
    assert jnp.isclose(y.mean(), 1.0, atol=0.02)
    kept = y[y != 0.0]
    assert jnp.allclose(kept, 1.0 / (1.0 - p))


def test_dropout_in_sequential_under_jit():
    with pax.seed(0):
        model = pax.sequential(Linear(4, 8), Dropout(0.5))
    state = model.state()
    x = jnp.ones((3, 4))
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted)


# -- one training flag drives BatchNorm and Dropout together -------------------


class BNDropNet(Module):
    def __init__(self, d: int) -> None:
        self.lin = Linear(d, d)
        self.bn = BatchNorm(d)
        self.drop = Dropout(0.5)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.drop(self.bn(self.lin(x)))


def test_one_flag_switches_both_layers():
    model = build(BNDropNet, 8)
    state = model.state()
    # A single shared 'training' key in the global flags namespace.
    assert dict(state["flags"].data) == {"training": True}
    x = jax.random.normal(jax.random.key(4), (64, 8))

    train_state, y_train = model.forward(state, x)
    assert jnp.any(y_train == 0.0)  # dropout active
    assert not jnp.array_equal(
        train_state["buffers"]["bn"]["running_mean"],
        state["buffers"]["bn"]["running_mean"],
    )  # batchnorm updating

    eval_state: dict[str, Any] = {**state, "flags": Static({"training": False})}
    new_eval, y_eval = model.forward(eval_state, x)
    assert not jnp.any(y_eval == 0.0)  # dropout off
    assert jnp.array_equal(
        new_eval["buffers"]["bn"]["running_mean"],
        eval_state["buffers"]["bn"]["running_mean"],
    )  # batchnorm frozen


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
