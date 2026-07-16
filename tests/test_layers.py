"""Unit tests for the standard layers (contract §3, Task A).

Shape/correctness per layer; BatchNorm buffers update in train and are read but
not updated in eval; Attention's KV cache grows across calls under the flag;
every layer composes with `jax.jit(model.forward)`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

import pax
from pax.layers import Attention, BatchNorm, Embedding, LayerNorm, Linear
from pax.namespaces import Static


def build(cls: type, *args: Any) -> Any:
    with pax.seed(0):
        return cls(*args)


# -- Linear --------------------------------------------------------------------


def test_linear_shape_and_correctness():
    model = build(Linear, 4, 8)
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (3, 4))
    _, y = model.forward(state, x)
    assert y.shape == (3, 8)
    expected = x @ state["params"]["W"] + state["params"]["b"]
    assert jnp.allclose(y, expected)


def test_linear_under_jit():
    model = build(Linear, 4, 8)
    state = model.state()
    x = jnp.ones((3, 4))
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted)


# -- Embedding -----------------------------------------------------------------


def test_embedding_shape_and_correctness():
    model = build(Embedding, 10, 4)
    state = model.state()
    x = jnp.array([0, 3, 9])
    _, y = model.forward(state, x)
    assert y.shape == (3, 4)
    assert jnp.allclose(y, state["params"]["E"][x])


def test_embedding_under_jit():
    model = build(Embedding, 10, 4)
    state = model.state()
    x = jnp.array([1, 2, 3])
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted)


# -- LayerNorm -----------------------------------------------------------------


def test_layernorm_normalizes_last_axis():
    model = build(LayerNorm, 16)
    state = model.state()
    x = jax.random.normal(jax.random.key(2), (5, 16)) * 3.0 + 7.0
    _, y = model.forward(state, x)
    assert y.shape == (5, 16)
    assert jnp.allclose(y.mean(-1), 0.0, atol=1e-5)
    assert jnp.allclose(y.std(-1), 1.0, atol=1e-3)


def test_layernorm_no_buffers():
    state = build(LayerNorm, 16).state()
    assert set(state) == {"params"}
    assert set(state["params"]) == {"g", "b"}


def test_layernorm_under_jit():
    model = build(LayerNorm, 16)
    state = model.state()
    x = jax.random.normal(jax.random.key(2), (5, 16))
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted)


# -- BatchNorm -----------------------------------------------------------------


def test_batchnorm_train_updates_buffers():
    model = build(BatchNorm, 8)
    state = model.state()
    x = jax.random.normal(jax.random.key(3), (16, 8))
    new_state, y = model.forward(state, x)
    assert y.shape == (16, 8)
    assert not jnp.array_equal(
        new_state["buffers"]["running_mean"], state["buffers"]["running_mean"]
    )
    assert not jnp.array_equal(
        new_state["buffers"]["running_var"], state["buffers"]["running_var"]
    )


def test_batchnorm_eval_reads_but_does_not_update_buffers():
    model = build(BatchNorm, 8)
    state = model.state()
    eval_state: dict[str, Any] = {**state, "flags": Static({"training": False})}
    x = jax.random.normal(jax.random.key(3), (16, 8))
    new_state, _ = model.forward(eval_state, x)
    assert jnp.array_equal(
        new_state["buffers"]["running_mean"], eval_state["buffers"]["running_mean"]
    )
    assert jnp.array_equal(
        new_state["buffers"]["running_var"], eval_state["buffers"]["running_var"]
    )


def test_batchnorm_eval_uses_running_stats():
    model = build(BatchNorm, 4)
    state = model.state()  # running_mean=0, running_var=1, so eval is identity affine
    eval_state: dict[str, Any] = {**state, "flags": Static({"training": False})}
    x = jax.random.normal(jax.random.key(4), (6, 4))
    _, y = model.forward(eval_state, x)
    assert jnp.allclose(y, x / jnp.sqrt(1.0 + 1e-5))


def test_batchnorm_under_jit():
    model = build(BatchNorm, 8)
    state = model.state()
    x = jax.random.normal(jax.random.key(3), (16, 8))
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted)


# -- Attention -----------------------------------------------------------------


def test_attention_shape():
    model = build(Attention, 16, 4)
    state = model.state()
    x = jax.random.normal(jax.random.key(5), (7, 16))
    _, y = model.forward(state, x)
    assert y.shape == (7, 16)


def test_attention_under_jit():
    model = build(Attention, 16, 4)
    state = model.state()
    x = jax.random.normal(jax.random.key(5), (7, 16))
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted)


def test_attention_cache_grows_across_calls():
    model = build(Attention, 16, 4)
    state: dict[str, Any] = {
        **model.state(),
        "flags": Static({"use_cache": True}),
    }
    assert state["cache"]["k"].shape == (4, 0, 4)

    x1 = jax.random.normal(jax.random.key(6), (2, 16))
    state, _ = model.forward(state, x1)
    assert state["cache"]["k"].shape == (4, 2, 4)
    assert state["cache"]["v"].shape == (4, 2, 4)

    x2 = jax.random.normal(jax.random.key(7), (3, 16))
    state, _ = model.forward(state, x2)
    assert state["cache"]["k"].shape == (4, 5, 4)
    assert state["cache"]["v"].shape == (4, 5, 4)


def test_attention_cache_off_does_not_grow():
    model = build(Attention, 16, 4)
    state = model.state()
    x = jax.random.normal(jax.random.key(8), (3, 16))
    new_state, _ = model.forward(state, x)
    assert new_state["cache"]["k"].shape == (4, 0, 4)
