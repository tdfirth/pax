"""Combinator tests: sequential / repeat / parallel (contract §8).

Uses hand-written modules (no dependency on Task A layers). Constructs under
`with pax.seed(0)` for reproducibility. Exercises: threading + `ClassName_N`
naming, mixed Module/callable pipelines, `repeat` weight-tying (one param copy)
and its equality to `n` manual applications, tuple output from `parallel`, and
everything under `jax.jit(model.forward)`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import pax
from pax import parallel, repeat, sequential
from pax.module import Module
from pax.namespaces import Static, buffer, flags


class Linear(Module):
    def __init__(self, i: int, o: int) -> None:
        self.W = jax.random.normal(self.key(), (i, o)) * 0.1
        self.b = jnp.zeros(o)

    def __call__(self, x):
        return x @ self.W + self.b


class AddOne(Module):
    """A shape-preserving layer with a param, for `repeat` (tied) tests."""

    def __init__(self, d: int) -> None:
        self.w = jnp.ones(d)

    def __call__(self, x):
        return x + self.w


class Accumulate(Module):
    """Shape-preserving; writes a buffer each call — exercises repeat's carry."""

    def __init__(self, d: int) -> None:
        self.w = jnp.ones(d)
        self.count = buffer(jnp.zeros(d))

    def __call__(self, x):
        self.count = self.count + 1.0
        return x + self.w


class Gated(Module):
    """Reads a static flag — checks flags thread through repeat's scan carry."""

    def __init__(self, d: int) -> None:
        self.w = jnp.ones(d)
        self.on = flags(True)

    def __call__(self, x):
        return x + self.w if self.flags.on else x


def build(cls, *args):
    with pax.seed(0):
        return cls(*args)


def _rep(layer, n):
    with pax.seed(0):
        return repeat(layer, n)


# -- sequential ---------------------------------------------------------------


def test_sequential_threads_input_in_order():
    with pax.seed(0):
        model = sequential(Linear(4, 8), Linear(8, 2))
    state = model.state()
    x = jnp.ones((3, 4))
    _, y = model.forward(state, x)
    assert y.shape == (3, 2)


def test_sequential_names_children_positionally():
    with pax.seed(0):
        model = sequential(Linear(4, 8), Linear(8, 2))
    keys = set(model.state()["params"])
    assert keys == {"Linear_0", "Linear_1"}
    assert set(model.state()["params"]["Linear_0"]) == {"W", "b"}


def test_sequential_matches_manual_composition():
    with pax.seed(0):
        a, b = Linear(4, 8), Linear(8, 2)
    with pax.seed(0):
        model = sequential(Linear(4, 8), Linear(8, 2))
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (3, 4))
    _, y = model.forward(state, x)
    manual = (x @ a.W + a.b) @ b.W + b.b
    assert jnp.allclose(y, manual)


def test_sequential_mixes_modules_and_callables():
    with pax.seed(0):
        model = sequential(Linear(4, 8), jax.nn.relu, Linear(8, 2))
    keys = set(model.state()["params"])
    assert keys == {"Linear_0", "Linear_2"}  # positional N skips the callable
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (3, 4))
    _, y = model.forward(state, x)
    assert y.shape == (3, 2)
    assert jnp.all(jnp.isfinite(y))


def test_sequential_relu_actually_applied():
    with pax.seed(0):
        model = sequential(Linear(4, 4), jax.nn.relu)
    state = model.state()
    x = jax.random.normal(jax.random.key(2), (5, 4))
    _, y = model.forward(state, x)
    assert jnp.all(y >= 0.0)


def test_sequential_under_jit():
    with pax.seed(0):
        model = sequential(Linear(4, 8), jax.nn.relu, Linear(8, 2))
    state = model.state()
    x = jnp.ones((3, 4))
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted)


# -- repeat -------------------------------------------------------------------


def test_repeat_has_exactly_one_param_copy():
    layer = build(AddOne, 4)
    model = _rep(layer, 5)
    params = model.state()["params"]
    assert set(params) == {"AddOne_0"}
    assert set(params["AddOne_0"]) == {"w"}


def test_repeat_equals_n_manual_applications():
    layer = build(AddOne, 4)
    n = 5
    model = _rep(layer, n)
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (3, 4))
    _, y = model.forward(state, x)

    manual = x
    layer_state = state["params"]["AddOne_0"]
    for _i in range(n):
        manual = manual + layer_state["w"]
    assert jnp.allclose(y, manual)


def test_repeat_threads_buffer_writes_through_scan():
    layer = build(Accumulate, 4)
    n = 3
    model = _rep(layer, n)
    state = model.state()
    x = jnp.zeros((2, 4))
    new_state, _ = model.forward(state, x)
    counted = new_state["buffers"]["Accumulate_0"]["count"]
    assert jnp.allclose(counted, jnp.full((4,), float(n)))


def test_repeat_reads_static_flag_through_carry():
    layer = build(Gated, 4)
    model = _rep(layer, 4)
    state = model.state()
    x = jnp.zeros((2, 4))
    _, on = model.forward(state, x)
    assert jnp.allclose(on, jnp.full((2, 4), 4.0))

    off = {**state, "flags": Static({"on": False})}
    _, y_off = model.forward(off, x)
    assert jnp.allclose(y_off, x)


def test_repeat_leaves_module_pure_between_forwards():
    layer = build(AddOne, 4)
    model = _rep(layer, 3)
    state = model.state()
    x = jnp.ones((2, 4))
    model.forward(state, x)
    assert model._bound == []
    assert model.AddOne_0._bound == []


def test_repeat_under_jit():
    layer = build(Accumulate, 4)
    model = _rep(layer, 3)
    state = model.state()
    x = jnp.zeros((2, 4))
    eager_state, eager = model.forward(state, x)
    jit_state, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted)
    assert jnp.allclose(
        eager_state["buffers"]["Accumulate_0"]["count"],
        jit_state["buffers"]["Accumulate_0"]["count"],
    )


# -- parallel -----------------------------------------------------------------


def test_parallel_returns_a_tuple():
    with pax.seed(0):
        model = parallel(Linear(4, 8), Linear(4, 2))
    state = model.state()
    x = jnp.ones((3, 4))
    _, ys = model.forward(state, x)
    assert isinstance(ys, tuple)
    assert len(ys) == 2
    assert ys[0].shape == (3, 8)
    assert ys[1].shape == (3, 2)


def test_parallel_applies_each_layer_to_the_same_input():
    with pax.seed(0):
        a, b = Linear(4, 3), Linear(4, 3)
    with pax.seed(0):
        model = parallel(Linear(4, 3), Linear(4, 3))
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (2, 4))
    _, (ya, yb) = model.forward(state, x)
    assert jnp.allclose(ya, x @ a.W + a.b)
    assert jnp.allclose(yb, x @ b.W + b.b)


def test_parallel_names_children_positionally():
    with pax.seed(0):
        model = parallel(Linear(4, 8), Linear(4, 2))
    assert set(model.state()["params"]) == {"Linear_0", "Linear_1"}


def test_parallel_mixes_modules_and_callables():
    with pax.seed(0):
        model = parallel(Linear(4, 3), jax.nn.relu)
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (2, 4))
    _, (proj, activated) = model.forward(state, x)
    assert proj.shape == (2, 3)
    assert jnp.allclose(activated, jax.nn.relu(x))


def test_parallel_under_jit():
    with pax.seed(0):
        model = parallel(Linear(4, 8), Linear(4, 2))
    state = model.state()
    x = jnp.ones((3, 4))
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager[0], jitted[0])
    assert jnp.allclose(eager[1], jitted[1])
