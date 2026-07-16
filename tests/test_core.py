"""Core Module mechanism: state layout, dual-mode access, forward purity,
static-under-jit, transforms, and the resolved decisions (contract §1–§6, §11)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

import pax
from pax.module import Module
from pax.namespaces import Static, buffer, flags, namespace

shared = namespace("shared", scoped=False)  # traced, global — tied weights


# -- hand-written modules (no dependency on Task A) ----------------------------


class Linear(Module):
    def __init__(self, i: int, o: int) -> None:
        self.W = jax.random.normal(self.key(), (i, o)) * 0.01
        self.b = jnp.zeros(o)

    def __call__(self, x):
        return x @ self.W + self.b


class BatchNorm(Module):
    """params + buffers + a static flag + write-collection — the richest case."""

    def __init__(self, d: int) -> None:
        self.g = jnp.ones(d)
        self.b = jnp.zeros(d)
        self.running_mean = buffer(jnp.zeros(d))
        self.running_var = buffer(jnp.ones(d))
        self.training = flags(True)

    def __call__(self, x):
        if self.flags.training:
            mean = x.mean(0)
            var = x.var(0)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean = self.running_mean
            var = self.running_var
        return self.g * (x - mean) / jnp.sqrt(var + 1e-5) + self.b


class Model(Module):
    def __init__(self, i: int, h: int) -> None:
        self.encoder = Linear(i, h)
        self.bn = BatchNorm(h)

    def __call__(self, x):
        return self.bn(self.encoder(x))


def build(cls, *args):
    with pax.seed(0):
        return cls(*args)


# -- §1 state layout -----------------------------------------------------------


def test_bare_array_defaults_to_params():
    state = build(Linear, 4, 8).state()
    assert set(state) == {"params"}
    assert set(state["params"]) == {"W", "b"}


def test_scoped_namespaces_mirror_the_module_tree_sparsely():
    state = build(Model, 4, 8).state()
    assert set(state["params"]) == {"encoder", "bn"}
    assert set(state["params"]["encoder"]) == {"W", "b"}
    assert set(state["params"]["bn"]) == {"g", "b"}
    # buffers only where written — encoder never declared one.
    assert set(state["buffers"]) == {"bn"}
    assert set(state["buffers"]["bn"]) == {"running_mean", "running_var"}


def test_static_namespace_is_wrapped_and_flat():
    state = build(Model, 4, 8).state()
    assert isinstance(state["flags"], Static)
    assert state["flags"].data == {"training": True}


def test_state_is_a_plain_jax_pytree():
    state = build(Model, 4, 8).state()
    leaves = jax.tree_util.tree_leaves(state)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)


def test_state_returns_a_fresh_tree_each_call():
    model = build(Model, 4, 8)
    a, b = model.state(), model.state()
    assert a is not b
    assert a["params"] is not b["params"]


# -- §3/§4 dual-mode attribute access -----------------------------------------


def test_unbound_read_returns_init_value():
    model = build(Linear, 4, 8)
    assert jnp.array_equal(model.W, model.state()["params"]["W"])


def test_forward_reads_bound_state_not_init_value():
    model = build(Linear, 4, 8)
    state = model.state()
    params = {**state["params"], "b": state["params"]["b"] + 5.0}
    perturbed = {**state, "params": params}
    _, y = model.forward(perturbed, jnp.zeros((1, 4)))
    assert jnp.allclose(y, 5.0)


def test_buffer_writes_appear_in_new_state():
    model = build(BatchNorm, 8)
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (16, 8))
    new_state, _ = model.forward(state, x)
    assert not jnp.array_equal(
        new_state["buffers"]["running_mean"], state["buffers"]["running_mean"]
    )


def test_eval_mode_reads_buffers_without_updating_them():
    model = build(BatchNorm, 8)
    state = model.state()
    eval_state: dict[str, Any] = {**state, "flags": Static({"training": False})}
    x = jax.random.normal(jax.random.key(1), (16, 8))
    new_state, _ = model.forward(eval_state, x)
    assert jnp.array_equal(
        new_state["buffers"]["running_mean"], eval_state["buffers"]["running_mean"]
    )


# -- §4 forward purity ---------------------------------------------------------


def test_module_holds_no_arrays_between_forwards():
    model = build(Model, 4, 8)
    state = model.state()
    model.forward(state, jnp.ones((2, 4)))
    assert model._bound == []
    assert model._writes == []
    assert model.bn._bound == []


def test_new_state_has_same_namespaces_as_input():
    model = build(Model, 4, 8)
    state = model.state()
    new_state, _ = model.forward(state, jnp.ones((2, 4)))
    assert set(new_state) == set(state)


def test_sequential_forwards_do_not_leak_state():
    model = build(BatchNorm, 8)
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (16, 8))
    s1, _ = model.forward(state, x)
    s2, _ = model.forward(state, x)  # same input state -> identical result
    assert jnp.array_equal(s1["buffers"]["running_mean"], s2["buffers"]["running_mean"])


# -- §2.2 static under jit -----------------------------------------------------


def test_jit_compiles_once_per_flag_combination():
    model = build(BatchNorm, 8)
    x = jax.random.normal(jax.random.key(1), (16, 8))
    traces: list[int] = []

    @jax.jit
    def step(state, x):
        traces.append(0)
        return model.forward(state, x)

    train_state = model.state()
    eval_state = {**train_state, "flags": Static({"training": False})}
    step(train_state, x)
    step(eval_state, x)
    step(train_state, x)  # cache hit
    assert len(traces) == 2


# -- transforms on forward (§4) ------------------------------------------------


def test_jit_matches_eager():
    model = build(Linear, 4, 8)
    state = model.state()
    x = jnp.ones((3, 4))
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted)


def test_vmap_over_batch():
    model = build(Linear, 4, 8)
    state = model.state()
    xs = jnp.ones((5, 4))
    batched = jax.vmap(model.forward, in_axes=(None, 0))
    _, ys = batched(state, xs)
    assert ys.shape == (5, 8)


def test_scan_threads_state():
    model = build(BatchNorm, 8)
    state = model.state()
    xs = jax.random.normal(jax.random.key(1), (4, 16, 8))
    final_state, ys = jax.lax.scan(model.forward, state, xs)
    assert ys.shape == (4, 16, 8)
    assert set(final_state) == set(state)


def test_grad_flows_through_params_only():
    model = build(Linear, 4, 8)
    state = model.state()
    x = jnp.ones((3, 4))

    def loss(params, state, x):
        _, y = model.forward({**state, "params": params}, x)
        return jnp.sum(y**2)

    grads = jax.grad(loss)(state["params"], state, x)
    assert set(grads) == {"W", "b"}
    assert grads["W"].shape == (4, 8)


def test_remat_matches_eager():
    model = build(Linear, 4, 8)
    state = model.state()
    x = jnp.ones((3, 4))
    _, eager = model.forward(state, x)
    _, remat = jax.remat(model.forward)(state, x)
    assert jnp.allclose(eager, remat)


# -- §3 resolved decisions -----------------------------------------------------


def test_d3_forward_write_to_unregistered_name_raises():
    class Bad(Module):
        def __init__(self) -> None:
            self.W = jnp.zeros(3)

        def __call__(self, x):
            self.surprise = x  # never registered at init
            return x

    model = build(Bad)
    with pytest.raises(AttributeError, match="never declared"):
        model.forward(model.state(), jnp.ones(3))


def test_d2_tied_weights_via_explicit_shared_accessor():
    class Encoder(Module):
        def __init__(self, vocab: int, d: int) -> None:
            self.embed = shared(jax.random.normal(self.key(), (vocab, d)))

        def __call__(self, x):
            return self.embed[x]

    class Decoder(Module):
        def __call__(self, h):
            return h @ self.shared.embed.T

    class TiedModel(Module):
        def __init__(self, vocab: int, d: int) -> None:
            self.enc = Encoder(vocab, d)
            self.dec = Decoder()

        def __call__(self, tokens):
            return self.dec(self.enc(tokens))

    model = build(TiedModel, 10, 4)
    state = model.state()
    # One shared copy, flat in the global namespace.
    assert set(state["shared"]) == {"embed"}
    assert state["shared"]["embed"].shape == (10, 4)
    _, logits = model.forward(state, jnp.array([1, 2, 3]))
    assert logits.shape == (3, 10)


def test_reentrant_forward_via_bound_stack():
    """A bound parent that calls child.forward pushes/pops a frame without
    clobbering its own binding (the `repeat`-over-`scan` canary, §5.1)."""

    class Inner(Module):
        def __init__(self, d: int) -> None:
            self.W = jax.random.normal(self.key(), (d, d)) * 0.01

        def __call__(self, x):
            return x @ self.W

    class Outer(Module):
        def __init__(self, d: int) -> None:
            self.bias = jnp.ones(d)
            self.inner = Inner(d)

        def __call__(self, x):
            # Re-enter forward on a child while Outer is itself bound.
            inner_state = {"params": self._bound[-1]["params"]["inner"]}
            _, h = self.inner.forward(inner_state, x)
            return h + self.bias  # reads Outer's still-intact binding

    model = build(Outer, 4)
    state = model.state()
    _, y = model.forward(state, jnp.ones((2, 4)))
    assert y.shape == (2, 4)
    # Outer's frame was restored after the nested forward popped.
    assert model._bound == []
    assert model.inner._bound == []


def test_undeclared_bare_name_raises_no_fallthrough():
    class NeedsShared(Module):
        def __call__(self, x):
            return self.embed  # bare name it never declared — no fallthrough

    model = build(NeedsShared)
    with pytest.raises(AttributeError):
        model.forward({"params": {}, "shared": {"embed": jnp.zeros(3)}}, None)
