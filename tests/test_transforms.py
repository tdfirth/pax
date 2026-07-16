"""The transform matrix on a real *composed* model (contract §4).

Every transform in the §4 list is exercised end to end on models built from the
actual layers and combinators — an MLP (``sequential`` of ``Linear`` + ``relu``)
and a transformer block (``LayerNorm`` + ``Attention`` + MLP) — not on a single
hand-written layer. Each test asserts a real invariant (values equal to eager,
shapes, gradient structure), never merely "it ran".

Covered: ``jax.jit``, ``jax.vmap`` (``in_axes=(None, 0)`` with matching
``out_axes`` so ``new_state`` is not spuriously batched), ``jax.lax.scan``,
``jax.grad`` (params-only via ``{**state, 'params': p}``), ``jax.remat``, plus
``jax.lax.cond`` and ``jax.lax.fori_loop`` wrapped around ``forward``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

import pax
from pax.layers import Attention, LayerNorm, Linear
from pax.module import Module

State = dict[str, Any]


def mlp(d: int) -> Module:
    """Shape-preserving ReLU MLP ``d -> d``, so it also fits scan/fori carries."""
    return pax.sequential(Linear(d, d), jax.nn.relu, Linear(d, d))


class Block(Module):
    """A pre-norm transformer block — a genuinely composed, stateful model."""

    def __init__(self, d: int, heads: int, hidden: int) -> None:
        self.ln1 = LayerNorm(d)
        self.attn = Attention(d, heads)
        self.ln2 = LayerNorm(d)
        self.ff = pax.sequential(Linear(d, hidden), jax.nn.relu, Linear(hidden, d))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


def build(fn, *args) -> Module:
    with pax.seed(0):
        return fn(*args)


# -- jit -----------------------------------------------------------------------


def test_jit_matches_eager_on_composed_mlp():
    model = build(mlp, 8)
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (5, 8))
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted, atol=1e-5)


def test_jit_matches_eager_on_transformer_block():
    with pax.seed(0):
        model = Block(16, 4, 32)
    state = model.state()
    x = jax.random.normal(jax.random.key(1), (7, 16))
    _, eager = model.forward(state, x)
    _, jitted = jax.jit(model.forward)(state, x)
    assert jnp.allclose(eager, jitted, atol=1e-5)


# -- vmap ----------------------------------------------------------------------


def test_vmap_batches_x_only_and_keeps_new_state_unbatched():
    model = build(mlp, 8)
    state = model.state()
    xs = jax.random.normal(jax.random.key(1), (5, 8))

    batched = jax.vmap(model.forward, in_axes=(None, 0), out_axes=(None, 0))
    new_state, ys = batched(state, xs)

    assert ys.shape == (5, 8)
    # out_axes=None on new_state requires it to be batch-invariant; a pure model
    # returns its params unchanged, so this both type-checks and equals input.
    assert jnp.array_equal(
        new_state["params"]["Linear_0"]["W"], state["params"]["Linear_0"]["W"]
    )
    # Each mapped row equals the eager single-example forward.
    _, y0 = model.forward(state, xs[0])
    assert jnp.allclose(ys[0], y0, atol=1e-5)


# -- scan ----------------------------------------------------------------------


def test_scan_threads_state_and_matches_manual_loop():
    model = build(mlp, 8)
    state = model.state()
    xs = jax.random.normal(jax.random.key(2), (6, 8))

    final_state, ys = jax.lax.scan(model.forward, state, xs)

    assert ys.shape == (6, 8)
    assert set(final_state) == set(state)
    # forward is params-only here, so scan's carry is unchanged.
    assert jnp.array_equal(
        final_state["params"]["Linear_0"]["W"], state["params"]["Linear_0"]["W"]
    )
    manual = jnp.stack([model.forward(state, xs[i])[1] for i in range(xs.shape[0])])
    assert jnp.allclose(ys, manual, atol=1e-5)


# -- grad ----------------------------------------------------------------------


def test_grad_is_params_only_and_matches_param_structure():
    model = build(mlp, 8)
    state = model.state()
    x = jax.random.normal(jax.random.key(3), (4, 8))

    def loss(params: State, state: State, x: jax.Array) -> jax.Array:
        _, y = model.forward({**state, "params": params}, x)
        return jnp.sum(y**2)

    grads = jax.grad(loss)(state["params"], state, x)

    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(
        state["params"]
    )
    assert set(grads) == {"Linear_0", "Linear_2"}
    for leaf in jax.tree_util.tree_leaves(grads):
        assert jnp.all(jnp.isfinite(leaf))
    assert any(jnp.any(leaf != 0.0) for leaf in jax.tree_util.tree_leaves(grads))


def test_grad_flows_into_transformer_block_params():
    with pax.seed(0):
        model = Block(16, 4, 32)
    state = model.state()
    x = jax.random.normal(jax.random.key(3), (7, 16))

    def loss(params: State, state: State, x: jax.Array) -> jax.Array:
        _, y = model.forward({**state, "params": params}, x)
        return jnp.sum(y**2)

    grads = jax.grad(loss)(state["params"], state, x)
    assert jax.tree_util.tree_structure(grads) == jax.tree_util.tree_structure(
        state["params"]
    )
    assert jnp.any(grads["attn"]["Wq"] != 0.0)


# -- remat ---------------------------------------------------------------------


def test_remat_matches_eager():
    model = build(mlp, 8)
    state = model.state()
    x = jax.random.normal(jax.random.key(4), (5, 8))
    _, eager = model.forward(state, x)
    _, remat = jax.remat(model.forward)(state, x)
    assert jnp.allclose(eager, remat, atol=1e-5)


def test_remat_gradients_match_plain_gradients():
    model = build(mlp, 8)
    state = model.state()
    x = jax.random.normal(jax.random.key(4), (5, 8))

    def loss_with(fwd):
        def loss(params: State) -> jax.Array:
            _, y = fwd({**state, "params": params}, x)
            return jnp.sum(y**2)

        return loss

    plain = jax.grad(loss_with(model.forward))(state["params"])
    remat = jax.grad(loss_with(jax.remat(model.forward)))(state["params"])
    for a, b in zip(
        jax.tree_util.tree_leaves(plain),
        jax.tree_util.tree_leaves(remat),
        strict=True,
    ):
        assert jnp.allclose(a, b, atol=1e-5)


# -- cond ----------------------------------------------------------------------


def test_cond_selects_branch_around_forward():
    model = build(mlp, 8)
    state = model.state()
    x = jax.random.normal(jax.random.key(5), (3, 8))

    def run(pred: jax.Array) -> jax.Array:
        return jax.lax.cond(
            pred,
            lambda: model.forward(state, x)[1],
            lambda: model.forward(state, jnp.zeros_like(x))[1],
        )

    _, on_x = model.forward(state, x)
    _, on_zero = model.forward(state, jnp.zeros_like(x))
    assert jnp.allclose(run(jnp.array(True)), on_x, atol=1e-5)
    assert jnp.allclose(run(jnp.array(False)), on_zero, atol=1e-5)


# -- fori_loop -----------------------------------------------------------------


def test_fori_loop_iterates_forward():
    model = build(mlp, 8)
    state = model.state()
    x0 = jax.random.normal(jax.random.key(6), (3, 8))
    n = 4

    def body(_: int, carry: tuple[State, jax.Array]) -> tuple[State, jax.Array]:
        state, x = carry
        return model.forward(state, x)

    final_state, y = jax.lax.fori_loop(0, n, body, (state, x0))

    manual = x0
    for _ in range(n):
        _, manual = model.forward(state, manual)
    assert jnp.allclose(y, manual, atol=1e-5)
    assert set(final_state) == set(state)


def test_fori_loop_threads_batchnorm_buffers_and_static_flags():
    from pax.layers import BatchNorm

    class Net(Module):
        def __init__(self, d: int) -> None:
            self.lin = Linear(d, d)
            self.bn = BatchNorm(d)

        def __call__(self, x: jax.Array) -> jax.Array:
            return self.bn(self.lin(x))

    with pax.seed(0):
        model = Net(8)
    state = model.state()
    x0 = jax.random.normal(jax.random.key(7), (16, 8))

    def body(_: int, carry: tuple[State, jax.Array]) -> tuple[State, jax.Array]:
        state, x = carry
        return model.forward(state, x)

    final_state, _ = jax.lax.fori_loop(0, 3, body, (state, x0))
    # Static flag rides in the treedef, so the carry structure is stable and the
    # buffers evolved across the loop.
    assert final_state["flags"] == state["flags"]
    assert not jnp.array_equal(
        final_state["buffers"]["bn"]["running_mean"],
        state["buffers"]["bn"]["running_mean"],
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
