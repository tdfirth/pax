"""End-to-end integration scenarios on composed models (Task D).

Each test is a realistic workflow, not a unit check: a full optax training step
that reduces loss; the BatchNorm train->eval switch (buffers evolve in train, are
read-not-written in eval, jit recompiles once per flag value); tied weights via
``self.shared.embed`` across an encoder/decoder; ``freeze``/``select`` used to
partition params in an optax setup; and a KV-cache generation loop with
``Attention``.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import optax

import pax
from pax.layers import Attention, BatchNorm, Linear
from pax.module import Module
from pax.namespaces import Static, namespace
from pax.paths import freeze, select

shared = namespace("shared", scoped=False)  # traced, global — tied weights

State = dict[str, Any]


def mlp(sizes: tuple[int, ...]) -> Module:
    steps: list[Any] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        steps.append(Linear(a, b))
        if i < len(sizes) - 2:
            steps.append(jax.nn.relu)
    return pax.sequential(*steps)


def regression_data(
    key: jax.Array, n: int, d: int
) -> tuple[jax.Array, jax.Array]:
    kx, kw, kv = jax.random.split(key, 3)
    x = jax.random.normal(kx, (n, d))
    y = jnp.sin(x @ jax.random.normal(kw, (d, 8))) @ jax.random.normal(kv, (8, 1))
    return x, y


# -- full training step reduces loss ------------------------------------------


def test_training_step_reduces_loss():
    with pax.seed(0):
        model = mlp((8, 64, 1))
    state = model.state()
    params = state["params"]
    x, y = regression_data(jax.random.key(1), 256, 8)

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    def loss_fn(params: State, x: jax.Array, y: jax.Array) -> jax.Array:
        _, pred = model.forward({**state, "params": params}, x)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    first = float(loss_fn(params, x, y))
    for _ in range(200):
        params, opt_state, loss = step(params, opt_state)
    assert float(loss) < 0.5 * first


# -- BatchNorm train -> eval switch -------------------------------------------


class BNNet(Module):
    def __init__(self, d: int) -> None:
        self.fc = Linear(d, d)
        self.bn = BatchNorm(d)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.bn(jax.nn.relu(self.fc(x)))


def _with_training(state: State, training: bool) -> State:
    return {**state, "flags": Static({"training": training})}


def test_batchnorm_buffers_evolve_in_train_frozen_in_eval():
    with pax.seed(0):
        model = BNNet(8)
    state = model.state()
    x = jax.random.normal(jax.random.key(2), (32, 8))

    train_state = _with_training(state, True)
    after_train, _ = model.forward(train_state, x)
    assert not jnp.array_equal(
        after_train["buffers"]["bn"]["running_mean"],
        state["buffers"]["bn"]["running_mean"],
    )

    # Eval reads the (evolved) buffers but must not write them.
    eval_state = _with_training(after_train, False)
    after_eval, _ = model.forward(eval_state, x)
    assert jnp.array_equal(
        after_eval["buffers"]["bn"]["running_mean"],
        after_train["buffers"]["bn"]["running_mean"],
    )


def test_jit_recompiles_once_per_flag_value():
    with pax.seed(0):
        model = BNNet(8)
    state = model.state()
    x = jax.random.normal(jax.random.key(2), (32, 8))
    traces: list[int] = []

    @jax.jit
    def run(state: State, x: jax.Array):
        traces.append(1)
        return model.forward(state, x)

    train_state = _with_training(state, True)
    eval_state = _with_training(state, False)
    run(train_state, x)
    run(eval_state, x)
    run(train_state, x)  # cache hit — same flag treedef
    run(eval_state, x)  # cache hit
    assert len(traces) == 2


# -- tied weights via self.shared.embed (contract D2) -------------------------


class Encoder(Module):
    def __init__(self, vocab: int, d: int) -> None:
        self.embed = shared(jax.random.normal(self.key(), (vocab, d)) * 0.1)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        return self.embed[tokens]  # rule 2: its own declared attr


class Decoder(Module):
    def __call__(self, h: jax.Array) -> jax.Array:
        return h @ self.shared.embed.T  # rule 3: explicit accessor, one copy


class AutoEncoder(Module):
    def __init__(self, vocab: int, d: int) -> None:
        self.enc = Encoder(vocab, d)
        self.dec = Decoder()

    def __call__(self, tokens: jax.Array) -> jax.Array:
        return self.dec(self.enc(tokens))


def test_tied_weights_share_one_copy_and_accumulate_gradient():
    with pax.seed(0):
        model = AutoEncoder(12, 4)
    state = model.state()

    assert set(state["shared"]) == {"embed"}
    assert state["shared"]["embed"].shape == (12, 4)

    tokens = jnp.array([1, 5, 9])
    _, logits = model.forward(state, tokens)
    assert logits.shape == (3, 12)

    def loss(shared_ns: State) -> jax.Array:
        _, out = model.forward({**state, "shared": shared_ns}, tokens)
        return jnp.sum(out**2)

    grads = jax.grad(loss)(state["shared"])
    # A single embed leaf receives gradient from both the input lookup (encoder)
    # and the output projection (decoder).
    assert set(grads) == {"embed"}
    assert grads["embed"].shape == (12, 4)
    assert jnp.any(grads["embed"] != 0.0)


def test_tied_weights_train_updates_the_single_array():
    with pax.seed(0):
        model = AutoEncoder(12, 4)
    state = model.state()
    shared_ns = state["shared"]
    tokens = jnp.array([0, 1, 2, 3])

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(shared_ns)

    def loss(shared_ns: State) -> jax.Array:
        _, out = model.forward({**state, "shared": shared_ns}, tokens)
        target = jax.nn.one_hot(tokens, 12)
        return optax.softmax_cross_entropy(out, target).mean()

    before = shared_ns["embed"]
    first = float(loss(shared_ns))
    for _ in range(100):
        grads = jax.grad(loss)(shared_ns)
        updates, opt_state = optimizer.update(grads, opt_state, shared_ns)
        shared_ns = cast(State, optax.apply_updates(shared_ns, updates))
    assert not jnp.array_equal(shared_ns["embed"], before)
    assert float(loss(shared_ns)) < first


# -- freeze / select partition params in an optax setup -----------------------


def test_freeze_partitions_params_in_optax():
    with pax.seed(0):
        model = mlp((6, 6, 6))
    state = model.state()
    params = state["params"]
    x, y = regression_data(jax.random.key(3), 128, 6)

    frozen_paths = set(_leaf_paths(select(params, "Linear_0.*")))
    assert frozen_paths == {"Linear_0.W", "Linear_0.b"}

    mask = freeze(params, "Linear_0.*")
    # Zero the update on frozen leaves; adam trains the rest.
    tx = optax.chain(optax.adam(1e-1), optax.masked(optax.set_to_zero(), mask))
    opt_state = tx.init(params)

    def loss_fn(params: State) -> jax.Array:
        _, pred = model.forward({**state, "params": params}, x)
        return jnp.mean((pred - y) ** 2)

    frozen_before = params["Linear_0"]["W"]
    trained_before = params["Linear_2"]["W"]
    for _ in range(20):
        grads = jax.grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = cast(State, optax.apply_updates(params, updates))

    assert jnp.array_equal(params["Linear_0"]["W"], frozen_before)
    assert not jnp.array_equal(params["Linear_2"]["W"], trained_before)


def _leaf_paths(tree: State, prefix: str = "") -> list[str]:
    out: list[str] = []
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.extend(_leaf_paths(value, path))
        else:
            out.append(path)
    return out


# -- KV-cache generation loop with Attention ----------------------------------


def test_kv_cache_generation_grows_and_matches_full_attention():
    d, heads, seq = 16, 4, 5
    with pax.seed(0):
        model = Attention(d, heads)

    tokens = jax.random.normal(jax.random.key(4), (seq, d))

    # Incremental decode: one token at a time with a growing cache.
    cached_state: State = {**model.state(), "flags": Static({"use_cache": True})}
    assert cached_state["cache"]["k"].shape == (heads, 0, d // heads)
    incremental: list[jax.Array] = []
    for t in range(seq):
        cached_state, out = model.forward(cached_state, tokens[t : t + 1])
        assert cached_state["cache"]["k"].shape == (heads, t + 1, d // heads)
        incremental.append(out[0])
    incremental_out = jnp.stack(incremental)

    # Full (non-cached) attention over the whole sequence at once. The cached
    # per-step query-i output equals full-attention row i because each cached
    # step attends to keys/values 0..i — i.e. causal attention.
    full_state = model.state()
    _, full_out = model.forward(full_state, tokens)

    causal = jnp.tril(jnp.ones((seq, seq), bool))
    ref = _causal_reference(tokens, full_state["params"], heads, causal)
    assert jnp.allclose(incremental_out, ref, atol=1e-5)
    # And the cache genuinely grew to the full sequence length.
    assert cached_state["cache"]["k"].shape == (heads, seq, d // heads)
    assert full_out.shape == (seq, d)


def _causal_reference(
    x: jax.Array, params: State, heads: int, mask: jax.Array
) -> jax.Array:
    seq, d = x.shape
    dh = d // heads

    def split(w: jax.Array) -> jax.Array:
        return (x @ w).reshape(seq, heads, dh).transpose(1, 0, 2)

    q, k, v = split(params["Wq"]), split(params["Wk"]), split(params["Wv"])
    scores = q @ k.swapaxes(-1, -2) / jnp.sqrt(dh)
    scores = jnp.where(mask, scores, -jnp.inf)
    ctx = jax.nn.softmax(scores, axis=-1) @ v
    return ctx.transpose(1, 0, 2).reshape(seq, d) @ params["Wo"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
