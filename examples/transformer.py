"""A small transformer: repeated blocks, tied embeddings, KV-cache generation.

Three Pax idioms in one model:

- **Depth via ``repeat``.** A stack of identical blocks is one weight-tied layer
  applied ``n`` times through ``jax.lax.scan`` — one parameter copy in state.
- **Tied embeddings via the explicit ``self.shared.embed`` accessor** (contract
  D2). The token table lives once in the global ``shared`` namespace; the input
  lookup and the output projection both read that single array, so gradients
  accumulate into it. No module ever reads a bare ``self.embed`` it did not
  declare.
- **KV-cache generation.** ``Attention`` under ``flags.use_cache`` concatenates
  new keys/values onto a growing ``cache`` namespace, so decoding one token at a
  time is O(1) attention work per step instead of re-encoding the prefix.

Run: ``uv run python examples/transformer.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import pax
from pax.layers import Attention, LayerNorm, Linear
from pax.module import Module
from pax.namespaces import Static, namespace

shared = namespace("shared", scoped=False)  # traced, global — one tied copy

State = dict[str, object]


class Block(Module):
    """Pre-norm transformer block: attention + MLP, both residual."""

    def __init__(self, d: int, heads: int, hidden: int) -> None:
        self.ln1 = LayerNorm(d)
        self.attn = Attention(d, heads)
        self.ln2 = LayerNorm(d)
        self.ff = pax.sequential(Linear(d, hidden), jax.nn.relu, Linear(hidden, d))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class Transformer(Module):
    """Token -> logits with a `repeat`-stacked body and tied embeddings."""

    def __init__(
        self, vocab: int, d: int, heads: int, hidden: int, depth: int, maxlen: int
    ) -> None:
        self.embed = shared(jax.random.normal(self.key(), (vocab, d)) * 0.02)
        self.pos = jax.random.normal(self.key(), (maxlen, d)) * 0.02
        self.blocks = pax.repeat(Block(d, heads, hidden), depth)
        self.ln_f = LayerNorm(d)

    def __call__(self, tokens: jax.Array) -> jax.Array:
        n = tokens.shape[0]
        x = self.shared.embed[tokens] + self.pos[:n]
        x = self.blocks(x)
        x = self.ln_f(x)
        return x @ self.shared.embed.T  # tied output projection (D2 accessor)


def kv_cache_generation(
    d: int, heads: int, n_steps: int
) -> list[tuple[int, int]]:
    """Decode `n_steps` tokens through one cached Attention; report cache growth.

    Returns the `(cached_keys, cached_values)` sequence length after each step,
    demonstrating the O(1)-per-token growth of the `cache` namespace.
    """
    with pax.seed(1):
        attn = Attention(d, heads)
    state: State = {**attn.state(), "flags": Static({"use_cache": True})}
    sizes: list[tuple[int, int]] = []
    for step in range(n_steps):
        token = jax.random.normal(jax.random.key(step), (1, d))
        state, _ = attn.forward(state, token)
        cache = state["cache"]  # type: ignore[assignment]
        sizes.append((cache["k"].shape[1], cache["v"].shape[1]))
    return sizes


def main() -> None:
    with pax.seed(0):
        model = Transformer(vocab=32, d=16, heads=4, hidden=32, depth=3, maxlen=16)
    state = model.state()

    print("state namespaces:", sorted(state))
    print("tied embed copies in state:", list(state["shared"]))  # exactly one
    assert set(state["shared"]) == {"embed"}

    # The `repeat`-stacked body is one weight-tied copy, not `depth` copies.
    print("blocks param keys:", list(state["params"]["blocks"]))
    assert list(state["params"]["blocks"]) == ["Block_0"]

    tokens = jnp.array([3, 1, 4, 1, 5, 9])
    _, logits = model.forward(state, tokens)
    print("logits shape:", logits.shape)
    assert logits.shape == (tokens.shape[0], 32)

    jit_logits = jax.jit(model.forward)(state, tokens)[1]
    assert jnp.allclose(logits, jit_logits, atol=1e-4)
    print("jit(forward) matches eager")

    # Gradients accumulate into the single tied embedding array.
    def loss_fn(shared_ns, state, tokens):
        _, out = model.forward({**state, "shared": shared_ns}, tokens)
        return jnp.mean(out**2)

    grad_embed = jax.grad(loss_fn)(state["shared"], state, tokens)["embed"]
    print("tied-embed grad shape:", grad_embed.shape)
    assert grad_embed.shape == state["shared"]["embed"].shape
    assert jnp.any(grad_embed != 0.0)

    sizes = kv_cache_generation(d=16, heads=4, n_steps=5)
    print("KV-cache seq length after each decode step:", [k for k, _ in sizes])
    assert sizes == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]
    print("KV cache grows one position per generated token")


if __name__ == "__main__":
    main()
