"""Train an MLP with optax — the canonical Pax training loop.

Shows the whole story in one screen: build a model under ``pax.seed``, extract a
plain state pytree, and thread it through a ``jax.jit``-compiled optax step. The
model never appears inside the traced step — only its pure ``forward`` and the
state pytree do — so there are no wrappers, no custom optimizer, no framework.

Run: ``uv run python examples/mlp.py`` — prints the loss decreasing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import optax

import pax
from pax.layers import Linear

Params = dict[str, Any]
State = dict[str, Any]
Step = pax.Module | Callable[[jax.Array], jax.Array]


def make_mlp(sizes: tuple[int, ...]) -> pax.Module:
    """A ReLU MLP: ``Linear -> relu -> ... -> Linear`` over the given widths."""
    steps: list[Step] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        steps.append(Linear(a, b))
        if i < len(sizes) - 2:
            steps.append(jax.nn.relu)
    return pax.sequential(*steps)


def synthetic_data(key: jax.Array, n: int, d: int) -> tuple[jax.Array, jax.Array]:
    """A fixed nonlinear target ``y = sin(Xw) . v`` — learnable by a small MLP."""
    kx, kw, kv = jax.random.split(key, 3)
    x = jax.random.normal(kx, (n, d))
    w = jax.random.normal(kw, (d, 16))
    v = jax.random.normal(kv, (16, 1))
    y = jnp.sin(x @ w) @ v
    return x, y


def main() -> None:
    with pax.seed(0):
        model = make_mlp((8, 64, 64, 1))
    state = model.state()
    params: Params = state["params"]

    x, y = synthetic_data(jax.random.key(1), n=512, d=8)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    def loss_fn(params: Params, state: State, x: jax.Array, y: jax.Array) -> jax.Array:
        _, pred = model.forward({**state, "params": params}, x)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def train_step(
        params: Params, opt_state: optax.OptState, state: State
    ) -> tuple[Params, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(params, state, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = cast(Params, optax.apply_updates(params, updates))
        return params, opt_state, loss

    losses: list[float] = []
    for step in range(300):
        params, opt_state, loss = train_step(params, opt_state, state)
        if step % 50 == 0:
            losses.append(float(loss))
            print(f"step {step:4d}  loss {float(loss):.4f}")

    print(f"loss {losses[0]:.4f} -> {float(loss):.4f}")
    assert float(loss) < losses[0], "training did not reduce the loss"


if __name__ == "__main__":
    main()
