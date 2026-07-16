"""BatchNorm: train/eval mode as a static flag, buffers as traced state.

Demonstrates the two things that make Pax's namespace design pay off:

- **Buffers evolve during training.** ``running_mean`` / ``running_var`` are a
  traced ``buffers`` namespace: forward returns updated copies, and we thread
  them through the loop exactly like params.
- **Mode is a static flag, so it partitions the compile cache.** ``flags`` rides
  in the pytree treedef, so ``jax.jit`` compiles one program per flag value and
  reuses it — flipping ``training`` True/False triggers exactly one recompile
  each, never per-step.

Run: ``uv run python examples/batchnorm.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import pax
from pax.layers import BatchNorm, Linear
from pax.module import Module
from pax.namespaces import Static

State = dict[str, object]


class Net(Module):
    """Linear -> BatchNorm -> Linear — params + buffers + a static flag."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int) -> None:
        self.fc1 = Linear(d_in, d_hidden)
        self.bn = BatchNorm(d_hidden)
        self.fc2 = Linear(d_hidden, d_out)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.relu(self.bn(self.fc1(x))))


def train_flag(state: State, training: bool) -> State:
    """Return `state` with the `training` flag set (a new static treedef)."""
    return {**state, "flags": Static({"training": training})}


def main() -> None:
    with pax.seed(0):
        model = Net(8, 32, 1)
    state = model.state()
    params = state["params"]
    buffers = state["buffers"]

    key = jax.random.key(1)
    kx, kw = jax.random.split(key)
    x = jax.random.normal(kx, (256, 8))
    y = jnp.sin(x @ jax.random.normal(kw, (8, 1)))

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)

    traces: list[str] = []

    def loss_fn(params, buffers, state, x, y):
        s = {**state, "params": params, "buffers": buffers}
        new_state, pred = model.forward(s, x)
        return jnp.mean((pred - y) ** 2), new_state["buffers"]

    @jax.jit
    def train_step(params, buffers, opt_state, state):
        traces.append("train")
        (loss, buffers), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, buffers, state, x, y
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, buffers, opt_state, loss

    train_state = train_flag(state, True)
    running_mean_start = buffers["bn"]["running_mean"]
    first_loss = 0.0
    for step in range(200):
        params, buffers, opt_state, loss = train_step(
            params, buffers, opt_state, train_state
        )
        if step == 0:
            first_loss = float(loss)
        if step % 50 == 0:
            print(f"train step {step:4d}  loss {float(loss):.4f}")

    moved = float(jnp.linalg.norm(buffers["bn"]["running_mean"] - running_mean_start))
    print(f"loss {first_loss:.4f} -> {float(loss):.4f}")
    print(f"running_mean moved by {moved:.4f} during training")
    assert float(loss) < first_loss
    assert moved > 0.0

    @jax.jit
    def eval_predict(params, buffers, state, x):
        traces.append("eval")
        _, pred = model.forward(
            {**state, "params": params, "buffers": buffers}, x
        )
        return pred

    eval_state = train_flag(state, False)
    train_recompiles = traces.count("train")
    eval_before = buffers["bn"]["running_mean"]
    for _ in range(3):
        eval_predict(params, buffers, eval_state, x)
    eval_after = buffers["bn"]["running_mean"]

    n_train_traces = traces.count("train")
    n_eval_traces = traces.count("eval")
    print(f"train step compiled {n_train_traces} time(s) across 200 steps")
    print(f"eval step compiled {n_eval_traces} time(s) across 3 calls")
    assert n_train_traces == 1, "train step should compile exactly once"
    assert n_eval_traces == 1, "eval flag value should compile exactly once"
    assert train_recompiles == 1
    assert jnp.array_equal(eval_before, eval_after), "eval must not write buffers"
    print("eval reads buffers without updating them; one compile per flag value")


if __name__ == "__main__":
    main()
