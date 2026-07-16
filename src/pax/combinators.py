"""Combinators: sequential, repeat, parallel (contract §8).

Each is a *function* returning a `Module` instance whose children are the given
sub-modules. Bare callables (activations like `jax.nn.relu`) are detected via
`not isinstance(x, Module)` and invoked directly with no scope management; module
children are registered under positional `ClassName_N` names (§5.5) so the core's
`_bind` binds them top-down before `__call__` runs.

`repeat` shares one weight-tied copy of its layer and applies it `n` times via
`jax.lax.scan`, threading the layer's state through the scan carry. It re-enters
`child.forward` inside an already-bound parent, so it exercises the re-entrant
`_bound` stack (§5.1) — the canary for core re-entrancy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax

from .module import Module
from .namespaces import Static, spec_of

Step = str | Callable[[Any], Any]


def _register(
    module: Module, layers: tuple[Module | Callable[[Any], Any], ...]
) -> list[Step]:
    """Register module children under `ClassName_N`; return the ordered plan.

    A plan entry is a child's registered name (`str`) or a bare callable, in the
    order given. Positional index `N` is the layer's position (§5.5).
    """
    plan: list[Step] = []
    for i, layer in enumerate(layers):
        if isinstance(layer, Module):
            name = f"{type(layer).__name__}_{i}"
            setattr(module, name, layer)
            plan.append(name)
            continue
        plan.append(layer)
    return plan


def _apply(module: Module, step: Step, x: Any) -> Any:
    """Run one plan step: a bound child by name, or a bare callable directly."""
    if isinstance(step, str):
        return getattr(module, step)(x)
    return step(x)


class _Sequential(Module):
    """Thread `x` through each layer in order (contract §8)."""

    def __init__(self, layers: tuple[Module | Callable[[Any], Any], ...]) -> None:
        self.steps = _register(self, layers)

    def __call__(self, x: Any) -> Any:
        for step in self.steps:
            x = _apply(self, step, x)
        return x


class _Parallel(Module):
    """Apply each layer to the input; return a tuple of outputs (contract §8)."""

    def __init__(self, layers: tuple[Module | Callable[[Any], Any], ...]) -> None:
        self.steps = _register(self, layers)

    def __call__(self, x: Any) -> tuple[Any, ...]:
        return tuple(_apply(self, step, x) for step in self.steps)


class _Repeat(Module):
    """`n` weight-tied applications of one layer via `jax.lax.scan` (contract §8).

    The layer is registered once, so it holds a single param copy in state. Its
    scoped state slice is extracted from the parent's bound frame and threaded,
    together with the activation, through the scan carry; the final slice is
    written back into the child's bound frame so the core collects it into
    `new_state`.
    """

    def __init__(self, layer: Module, n: int) -> None:
        self.child_name = f"{type(layer).__name__}_0"
        setattr(self, self.child_name, layer)
        self.times = n

    def _child_state(self) -> dict[str, Any]:
        frame = self._bound[-1]
        state: dict[str, Any] = {}
        for ns, data in frame.items():
            spec = spec_of(ns)
            if spec.scoped:
                if isinstance(data, dict) and self.child_name in data:
                    state[ns] = data[self.child_name]
                continue
            state[ns] = Static(data) if spec.static else data
        return state

    def _writeback(self, child: Module, final: dict[str, Any]) -> None:
        frame = child._bound[-1]
        for ns, value in final.items():
            if not spec_of(ns).scoped:
                continue
            frame[ns] = value.data if isinstance(value, Static) else value

    def __call__(self, x: Any) -> Any:
        child = getattr(self, self.child_name)
        child_state = self._child_state()

        def step(carry: tuple[dict[str, Any], Any], _: Any) -> tuple[
            tuple[dict[str, Any], Any], None
        ]:
            state, h = carry
            new_state, y = child.forward(state, h)
            return (new_state, y), None

        (final_state, out), _ = jax.lax.scan(
            step, (child_state, x), xs=None, length=self.times
        )
        self._writeback(child, final_state)
        return out


def sequential(*layers: Module | Callable[[Any], Any]) -> Module:
    """Thread `x` through each layer in order (contract §8)."""
    return _Sequential(layers)


def repeat(layer: Module, n: int) -> Module:
    """`n` weight-tied applications of `layer` via `jax.lax.scan` (contract §8)."""
    return _Repeat(layer, n)


def parallel(*layers: Module | Callable[[Any], Any]) -> Module:
    """Apply each layer to the input; return a tuple of outputs (contract §8)."""
    return _Parallel(layers)
