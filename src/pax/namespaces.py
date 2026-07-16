"""Namespaces: the two-axis (static/traced × scoped/global) state partitioning.

A *namespace* is a top-level key in the state pytree (contract §2). Declaring one
registers its `(static, scoped)` spec in a process-global registry so that
`state()` and `forward()` know how to lay it out and how JAX should treat it.

`namespace(...)` returns a *tagger*: a callable that wraps a value in `Tagged`,
marking it for storage in that namespace when assigned to `self` in `__init__`.

Static namespaces are wrapped in `Static` (§2.2) so their contents ride in the
pytree's *treedef* (as `aux_data`) rather than as leaves — which makes JAX treat
them as part of the compile cache key and exposes them as concrete Python values
inside a trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from jax.tree_util import register_pytree_node_class

T = TypeVar("T")


@dataclass(frozen=True)
class NamespaceSpec:
    """The immutable `(static, scoped)` classification of a namespace."""

    name: str
    static: bool
    scoped: bool


_REGISTRY: dict[str, NamespaceSpec] = {}


class Tagged:
    """A value marked for storage in a specific namespace.

    Produced by a tagger (e.g. `buffer(x)`); consumed by `Module.__setattr__`,
    which reads `.ns` to route the value and `.value` to store it.
    """

    __slots__ = ("ns", "value")

    def __init__(self, ns: str, value: Any) -> None:
        self.ns = ns
        self.value = value


class Tagger:
    """Callable returned by `namespace()`; wraps a value in `Tagged`.

    Typed to return its input type `T`, not `Tagged`. This is deliberate: from a
    module author's perspective `self.running_mean = buffer(x)` makes
    `self.running_mean` behave as `x` (reads during `forward` yield the array).
    The `Tagged` wrapper is an init-time storage detail that never surfaces to
    user code, so the annotation reflects the *runtime read* type, not the
    intermediate wrapper.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, value: T) -> T:
        if isinstance(value, Tagged):
            raise TypeError(
                f"value is already tagged for namespace {value.ns!r}; do not "
                f"double-tag it for {self.name!r}"
            )
        return Tagged(self.name, value)  # ty: ignore[invalid-return-type]


def namespace(name: str, *, static: bool = False, scoped: bool = True) -> Tagger:
    """Declare a namespace and return its tagger (contract §2.1).

    Idempotent for an identical spec; raises on a conflicting redeclaration.
    """
    spec = NamespaceSpec(name, static, scoped)
    existing = _REGISTRY.get(name)
    if existing is None:
        _REGISTRY[name] = spec
    elif existing != spec:
        raise ValueError(
            f"namespace {name!r} already declared as {existing}, "
            f"cannot redeclare as {spec}"
        )
    return Tagger(name)


def spec_of(name: str) -> NamespaceSpec:
    """Look up a declared namespace's spec, or raise if it was never declared."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"namespace {name!r} was never declared via namespace()"
        ) from None


def is_namespace(name: str) -> bool:
    """Whether `name` is a declared namespace."""
    return name in _REGISTRY


def _harden(value: Any) -> Any:
    """Recursively make a static payload hashable and comparable.

    Dicts become `_HashableDict`; lists become tuples. Leaves must already be
    hashable (bool, int, str, tuple) — the natural constraint for "small set of
    possible states" config (contract §2.2).
    """
    if isinstance(value, dict):
        return _HashableDict({k: _harden(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_harden(v) for v in value)
    return value


class _HashableDict(dict):
    """A dict that is hashable by value, for use as pytree `aux_data`.

    Relies on its values being hashable (guaranteed by `_harden`). Never mutated
    after construction, so the cached hash stays valid.
    """

    __slots__ = ()

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(tuple(sorted(self.items())))


@register_pytree_node_class
class Static:
    """A static namespace's payload, carried in the treedef (contract §2.2).

    Flattens to *no leaves* with its data as `aux_data`, so under `jax.jit` the
    values become part of the compile cache key and appear as concrete Python
    values inside the trace.
    """

    __slots__ = ("data",)

    def __init__(self, data: dict) -> None:
        self.data = _harden(data)

    def tree_flatten(self) -> tuple[tuple[()], _HashableDict]:
        return (), self.data  # type: ignore[return-value]

    @classmethod
    def tree_unflatten(cls, aux: dict, children: tuple[()]) -> Static:
        return cls(aux)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Static) and self.data == other.data

    def __hash__(self) -> int:
        return hash(self.data)

    def __repr__(self) -> str:
        return f"Static({dict(self.data)!r})"


# Built-in namespaces (contract §2.1).
params = namespace("params")  # traced, scoped — the default for bare arrays
buffer = namespace("buffers")  # traced, scoped — e.g. BatchNorm running stats
flags = namespace("flags", static=True, scoped=False)  # static, global — mode flags
