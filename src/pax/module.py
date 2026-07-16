"""The `Module` base class: dual-mode attribute access + pure `forward`.

A `Module` subclass has two jobs (contract §0):

1. **Declare/initialize state** in `__init__` — assignments to `self` are
   classified by value type and stored in per-instance registries, *not* as real
   instance attributes.
2. **Route state during the forward pass** in `__call__` — reads and writes to
   `self` are redirected to the *bound* state slice for the current `forward`.

Because params/buffers live in `_init_values` rather than as real attributes,
normal attribute lookup misses and routes through `__getattr__`, which reads the
bound state during `forward` and the init value otherwise. `forward` is a pure
function of `(state, x)`: between calls the module holds no traced arrays, so
every JAX transform works on it with no wrappers (contract §4).
"""

from __future__ import annotations

from typing import Any

import jax

from ._types import PRNGKey
from .namespaces import Static, Tagged, is_namespace, spec_of
from .random import _next_module_key, _use_key

# Namespace slices are nested dicts of arrays (scoped) or flat dicts (global).
Slice = dict[str, Any]


class Module:
    """Base class for all Pax modules (contract §3–§5)."""

    _registry: dict[str, str]
    _init_values: dict[str, Any]
    _children: dict[str, Module]
    _key: PRNGKey
    _bound: list[dict[str, Slice]]
    _writes: list[dict[str, Slice]]
    _ctor: tuple[tuple[Any, ...], dict[str, Any]]

    def __new__(cls, *args: Any, **kwargs: Any) -> Module:
        obj = object.__new__(cls)
        setup = object.__setattr__
        setup(obj, "_registry", {})
        setup(obj, "_init_values", {})
        setup(obj, "_children", {})
        setup(obj, "_bound", [])
        setup(obj, "_writes", [])
        setup(obj, "_key", _next_module_key())
        setup(obj, "_ctor", (args, kwargs))
        return obj

    # -- initialization & attribute routing (contract §5.2, §3) --------------

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if self._bound:
            self._record_write(name, value)
            return
        if isinstance(value, Module):
            self._assign_child(name, value)
        elif isinstance(value, Tagged):
            self._assign_registered(name, value.ns, value.value)
        elif isinstance(value, jax.Array):
            self._assign_registered(name, "params", value)
        elif hasattr(value, "__array__"):
            raise TypeError(
                f"{name!r} is a {type(value).__name__}, not a jax.Array; assign a "
                f"JAX array (jnp.* / jax.random.*, seeded via self.key()) for a "
                f"weight, or wrap it with a namespace tagger. Plain config "
                f"(int/float/str/tuple) is stored as-is."
            )
        else:
            object.__setattr__(self, name, value)

    def _assign_child(self, name: str, child: Module) -> None:
        for other, existing in self._children.items():
            if existing is child and other != name:
                raise ValueError(
                    f"module instance is already assigned as {other!r}; assigning "
                    f"the same instance to {name!r} would alias its state. Tie "
                    f"weights via a global namespace (e.g. shared), not instance "
                    f"reuse; repeat a layer with pax.repeat."
                )
        if name in self._registry:
            raise ValueError(
                f"{name!r} is already a {self._registry[name]!r} attribute; "
                f"cannot also assign it as a child module."
            )
        self._children[name] = child

    def _assign_registered(self, name: str, ns: str, value: Any) -> None:
        if name in self._children:
            raise ValueError(
                f"{name!r} is already a child module; cannot also assign it as a "
                f"{ns!r} attribute."
            )
        self._registry[name] = ns
        self._init_values[name] = value

    def _record_write(self, name: str, value: Any) -> None:
        """Forward-time write: record under the attr's registered namespace.

        Writing to a name never registered at init raises (contract D3):
        silently dropping a state write is a multi-hour failure.
        """
        ns = self._registry.get(name)
        if ns is None:
            raise AttributeError(
                f"{type(self).__name__}.forward assigned to {name!r}, which was "
                f"never declared in __init__; forward may only write to registered "
                f"params/buffers (contract D3)"
            )
        self._writes[-1].setdefault(ns, {})[name] = value

    def __getattr__(self, name: str) -> Any:
        # Internal slots are set via object.__setattr__ and resolve normally;
        # anything underscore-prefixed reaching here is genuinely absent.
        if name.startswith("_"):
            raise AttributeError(name)
        children = object.__getattribute__(self, "_children")
        if name in children:
            return children[name]
        registry = object.__getattribute__(self, "_registry")
        if name in registry:
            return self._read_registered(name, registry[name])
        if is_namespace(name):
            return self._namespace_view(name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    def _read_registered(self, name: str, ns: str) -> Any:
        """Rule 2: a registered attr reads from bound state, else its init value."""
        if not self._bound:
            return self._init_values[name]
        writes = self._writes[-1].get(ns)
        if writes is not None and name in writes:
            return writes[name]
        slice_ = self._bound[-1].get(ns)
        data = slice_.data if isinstance(slice_, Static) else slice_
        if data is not None and name in data:
            return data[name]
        raise AttributeError(
            f"state passed to forward has no {name!r} under namespace {ns!r}"
        )

    def _namespace_view(self, ns: str) -> _NamespaceView:
        """Rule 3: `self.<ns>` is an accessor over that namespace's bound dict.

        Enables `self.flags.training` and tied-weight reads `self.shared.embed`.
        """
        if not self._bound:
            raise AttributeError(
                f"namespace {ns!r} is only accessible during forward, not at init"
            )
        slice_ = self._bound[-1].get(ns)
        data = slice_.data if isinstance(slice_, Static) else (slice_ or {})
        writes = self._writes[-1].get(ns, {})
        return _NamespaceView({**data, **writes})

    # -- public API (contract §4, §6) ----------------------------------------

    def key(self) -> PRNGKey:
        """Split and advance this module's init key. Init-time only (contract §6)."""
        self._key, sub = jax.random.split(self._key)
        return sub

    def state(self, *, key: PRNGKey | None = None) -> dict[str, Any]:
        """Extract a fresh state pytree (contract §4).

        With `key=`, re-materialize deterministically by replaying construction
        under that key (Option A escape hatch, §6) instead of returning the
        values fixed at the original construction.
        """
        if key is not None:
            args, kwargs = self._ctor
            if _has_module(args) or _has_module(kwargs):
                raise ValueError(
                    "state(key=...) cannot re-materialize a module built from "
                    "pre-existing child instances (combinators like sequential / "
                    "repeat / parallel, or any module taking a Module argument): "
                    "replaying construction would reuse those children, so their "
                    "params would not be redrawn. Rebuild the whole model under "
                    "`with pax.seed(...)` instead."
                )
            with _use_key(key):
                return type(self)(*args, **kwargs).state()
        used = self._used_namespaces()
        used.add("params")  # always present (contract §1)
        state: dict[str, Any] = {}
        for ns in sorted(used):
            spec = spec_of(ns)
            slice_ = (
                self._init_flat(ns, {})
                if not spec.scoped
                else self._init_scoped(ns)
            )
            state[ns] = Static(slice_) if spec.static else slice_
        return state

    def forward(self, state: dict[str, Any], x: Any) -> tuple[dict[str, Any], Any]:
        """The pure functional entry point `(state, x) -> (new_state, y)` (§4)."""
        self._bind(state)
        try:
            y = self(x)
            new_state = self._assemble(state)
        finally:
            self._unbind()
        return new_state, y

    # -- binding & collection (contract §5.3, §5.4) --------------------------

    def _bind(self, slices: dict[str, Slice]) -> None:
        """Top-down: push this module's frame and recurse into children (§5.3)."""
        frame: dict[str, Slice] = {}
        child_slices: dict[str, dict[str, Slice]] = {n: {} for n in self._children}
        for ns, raw in slices.items():
            spec = spec_of(ns)
            data = raw.data if isinstance(raw, Static) else raw
            frame[ns] = data
            if spec.scoped:
                for name in self._children:
                    if isinstance(data, dict) and name in data:
                        child_slices[name][ns] = data[name]
            else:
                for name in self._children:
                    child_slices[name][ns] = data
        self._bound.append(frame)
        self._writes.append({})
        for name, child in self._children.items():
            child._bind(child_slices[name])

    def _assemble(self, state: dict[str, Any]) -> dict[str, Any]:
        """Bottom-up: reassemble `new_state` with the input's namespaces (§5.4)."""
        new_state: dict[str, Any] = {}
        for ns, orig in state.items():
            spec = spec_of(ns)
            if spec.scoped:
                collected = self._collect_scoped(ns)
                slice_: Slice = collected if collected is not None else {}
            else:
                base = dict(orig.data) if isinstance(orig, Static) else dict(orig)
                self._collect_flat(ns, base)
                slice_ = base
            new_state[ns] = Static(slice_) if spec.static else slice_
        return new_state

    def _collect_scoped(self, ns: str) -> Slice | None:
        """This module's new slice for a scoped namespace, or None if it has none."""
        frame = self._bound[-1].get(ns)
        data = frame.data if isinstance(frame, Static) else frame
        writes = self._writes[-1].get(ns, {})
        out: Slice = {}
        for name, attr_ns in self._registry.items():
            if attr_ns != ns:
                continue
            if name in writes:
                out[name] = writes[name]
            elif isinstance(data, dict) and name in data:
                out[name] = data[name]
        for name, child in self._children.items():
            sub = child._collect_scoped(ns)
            if sub is not None:
                out[name] = sub
        return out or None

    def _collect_flat(self, ns: str, acc: Slice) -> None:
        """Overlay writes to a global namespace from this whole subtree (§5.4)."""
        acc.update(self._writes[-1].get(ns, {}))
        for child in self._children.values():
            child._collect_flat(ns, acc)

    def _unbind(self) -> None:
        """Pop the bound/write frames for this whole subtree (§5.4)."""
        self._bound.pop()
        self._writes.pop()
        for child in self._children.values():
            child._unbind()

    # -- init-state collection (contract §1, §4) -----------------------------

    def _used_namespaces(self) -> set[str]:
        used = set(self._registry.values())
        for child in self._children.values():
            used |= child._used_namespaces()
        return used

    def _init_scoped(self, ns: str) -> Slice:
        out: Slice = {
            name: self._init_values[name]
            for name, attr_ns in self._registry.items()
            if attr_ns == ns
        }
        for name, child in self._children.items():
            sub = child._init_scoped(ns)
            if sub:
                out[name] = sub
        return out

    def _init_flat(self, ns: str, acc: Slice) -> Slice:
        for name, attr_ns in self._registry.items():
            if attr_ns == ns:
                acc[name] = self._init_values[name]
        for child in self._children.values():
            child._init_flat(ns, acc)
        return acc

    def __call__(self, x: Any, /) -> Any:  # pragma: no cover - user-defined
        raise NotImplementedError(
            f"{type(self).__name__} must define __call__(self, x)"
        )


def _has_module(obj: Any) -> bool:
    """Whether a constructor argument (possibly nested) contains a `Module`."""
    if isinstance(obj, Module):
        return True
    if isinstance(obj, tuple | list):
        return any(_has_module(o) for o in obj)
    if isinstance(obj, dict):
        return any(_has_module(o) for o in obj.values())
    return False


class _NamespaceView:
    """Read-only attribute view over a bound namespace dict (contract §3 rule 3)."""

    __slots__ = ("_data",)

    def __init__(self, data: Slice) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(name)
