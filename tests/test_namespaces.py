"""Namespace registry + Static pytree mechanism (contract §2)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from pax.namespaces import (
    Static,
    Tagged,
    is_namespace,
    namespace,
    spec_of,
)


def test_namespace_returns_tagger_marking_values():
    metrics = namespace("metrics_a")
    tagged = metrics(jnp.zeros(3))
    assert isinstance(tagged, Tagged)
    assert tagged.ns == "metrics_a"


def test_redeclaration_is_idempotent_for_same_spec():
    a = namespace("dup_ns", static=False, scoped=True)
    b = namespace("dup_ns", static=False, scoped=True)
    assert a.name == b.name == "dup_ns"


def test_redeclaration_with_conflicting_spec_raises():
    namespace("conflict_ns", static=False, scoped=True)
    with pytest.raises(ValueError, match="already declared"):
        namespace("conflict_ns", static=True, scoped=True)


def test_spec_of_unknown_namespace_raises():
    with pytest.raises(KeyError):
        spec_of("never_declared_ns")


def test_is_namespace():
    namespace("declared_ns")
    assert is_namespace("declared_ns")
    assert not is_namespace("undeclared_ns")


def test_builtins_have_expected_specs():
    assert spec_of("params").scoped and not spec_of("params").static
    assert spec_of("buffers").scoped and not spec_of("buffers").static
    assert not spec_of("flags").scoped and spec_of("flags").static


# -- Static pytree mechanism (§2.2) --------------------------------------------


def test_static_flattens_to_no_leaves_data_in_treedef():
    s = Static({"training": True, "depth": 4})
    leaves, treedef = jax.tree_util.tree_flatten(s)
    assert leaves == []
    restored = jax.tree_util.tree_unflatten(treedef, [])
    assert restored.data == {"training": True, "depth": 4}


def test_static_is_hashable_and_value_equal():
    assert Static({"a": 1, "b": 2}) == Static({"b": 2, "a": 1})
    assert hash(Static({"a": 1})) == hash(Static({"a": 1}))


def test_static_nested_dicts_are_hardened_and_hashable():
    s = Static({"cfg": {"training": True}})
    assert hash(s) == hash(Static({"cfg": {"training": True}}))


def test_static_participates_in_jit_cache_key():
    traces: list[int] = []

    @jax.jit
    def f(payload):
        flags = payload["flags"]
        traces.append(len(traces))
        # Concrete Python branch inside the trace — no ConcretizationError.
        scale = 2.0 if flags.data["training"] else 1.0
        return payload["x"] * scale

    x = jnp.ones(3)
    out_train = f({"x": x, "flags": Static({"training": True})})
    out_eval = f({"x": x, "flags": Static({"training": False})})
    f({"x": x, "flags": Static({"training": True})})  # reuses first compilation

    assert jnp.allclose(out_train, 2.0)
    assert jnp.allclose(out_eval, 1.0)
    assert len(traces) == 2  # one compile per flag combination, then cache hit
