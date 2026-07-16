"""Tests for pax.paths — glob matcher, select, freeze (contract §9)."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest

from pax.paths import freeze, match_path, select


def make_params() -> dict[str, Any]:
    """A realistic ``state['params']``-style nested dict."""
    return {
        "encoder": {"W": jnp.ones((2, 2)), "b": jnp.zeros(2)},
        "bn": {"g": jnp.ones(2), "b": jnp.zeros(2)},
        "blocks": {
            "0": {
                "attn": {"Wq": jnp.ones((2, 2)), "Wk": jnp.ones((2, 2))},
                "mlp": {"W": jnp.ones((2, 2))},
            },
            "1": {
                "attn": {"Wq": jnp.ones((2, 2)), "Wk": jnp.ones((2, 2))},
                "mlp": {"W": jnp.ones((2, 2))},
            },
        },
    }


def leaf_paths(tree: dict[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            paths |= leaf_paths(value, path)
        else:
            paths.add(path)
    return paths


# -- matcher --------------------------------------------------------------


def test_single_segment_wildcard_matches_one_segment() -> None:
    assert match_path("encoder.*", "encoder.W")
    assert match_path("*", "encoder")


def test_wildcard_does_not_span_multiple_segments() -> None:
    assert not match_path("blocks.*", "blocks.0.attn.Wq")
    assert not match_path("*", "encoder.W")


def test_multiple_wildcards() -> None:
    assert match_path("blocks.*.attn.*", "blocks.0.attn.Wq")
    assert match_path("blocks.*.attn.*", "blocks.1.attn.Wk")
    assert not match_path("blocks.*.attn.*", "blocks.0.mlp.W")


def test_exact_match() -> None:
    assert match_path("encoder.W", "encoder.W")
    assert not match_path("encoder.W", "encoder.b")


def test_no_match_on_segment_count_mismatch() -> None:
    assert not match_path("blocks.*.attn.*", "blocks.0.attn")
    assert not match_path("encoder", "encoder.W")


def test_trailing_wildcard() -> None:
    assert match_path("encoder.*", "encoder.W")
    assert match_path("encoder.*", "encoder.b")
    assert not match_path("encoder.*", "bn.g")


# -- select ---------------------------------------------------------------


def test_select_restricts_and_prunes() -> None:
    params = make_params()
    got = select(params, "blocks.*.attn.*")
    assert set(got) == {"blocks"}
    assert set(got["blocks"]) == {"0", "1"}
    assert set(got["blocks"]["0"]) == {"attn"}
    assert set(got["blocks"]["0"]["attn"]) == {"Wq", "Wk"}
    assert "mlp" not in got["blocks"]["0"]


def test_select_preserves_leaf_identity() -> None:
    params = make_params()
    got = select(params, "encoder.*")
    assert got["encoder"]["W"] is params["encoder"]["W"]
    assert got["encoder"]["b"] is params["encoder"]["b"]


def test_select_exact_path() -> None:
    params = make_params()
    got = select(params, "encoder.W")
    assert got == {"encoder": {"W": params["encoder"]["W"]}}


def test_select_no_match_is_empty() -> None:
    assert select(make_params(), "does.not.exist") == {}


def test_select_prunes_non_matching_top_level() -> None:
    got = select(make_params(), "bn.*")
    assert set(got) == {"bn"}


# -- freeze ---------------------------------------------------------------


def test_freeze_shape_matches_tree() -> None:
    params = make_params()
    mask = freeze(params, "encoder.*")
    assert leaf_paths(mask) == leaf_paths(params)


def test_freeze_marks_matching_leaves_true() -> None:
    params = make_params()
    mask = freeze(params, "blocks.*.attn.*")
    assert mask["blocks"]["0"]["attn"]["Wq"] is True
    assert mask["blocks"]["1"]["attn"]["Wk"] is True
    assert mask["blocks"]["0"]["mlp"]["W"] is False
    assert mask["encoder"]["W"] is False


def test_freeze_leaves_are_plain_bools() -> None:
    mask = freeze(make_params(), "encoder.W")
    flat = [
        v
        for path in leaf_paths(mask)
        for v in [_lookup(mask, path)]
    ]
    assert all(isinstance(v, bool) for v in flat)


def test_freeze_all_false_when_no_match() -> None:
    mask = freeze(make_params(), "nope.*")
    assert not any(_lookup(mask, p) for p in leaf_paths(mask))


def _lookup(tree: dict[str, Any], path: str) -> Any:
    node: Any = tree
    for key in path.split("."):
        node = node[key]
    return node


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
