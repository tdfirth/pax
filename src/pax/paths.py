"""Path / selective-operation utilities over namespace dicts (contract §9).

A scoped namespace (e.g. ``state['params']``) is a nested dict of arrays that
mirrors the module tree, so its keys form meaningful dot-paths
(``encoder.W``, ``blocks.0.attn.Wq``). This module provides a glob matcher over
those paths plus two selective operations built on it: :func:`select` (restrict a
tree to matching leaves) and :func:`freeze` (a boolean mask marking matching
leaves for an optax training loop).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

Tree = dict[str, Any]
"""A nested dict of leaves — one namespace's slice of the state pytree."""


def match_path(pattern: str, path: str) -> bool:
    """Match a dot-path ``path`` against a glob ``pattern``.

    Both are split on ``.`` into segments. ``*`` matches exactly ONE segment;
    any other segment matches its counterpart literally. The match is
    whole-path: pattern and path must have the **same number of segments**, so
    a pattern never matches a partial prefix or a longer path. There is no
    multi-segment wildcard.

    Examples::

        match_path("blocks.*.attn.*", "blocks.0.attn.Wq")  # True
        match_path("blocks.*.attn.*", "blocks.0.attn")      # False (3 != 4)
        match_path("encoder.W", "encoder.W")                # True
        match_path("encoder.*", "encoder.W")                # True
        match_path("*", "encoder")                          # True
    """
    return _match_segments(pattern.split("."), path.split("."))


def _match_segments(pattern: Sequence[str], path: Sequence[str]) -> bool:
    if len(pattern) != len(path):
        return False
    return all(p == "*" or p == s for p, s in zip(pattern, path, strict=True))


def select(tree: Tree, pattern: str) -> Tree:
    """Return the subtree of ``tree`` holding only leaves matching ``pattern``.

    The result is a pytree of the same nesting as ``tree`` restricted to the
    matching leaves; branches with no matching leaf are pruned entirely. Because
    the matcher is whole-path (:func:`match_path`), a leaf is kept only when its
    full dot-path has the same segment count as ``pattern``.

    Example::

        select(params, "blocks.*.attn.*")  # -> {'blocks': {'0': {'attn': {...}}}}
    """
    result: Tree = {}
    for path, leaf in _iter_leaves(tree):
        if match_path(pattern, path):
            result = _assign(result, path.split("."), leaf)
    return result


def freeze(tree: Tree, pattern: str) -> Tree:
    """Return a boolean mask over ``tree`` marking leaves matching ``pattern``.

    The mask is a pytree of the **same shape** as ``tree``: every leaf is
    replaced by ``True`` if its dot-path matches ``pattern`` (i.e. "frozen"),
    ``False`` otherwise. This representation is chosen because it composes
    directly with optax and JAX:

    - ``optax.masked(optax.set_to_zero(), freeze(params, pat))`` zeros the
      updates of the matched leaves, freezing them, while the rest train.
    - ``optax.multi_transform`` / ``jax.tree_util.tree_map`` can partition on it.

    It is a plain pytree (no custom nodes, no closures), so it round-trips
    through ``jax.tree_util`` and serialization unchanged.

    Example::

        mask = freeze(params, "embed.*")   # True on embed.*, False elsewhere
    """
    return _mask(tree, pattern, "")


def _mask(tree: Tree, pattern: str, prefix: str) -> Tree:
    out: Tree = {}
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        out[key] = (
            _mask(value, pattern, path)
            if isinstance(value, dict)
            else match_path(pattern, path)
        )
    return out


def _iter_leaves(tree: Tree, prefix: str = "") -> Iterator[tuple[str, Any]]:
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _iter_leaves(value, path)
        else:
            yield path, value


def _assign(tree: Tree, keys: Sequence[str], value: Any) -> Tree:
    head, *rest = keys
    if not rest:
        return {**tree, head: value}
    child = tree.get(head, {})
    return {**tree, head: _assign(child, rest, value)}
