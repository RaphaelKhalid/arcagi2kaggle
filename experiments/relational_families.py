"""Coarse relation-normalized action families for ARC research.

The exact correspondence/action key is intentionally too literal for many ARC
tasks.  This module quotients object-local effects into clipped scene-graph
predicates.  It is still a candidate-family audit, not an executor or a score.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import sys
from typing import Any, Mapping

try:
    from experiments.object_correspondence import (
        Correspondence,
        top_k_correspondences,
    )
    from experiments.object_deltas import Object, extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/relational_families.py``
    from object_correspondence import Correspondence, top_k_correspondences
    from object_deltas import Object, extract_objects, normalize_grid


def _clip(value: int, limit: int = 2) -> int:
    return min(limit, value)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _area_relation(left: Object, right: Object) -> str:
    if len(left.cells) == len(right.cells):
        return "same"
    return "expanded" if len(right.cells) > len(left.cells) else "shrunk"


def _context(objects: tuple[Object, ...], index: int) -> tuple[int, ...]:
    """Clipped directional/alignment counts; object ids and distances vanish."""

    current = objects[index]
    left = right = above = below = same_row = same_col = 0
    for other_index, other in enumerate(objects):
        if other_index == index:
            continue
        dr = other.anchor[0] - current.anchor[0]
        dc = other.anchor[1] - current.anchor[1]
        left += dc < 0
        right += dc > 0
        above += dr < 0
        below += dr > 0
        same_row += dr == 0
        same_col += dc == 0
    return tuple(_clip(value) for value in (
        left, right, above, below, same_row, same_col,
    ))


def _motion_relation(displacement: tuple[int, int] | None) -> str:
    if displacement is None or displacement == (0, 0):
        return "none"
    dr, dc = displacement
    if dr == 0:
        return "horizontal"
    if dc == 0:
        return "vertical"
    return "diagonal"


def _kind(left: Object, right: Object) -> str:
    same_shape = left.shape == right.shape
    same_pixels = left.colored_shape == right.colored_shape
    same_anchor = left.anchor == right.anchor
    if same_shape and same_pixels and same_anchor:
        return "identity"
    if same_shape and same_pixels:
        return "move"
    if same_shape and same_anchor:
        return "recolor"
    if same_shape:
        return "move_recolor"
    return "transform"


def _unmatched_schema(
    kind: str,
    obj: Object,
    context: tuple[int, ...],
) -> tuple[Any, ...]:
    return (
        kind,
        "none",
        "none",
        "none",
        "none",
        min(3, len(obj.cells)),
        min(3, len({color for _, _, color in obj.cells})),
        context,
    )


def relational_action_family(
    source_grid: Any,
    target_grid: Any,
    correspondence: Correspondence,
) -> tuple[tuple[Any, ...], ...]:
    """Create a coarse schema that is invariant to ids and exact distances."""

    source = extract_objects(normalize_grid(source_grid))
    target = extract_objects(normalize_grid(target_grid))
    if (any(i >= len(source) or j >= len(target)
            for i, j in correspondence.pairs)
            or any(i >= len(source) for i in correspondence.unmatched_source)
            or any(j >= len(target) for j in correspondence.unmatched_target)):
        raise ValueError("correspondence refers to an object outside its grids")

    schemas: list[tuple[Any, ...]] = []
    for source_index, target_index in correspondence.pairs:
        left, right = source[source_index], target[target_index]
        displacement = (
            right.anchor[0] - left.anchor[0],
            right.anchor[1] - left.anchor[1],
        )
        schemas.append((
            _kind(left, right),
            "same" if left.shape == right.shape else "changed",
            _area_relation(left, right),
            "same" if len({color for _, _, color in left.cells})
            == len({color for _, _, color in right.cells}) else "changed",
            _motion_relation(displacement),
            min(3, len(left.cells)),
            min(3, len(right.cells)),
            _context(source, source_index),
            _context(target, target_index),
        ))
    for index in correspondence.unmatched_source:
        schemas.append(_unmatched_schema(
            "delete", source[index], _context(source, index)
        ))
    for index in correspondence.unmatched_target:
        schemas.append(_unmatched_schema(
            "add", target[index], _context(target, index)
        ))
    return tuple(sorted(schemas, key=repr))


def relational_action_families(
    source_grid: Any,
    target_grid: Any,
    *,
    k: int = 4,
    max_objects: int = 10,
) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    families: list[tuple[tuple[Any, ...], ...]] = []
    seen: set[tuple[tuple[Any, ...], ...]] = set()
    for correspondence in top_k_correspondences(
        source_grid, target_grid, k=k, max_objects=max_objects
    ):
        family = relational_action_family(source_grid, target_grid, correspondence)
        if family not in seen:
            seen.add(family)
            families.append(family)
    return tuple(families)


def task_relational_consensus(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, Any]:
    per_demo = [relational_action_families(
        pair["input"], pair["output"], k=k, max_objects=max_objects
    ) for pair in task.get("train", [])]
    if not per_demo:
        return {"n_demos": 0, "stable_families": (),
                "greedy_stable": False, "top_k_recovered": (),
                "skipped": False}
    stable = set(per_demo[0])
    for families in per_demo[1:]:
        stable.intersection_update(families)
    greedy = per_demo[0][0] if per_demo[0] else None
    greedy_stable = greedy is not None and all(
        greedy in families for families in per_demo
    )
    recovered = tuple(sorted(
        stable - ({greedy} if greedy is not None else set()), key=repr
    ))
    return {
        "n_demos": len(per_demo),
        "stable_families": tuple(sorted(stable, key=repr)),
        "greedy_stable": bool(greedy_stable),
        "top_k_recovered": recovered,
        "skipped": False,
    }


def dataset_relational_consensus(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "greedy_stable_tasks": 0,
        "top_k_recovered_tasks": 0,
        "stable_family_count": 0,
        "skipped_tasks": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        try:
            result = task_relational_consensus(
                task, k=k, max_objects=max_objects
            )
        except ValueError:
            summary["skipped_tasks"] += 1
            continue
        summary["stable_family_count"] += len(result["stable_families"])
        summary["greedy_stable_tasks"] += int(result["greedy_stable"])
        summary["top_k_recovered_tasks"] += int(bool(result["top_k_recovered"]))
    return dict(summary)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1], "r", encoding="utf-8") as handle:
            print(dataset_relational_consensus(json.load(handle)))
    else:
        print(relational_action_families(
            [[0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0]],
        ))
