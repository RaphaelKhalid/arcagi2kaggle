"""Typed action extraction over explicit object correspondences.

This module is intentionally an intermediate research seam.  It does not
claim to solve a task: it turns a correspondence into a canonical, palette-
aware action family and asks whether one family survives every demonstration.
That makes correspondence uncertainty measurable before adding an executor.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import sys
from typing import Any, Iterable, Mapping

try:
    from experiments.object_correspondence import (
        Correspondence,
        top_k_correspondences,
    )
    from experiments.object_deltas import Object, extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/correspondence_actions.py``
    from object_correspondence import Correspondence, top_k_correspondences
    from object_deltas import Object, extract_objects, normalize_grid


Shape = tuple[tuple[int, int], ...]
ActionFamily = tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class TypedAction:
    """One correspondence-local effect with no arbitrary object id in it."""

    kind: str
    source_shape: Shape | None
    target_shape: Shape | None
    source_palette_size: int
    target_palette_size: int
    displacement: tuple[int, int] | None


def _palette(obj: Object) -> frozenset[int]:
    return frozenset(color for _, _, color in obj.cells)


def _matched_kind(source: Object, target: Object) -> str:
    same_shape = source.shape == target.shape
    same_pixels = source.colored_shape == target.colored_shape
    same_anchor = source.anchor == target.anchor
    if same_shape and same_pixels and same_anchor:
        return "identity"
    if same_shape and same_pixels:
        return "move"
    if same_shape and same_anchor:
        return "recolor"
    if same_shape:
        return "move_recolor"
    return "transform"


def actions_for_correspondence(
    source_grid: Any,
    target_grid: Any,
    correspondence: Correspondence,
) -> tuple[TypedAction, ...]:
    """Convert one explicit matching into sorted typed local effects."""

    source = extract_objects(normalize_grid(source_grid))
    target = extract_objects(normalize_grid(target_grid))
    if (any(source_index >= len(source) or target_index >= len(target)
            for source_index, target_index in correspondence.pairs)
            or any(source_index >= len(source)
                   for source_index in correspondence.unmatched_source)
            or any(target_index >= len(target)
                   for target_index in correspondence.unmatched_target)):
        raise ValueError("correspondence refers to an object outside its grids")

    actions: list[TypedAction] = []
    for source_index, target_index in correspondence.pairs:
        left, right = source[source_index], target[target_index]
        displacement = (
            right.anchor[0] - left.anchor[0],
            right.anchor[1] - left.anchor[1],
        )
        actions.append(TypedAction(
            kind=_matched_kind(left, right),
            source_shape=left.shape,
            target_shape=right.shape,
            source_palette_size=len(_palette(left)),
            target_palette_size=len(_palette(right)),
            displacement=displacement,
        ))
    for source_index in correspondence.unmatched_source:
        left = source[source_index]
        actions.append(TypedAction(
            kind="delete",
            source_shape=left.shape,
            target_shape=None,
            source_palette_size=len(_palette(left)),
            target_palette_size=0,
            displacement=None,
        ))
    for target_index in correspondence.unmatched_target:
        right = target[target_index]
        actions.append(TypedAction(
            kind="add",
            source_shape=None,
            target_shape=right.shape,
            source_palette_size=0,
            target_palette_size=len(_palette(right)),
            displacement=None,
        ))
    return tuple(sorted(actions, key=_action_sort_key))


def _shape_key(shape: Shape | None) -> tuple[tuple[int, int], ...] | None:
    return shape


def _action_sort_key(action: TypedAction) -> tuple[Any, ...]:
    return (
        action.kind,
        _shape_key(action.source_shape) or (),
        _shape_key(action.target_shape) or (),
        action.source_palette_size,
        action.target_palette_size,
        action.displacement or (0, 0),
    )


def action_family(actions: Iterable[TypedAction]) -> ActionFamily:
    """Canonical family key; actual object indices and color ids are absent."""

    return tuple(
        (
            action.kind,
            action.source_shape,
            action.target_shape,
            action.source_palette_size,
            action.target_palette_size,
            action.displacement,
        )
        for action in sorted(actions, key=_action_sort_key)
    )


def correspondence_action_families(
    source_grid: Any,
    target_grid: Any,
    *,
    k: int = 4,
    max_objects: int = 10,
) -> tuple[ActionFamily, ...]:
    """Return deduplicated typed action families from the top-k matchings."""

    correspondences = top_k_correspondences(
        source_grid, target_grid, k=k, max_objects=max_objects
    )
    families: list[ActionFamily] = []
    seen: set[ActionFamily] = set()
    for correspondence in correspondences:
        family = action_family(actions_for_correspondence(
            source_grid, target_grid, correspondence
        ))
        if family not in seen:
            seen.add(family)
            families.append(family)
    return tuple(families)


def task_action_consensus(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, Any]:
    """Measure cross-demo family intersection and top-1 recovery.

    ``stable_families`` are exact, typed families that survive every demo.
    This is a proof precondition for a later executor, not proof of a complete
    task program: relation guards, color roles, and composition are still
    missing.
    """

    demo_families: list[tuple[ActionFamily, ...]] = []
    for pair in task.get("train", []):
        demo_families.append(correspondence_action_families(
            pair["input"], pair["output"], k=k, max_objects=max_objects
        ))
    if not demo_families:
        return {
            "n_demos": 0,
            "greedy_stable": False,
            "stable_families": (),
            "recovered_by_top_k": (),
            "skipped": False,
        }
    stable = set(demo_families[0])
    for families in demo_families[1:]:
        stable.intersection_update(families)
    greedy = demo_families[0][0] if demo_families[0] else None
    greedy_survives = greedy is not None and all(
        greedy in families for families in demo_families
    )
    recovered = tuple(sorted(
        stable - ({greedy} if greedy is not None else set()),
        key=repr,
    ))
    return {
        "n_demos": len(demo_families),
        "greedy_stable": bool(greedy_survives),
        "stable_families": tuple(sorted(stable, key=repr)),
        "recovered_by_top_k": recovered,
        "skipped": False,
    }


def dataset_action_consensus(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, int]:
    """Aggregate correspondence-action evidence without using solutions."""

    summary = Counter({
        "tasks": 0,
        "multi_demo_tasks": 0,
        "greedy_stable_tasks": 0,
        "top_k_recovered_tasks": 0,
        "stable_family_count": 0,
        "skipped_tasks": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        if len(task.get("train", [])) > 1:
            summary["multi_demo_tasks"] += 1
        try:
            result = task_action_consensus(
                task, k=k, max_objects=max_objects
            )
        except ValueError:
            summary["skipped_tasks"] += 1
            continue
        summary["stable_family_count"] += len(result["stable_families"])
        if result["greedy_stable"]:
            summary["greedy_stable_tasks"] += 1
        if result["recovered_by_top_k"]:
            summary["top_k_recovered_tasks"] += 1
    return dict(summary)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1], "r", encoding="utf-8") as handle:
            challenges = json.load(handle)
        print(dataset_action_consensus(challenges))
        raise SystemExit(0)
    families = correspondence_action_families(
        [[0, 1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0]],
    )
    assert families and families[0][0][0] == "move"
    print("correspondence_actions selftest: PASS", families[0])
