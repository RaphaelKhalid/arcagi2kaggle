"""Frame-condition and parallel-composition audits for object action traces.

The intended proof rule is a grid analogue of separation logic: independently
verified clauses may compose when their read/write footprints are disjoint and
the remaining cells satisfy the frame invariant.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

try:
    from experiments.graph_lgg import (
        ActionObservation,
        ActionSchema,
        lgg_observations,
        observations_for_correspondence,
    )
    from experiments.guarded_roles import select_roles
    from experiments.object_correspondence import Correspondence, top_k_correspondences
    from experiments.object_deltas import Object, extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/compositional_clauses.py``
    from graph_lgg import ActionObservation, ActionSchema, lgg_observations, observations_for_correspondence
    from guarded_roles import select_roles
    from object_correspondence import Correspondence, top_k_correspondences
    from object_deltas import Object, extract_objects, normalize_grid


Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ClauseFootprint:
    kind: str
    source_cells: frozenset[tuple[int, int]]
    target_cells: frozenset[tuple[int, int]]
    source_index: int | None
    target_index: int | None


def _cells(obj: Object) -> frozenset[tuple[int, int]]:
    return frozenset((row, col) for row, col, _ in obj.cells)


def clause_footprints(
    source_grid: Any,
    target_grid: Any,
    correspondence: Correspondence,
) -> tuple[ClauseFootprint, ...]:
    """Return one footprint per correspondence clause in canonical order."""

    source = extract_objects(normalize_grid(source_grid))
    target = extract_objects(normalize_grid(target_grid))
    if (any(i >= len(source) or j >= len(target)
            for i, j in correspondence.pairs)
            or any(i >= len(source) for i in correspondence.unmatched_source)
            or any(j >= len(target) for j in correspondence.unmatched_target)):
        raise ValueError("correspondence refers to an object outside its grids")
    clauses: list[ClauseFootprint] = []
    for source_index, target_index in correspondence.pairs:
        left, right = source[source_index], target[target_index]
        if left.shape == right.shape and left.colored_shape == right.colored_shape:
            kind = "identity" if left.anchor == right.anchor else "move"
        elif left.shape == right.shape and left.anchor == right.anchor:
            kind = "recolor"
        elif left.shape == right.shape:
            kind = "move_recolor"
        else:
            kind = "transform"
        clauses.append(ClauseFootprint(
            kind=kind,
            source_cells=_cells(left),
            target_cells=_cells(right),
            source_index=source_index,
            target_index=target_index,
        ))
    for source_index in correspondence.unmatched_source:
        clauses.append(ClauseFootprint(
            kind="delete", source_cells=_cells(source[source_index]),
            target_cells=frozenset(), source_index=source_index,
            target_index=None,
        ))
    for target_index in correspondence.unmatched_target:
        clauses.append(ClauseFootprint(
            kind="add", source_cells=frozenset(),
            target_cells=_cells(target[target_index]), source_index=None,
            target_index=target_index,
        ))
    return tuple(sorted(clauses, key=repr))


def _changed_cells(source: Grid, target: Grid) -> frozenset[tuple[int, int]]:
    if (len(source), len(source[0])) != (len(target), len(target[0])):
        return frozenset()
    return frozenset(
        (row, col)
        for row in range(len(source))
        for col in range(len(source[0]))
        if source[row][col] != target[row][col]
    )


def footprints_are_disjoint(footprints: Iterable[ClauseFootprint]) -> bool:
    source_seen: set[tuple[int, int]] = set()
    target_seen: set[tuple[int, int]] = set()
    for footprint in footprints:
        if source_seen.intersection(footprint.source_cells):
            return False
        if target_seen.intersection(footprint.target_cells):
            return False
        source_seen.update(footprint.source_cells)
        target_seen.update(footprint.target_cells)
    return True


def frame_condition_holds(
    source_grid: Any,
    target_grid: Any,
    footprints: Iterable[ClauseFootprint],
) -> bool:
    """Check that every changed cell lies inside an active clause footprint."""

    source, target = normalize_grid(source_grid), normalize_grid(target_grid)
    if (len(source), len(source[0])) != (len(target), len(target[0])):
        return False
    footprint_cells = frozenset().union(*(
        footprint.source_cells | footprint.target_cells
        for footprint in footprints
        if footprint.kind != "identity"
    ))
    return _changed_cells(source, target).issubset(footprint_cells)


def _unique_schema_roles(
    schema: ActionSchema,
    source_grid: Any,
    target_grid: Any,
) -> bool:
    if not schema.source_guard or not schema.target_guard:
        return False
    return (
        len(select_roles(source_grid, schema.source_guard)) == 1
        and len(select_roles(target_grid, schema.target_guard)) == 1
    )


def task_composition_profile(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, int | bool]:
    """Audit top-1 trace composition without accessing test solutions."""

    pairs = task.get("train", [])
    if not pairs:
        return {"bounded": False, "equal_trace": False, "lgg": False,
                "fully_typed": False, "unique_roles": False,
                "disjoint_frames": False, "active_clauses": 0}
    traces: list[tuple[ActionObservation, ...]] = []
    footprints: list[tuple[ClauseFootprint, ...]] = []
    try:
        for pair in pairs:
            correspondence = top_k_correspondences(
                pair["input"], pair["output"], k=k, max_objects=max_objects
            )[0]
            traces.append(observations_for_correspondence(
                pair["input"], pair["output"], correspondence
            ))
            footprints.append(clause_footprints(
                pair["input"], pair["output"], correspondence
            ))
    except (IndexError, ValueError):
        return {"bounded": False, "equal_trace": False, "lgg": False,
                "fully_typed": False, "unique_roles": False,
                "disjoint_frames": False, "active_clauses": 0}
    equal_trace = len({len(trace) for trace in traces}) == 1
    schema = lgg_observations(traces)
    if schema is None:
        return {"bounded": True, "equal_trace": equal_trace, "lgg": False,
                "fully_typed": False, "unique_roles": False,
                "disjoint_frames": False, "active_clauses": 0}
    fully_typed = all(item.kind is not None for item in schema)
    unique_roles = fully_typed and all(
        all(_unique_schema_roles(
            item, pair["input"], pair["output"]
        ) for item in schema)
        for pair in pairs
    )
    disjoint_frames = all(
        footprints_are_disjoint(footprint)
        and frame_condition_holds(pair["input"], pair["output"], footprint)
        for pair, footprint in zip(pairs, footprints)
    )
    active = sum(item.kind != "identity" for item in schema)
    return {
        "bounded": True,
        "equal_trace": equal_trace,
        "lgg": True,
        "fully_typed": fully_typed,
        "unique_roles": unique_roles,
        "disjoint_frames": disjoint_frames,
        "active_clauses": active,
    }


def dataset_composition_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "bounded": 0,
        "equal_trace": 0,
        "lgg": 0,
        "fully_typed": 0,
        "unique_roles": 0,
        "disjoint_frames": 0,
        "composable_proof_tasks": 0,
        "active_clauses": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        result = task_composition_profile(task, k=k, max_objects=max_objects)
        for key in ("bounded", "equal_trace", "lgg", "fully_typed",
                    "unique_roles", "disjoint_frames"):
            summary[key] += int(result[key])
        if result["fully_typed"] and result["unique_roles"] and result["disjoint_frames"]:
            summary["composable_proof_tasks"] += 1
        summary["active_clauses"] += int(result["active_clauses"])
    return dict(summary)


if __name__ == "__main__":
    source = [[0, 1, 0, 0, 2]]
    target = [[0, 0, 1, 0, 2]]
    correspondence = top_k_correspondences(source, target)[0]
    footprints = clause_footprints(source, target, correspondence)
    assert footprints_are_disjoint(footprints)
    assert frame_condition_holds(source, target, footprints)
    print("compositional_clauses selftest: PASS")
