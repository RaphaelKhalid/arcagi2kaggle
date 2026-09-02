"""Bounded graph least-general-generalization primitives for ARC traces."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import sys
from typing import Any, Iterable

try:
    from experiments.object_correspondence import Correspondence, top_k_correspondences
    from experiments.object_deltas import (
        Connectivity, Object, extract_objects, normalize_grid,
    )
except ModuleNotFoundError:  # direct ``python experiments/graph_lgg.py``
    from object_correspondence import Correspondence, top_k_correspondences
    from object_deltas import Connectivity, Object, extract_objects, normalize_grid


Predicate = tuple[Any, ...]


@dataclass(frozen=True)
class ActionObservation:
    kind: str
    motion_axis: str | None
    shape_relation: str | None
    area_relation: str | None
    palette_relation: str | None
    source_guard: frozenset[Predicate]
    target_guard: frozenset[Predicate]


@dataclass(frozen=True)
class ActionSchema:
    """A least-general schema; ``None`` means the attribute was not invariant."""

    kind: str | None
    motion_axis: str | None
    shape_relation: str | None
    area_relation: str | None
    palette_relation: str | None
    source_guard: frozenset[Predicate]
    target_guard: frozenset[Predicate]


def _axis(displacement: tuple[int, int] | None) -> str | None:
    if displacement is None or displacement == (0, 0):
        return "none"
    if displacement[0] == 0:
        return "horizontal"
    if displacement[1] == 0:
        return "vertical"
    return "diagonal"


def _kind(left: Object, right: Object) -> str:
    if left.shape == right.shape and left.colored_shape == right.colored_shape:
        return "identity" if left.anchor == right.anchor else "move"
    if left.shape == right.shape and left.anchor == right.anchor:
        return "recolor"
    if left.shape == right.shape:
        return "move_recolor"
    return "transform"


def _relation(left: Object, right: Object) -> tuple[str, str, str]:
    shape = "same" if left.shape == right.shape else "changed"
    if len(left.cells) == len(right.cells):
        area = "same"
    elif len(right.cells) > len(left.cells):
        area = "expanded"
    else:
        area = "shrunk"
    left_palette = {color for _, _, color in left.cells}
    right_palette = {color for _, _, color in right.cells}
    palette = "same" if left_palette == right_palette else "changed"
    return shape, area, palette


def _bucket(value: int) -> int:
    return min(3, value)


def role_predicates(
    value: Any, index: int, *, connectivity: Connectivity = 4
) -> frozenset[Predicate]:
    """Return bounded, id-free predicates for one object in a scene graph."""

    grid = normalize_grid(value)
    objects = extract_objects(grid, connectivity=connectivity)
    if index < 0 or index >= len(objects):
        raise ValueError("object index out of range")
    obj = objects[index]
    height, width = len(grid), len(grid[0])
    rows = [row for row, _, _ in obj.cells]
    cols = [col for _, col, _ in obj.cells]
    predicates: set[Predicate] = {
        ("area", _bucket(len(obj.cells))),
        ("bbox_height", _bucket(max(rows) - min(rows) + 1)),
        ("bbox_width", _bucket(max(cols) - min(cols) + 1)),
        ("palette_size", _bucket(len({color for _, _, color in obj.cells}))),
        ("border_top", obj.anchor[0] == 0),
        ("border_left", obj.anchor[1] == 0),
        ("border_bottom", max(rows) == height - 1),
        ("border_right", max(cols) == width - 1),
    }
    left = right = above = below = same_row = same_col = 0
    for other_index, other in enumerate(objects):
        if other_index == index:
            continue
        dr = other.anchor[0] - obj.anchor[0]
        dc = other.anchor[1] - obj.anchor[1]
        left += dc < 0
        right += dc > 0
        above += dr < 0
        below += dr > 0
        same_row += dr == 0
        same_col += dc == 0
    predicates.update({
        ("left_count", _bucket(left)),
        ("right_count", _bucket(right)),
        ("above_count", _bucket(above)),
        ("below_count", _bucket(below)),
        ("same_row_count", _bucket(same_row)),
        ("same_col_count", _bucket(same_col)),
        ("scene_degree", _bucket(len(objects) - 1)),
    })
    return frozenset(predicates)


def observations_for_correspondence(
    source_grid: Any,
    target_grid: Any,
    correspondence: Correspondence,
    *,
    connectivity: Connectivity = 4,
) -> tuple[ActionObservation, ...]:
    source_objects = extract_objects(
        normalize_grid(source_grid), connectivity=connectivity
    )
    target_objects = extract_objects(
        normalize_grid(target_grid), connectivity=connectivity
    )
    if (any(i >= len(source_objects) or j >= len(target_objects)
            for i, j in correspondence.pairs)
            or any(i >= len(source_objects) for i in correspondence.unmatched_source)
            or any(j >= len(target_objects) for j in correspondence.unmatched_target)):
        raise ValueError("correspondence refers to an object outside its grids")
    result: list[ActionObservation] = []
    for source_index, target_index in correspondence.pairs:
        left, right = source_objects[source_index], target_objects[target_index]
        shape, area, palette = _relation(left, right)
        displacement = (
            right.anchor[0] - left.anchor[0],
            right.anchor[1] - left.anchor[1],
        )
        result.append(ActionObservation(
            kind=_kind(left, right),
            motion_axis=_axis(displacement),
            shape_relation=shape,
            area_relation=area,
            palette_relation=palette,
            source_guard=role_predicates(
                source_grid, source_index, connectivity=connectivity
            ),
            target_guard=role_predicates(
                target_grid, target_index, connectivity=connectivity
            ),
        ))
    for source_index in correspondence.unmatched_source:
        result.append(ActionObservation(
            kind="delete", motion_axis=None, shape_relation=None,
            area_relation=None, palette_relation=None,
            source_guard=role_predicates(
                source_grid, source_index, connectivity=connectivity
            ),
            target_guard=frozenset(),
        ))
    for target_index in correspondence.unmatched_target:
        result.append(ActionObservation(
            kind="add", motion_axis=None, shape_relation=None,
            area_relation=None, palette_relation=None,
            source_guard=frozenset(),
            target_guard=role_predicates(
                target_grid, target_index, connectivity=connectivity
            ),
        ))
    return tuple(sorted(result, key=repr))


def _common(values: Iterable[Any]) -> Any:
    values = tuple(values)
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def lgg_observations(
    traces: Iterable[Iterable[ActionObservation]],
) -> tuple[ActionSchema, ...] | None:
    """Anti-unify equally sized bounded traces by common predicates."""

    trace_list = [tuple(trace) for trace in traces]
    if not trace_list or any(len(trace) != len(trace_list[0]) for trace in trace_list):
        return None
    schemas: list[ActionSchema] = []
    for index in range(len(trace_list[0])):
        observations = [trace[index] for trace in trace_list]
        schemas.append(ActionSchema(
            kind=_common(item.kind for item in observations),
            motion_axis=_common(item.motion_axis for item in observations),
            shape_relation=_common(item.shape_relation for item in observations),
            area_relation=_common(item.area_relation for item in observations),
            palette_relation=_common(item.palette_relation for item in observations),
            source_guard=frozenset.intersection(*(
                item.source_guard for item in observations
            )),
            target_guard=frozenset.intersection(*(
                item.target_guard for item in observations
            )),
        ))
    return tuple(schemas)


def dataset_lgg_profile(
    challenges: dict[str, dict[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
    connectivity: Connectivity = 4,
) -> dict[str, int]:
    """Profile bounded first-choice trace anti-unification without solutions."""

    summary = Counter({
        "tasks": 0,
        "skipped_over_cap": 0,
        "equal_trace_tasks": 0,
        "lgg_tasks": 0,
        "fully_typed_lgg_tasks": 0,
        "lgg_with_source_guard": 0,
        "lgg_with_target_guard": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        traces: list[tuple[ActionObservation, ...]] = []
        try:
            for pair in task.get("train", []):
                candidates = top_k_correspondences(
                    pair["input"], pair["output"],
                    k=k, max_objects=max_objects, connectivity=connectivity,
                )
                if not candidates:
                    raise ValueError("no correspondence")
                traces.append(observations_for_correspondence(
                    pair["input"], pair["output"], candidates[0],
                    connectivity=connectivity,
                ))
        except ValueError:
            summary["skipped_over_cap"] += 1
            continue
        if len({len(trace) for trace in traces}) == 1:
            summary["equal_trace_tasks"] += 1
        schema = lgg_observations(traces)
        if schema is None:
            continue
        summary["lgg_tasks"] += 1
        if all(item.kind is not None for item in schema):
            summary["fully_typed_lgg_tasks"] += 1
        if any(item.source_guard for item in schema):
            summary["lgg_with_source_guard"] += 1
        if any(item.target_guard for item in schema):
            summary["lgg_with_target_guard"] += 1
    return dict(summary)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1], "r", encoding="utf-8") as handle:
            print(dataset_lgg_profile(json.load(handle)))
        raise SystemExit(0)
    source = [[0, 1, 0], [0, 0, 0]]
    target = [[0, 0, 1], [0, 0, 0]]
    correspondence = top_k_correspondences(source, target)[0]
    trace = observations_for_correspondence(source, target, correspondence)
    assert lgg_observations((trace, trace))
    print("graph_lgg selftest: PASS")
