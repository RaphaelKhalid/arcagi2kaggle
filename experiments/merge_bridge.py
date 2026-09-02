"""Proof-gated two-object bridge renderer for many-to-one scene rewrites."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from experiments.graph_lgg import role_predicates
    from experiments.guarded_roles import select_roles
    from experiments.object_deltas import (
        Connectivity, Object, extract_objects, normalize_grid,
    )
except ModuleNotFoundError:  # direct ``python experiments/merge_bridge.py``
    from graph_lgg import role_predicates
    from guarded_roles import select_roles
    from object_deltas import Connectivity, Object, extract_objects, normalize_grid


Grid = tuple[tuple[int, ...], ...]
Predicate = tuple[Any, ...]
Orientation = str


@dataclass(frozen=True)
class BridgeProgram:
    left_guard: frozenset[Predicate]
    right_guard: frozenset[Predicate]
    orientation: Orientation
    bridge_color: int
    connectivity: Connectivity = 4


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _segment(left: Object, right: Object, orientation: Orientation) -> frozenset[tuple[int, int]] | None:
    if len(left.cells) != 1 or len(right.cells) != 1:
        return None
    if orientation == "horizontal":
        if left.anchor[0] != right.anchor[0]:
            return None
        row = left.anchor[0]
        return frozenset(
            (row, col)
            for col in range(min(left.anchor[1], right.anchor[1]),
                             max(left.anchor[1], right.anchor[1]) + 1)
        )
    if orientation == "vertical":
        if left.anchor[1] != right.anchor[1]:
            return None
        col = left.anchor[1]
        return frozenset(
            (row, col)
            for row in range(min(left.anchor[0], right.anchor[0]),
                             max(left.anchor[0], right.anchor[0]) + 1)
        )
    raise ValueError("orientation must be horizontal or vertical")


def _pick_roles(
    grid: Grid,
    program: BridgeProgram,
) -> tuple[Object, Object] | None:
    objects = extract_objects(grid, connectivity=program.connectivity)
    left = select_roles(
        grid, program.left_guard, connectivity=program.connectivity
    )
    right = select_roles(
        grid, program.right_guard, connectivity=program.connectivity
    )
    if len(left) != 1 or len(right) != 1 or left[0] == right[0]:
        return None
    return objects[left[0]], objects[right[0]]


def execute_bridge(program: BridgeProgram, grid: Any) -> Grid | None:
    """Paint exactly the empty cells between two uniquely selected objects."""

    source = normalize_grid(grid)
    objects = _pick_roles(source, program)
    if objects is None:
        return None
    left, right = objects
    segment = _segment(left, right, program.orientation)
    if segment is None:
        return None
    source_cells = {
        (row, col) for obj in objects for row, col, _ in obj.cells
    }
    new_cells = segment - source_cells
    result = [list(row) for row in source]
    height, width = len(result), len(result[0])
    for row, col in new_cells:
        if not (0 <= row < height and 0 <= col < width):
            return None
        if result[row][col] != _background(source):
            return None
        result[row][col] = program.bridge_color
    return tuple(tuple(row) for row in result)


def _demo_bridge(
    source_value: Any,
    target_value: Any,
    connectivity: Connectivity,
) -> tuple[str, int, int, int] | None:
    source = normalize_grid(source_value)
    target = normalize_grid(target_value)
    if source != target and (len(source), len(source[0])) != (len(target), len(target[0])):
        return None
    source_objects = extract_objects(source, connectivity=connectivity)
    if len(source_objects) != 2:
        return None
    if any(len(obj.cells) != 1 for obj in source_objects):
        return None
    ordered = tuple(sorted(source_objects, key=lambda obj: obj.anchor))
    left, right = ordered
    if left.anchor[0] == right.anchor[0]:
        orientation = "horizontal"
    elif left.anchor[1] == right.anchor[1]:
        orientation = "vertical"
    else:
        return None
    segment = _segment(left, right, orientation)
    if segment is None:
        return None
    background = _background(source)
    changed: list[int] = []
    source_cells = {(row, col) for obj in ordered for row, col, _ in obj.cells}
    for row in range(len(source)):
        for col in range(len(source[0])):
            if source[row][col] != target[row][col]:
                if (row, col) not in segment or source[row][col] != background:
                    return None
                changed.append(target[row][col])
    expected_new = segment - source_cells
    if not changed or len(changed) != len(expected_new) or len(set(changed)) != 1:
        return None
    if any(target[row][col] == background for row, col in expected_new):
        return None
    if any(
        source[row][col] != target[row][col]
        for row, col in source_cells
    ):
        return None
    return orientation, changed[0], source_objects.index(left), source_objects.index(right)


def compile_bridge(
    task: Mapping[str, Any],
    *,
    connectivity: Connectivity = 4,
) -> BridgeProgram | None:
    """Compile a two-singleton bridge and replay it on every demonstration."""

    pairs = task.get("train", [])
    if not pairs:
        return None
    observations: list[tuple[str, int, frozenset[Predicate], frozenset[Predicate]]] = []
    try:
        for pair in pairs:
            demo = _demo_bridge(pair["input"], pair["output"], connectivity)
            if demo is None:
                return None
            orientation, color, left_index, right_index = demo
            observations.append((
                orientation, color,
                role_predicates(pair["input"], left_index, connectivity=connectivity),
                role_predicates(pair["input"], right_index, connectivity=connectivity),
            ))
    except (IndexError, ValueError):
        return None
    if len({item[0] for item in observations}) != 1:
        return None
    if len({item[1] for item in observations}) != 1:
        return None
    left_guard = frozenset.intersection(*(item[2] for item in observations))
    right_guard = frozenset.intersection(*(item[3] for item in observations))
    if not left_guard or not right_guard:
        return None
    program = BridgeProgram(
        left_guard=left_guard,
        right_guard=right_guard,
        orientation=observations[0][0],
        bridge_color=observations[0][1],
        connectivity=connectivity,
    )
    for pair in pairs:
        if (len(select_roles(
                pair["input"], left_guard, connectivity=connectivity
            )) != 1
                or len(select_roles(
                    pair["input"], right_guard, connectivity=connectivity
                )) != 1
                or execute_bridge(program, pair["input"])
                != normalize_grid(pair["output"])):
            return None
    return program


def dataset_bridge_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, list[Any]] | None = None,
    *,
    connectivity: Connectivity = 4,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0, "compiled_tasks": 0,
        "candidate_outputs": 0, "correct_outputs": 0,
    })
    for task_id, task in challenges.items():
        summary["tasks"] += 1
        program = compile_bridge(task, connectivity=connectivity)
        if program is None:
            continue
        summary["compiled_tasks"] += 1
        for index, test in enumerate(task.get("test", [])):
            prediction = execute_bridge(program, test["input"])
            if prediction is None:
                continue
            summary["candidate_outputs"] += 1
            if solutions is not None and task_id in solutions:
                summary["correct_outputs"] += int(
                    prediction == normalize_grid(solutions[task_id][index])
                )
    return dict(summary)


if __name__ == "__main__":
    task = {"train": [{
        "input": [[0, 1, 0, 1, 0]],
        "output": [[0, 1, 3, 1, 0]],
    }]}
    assert compile_bridge(task) is not None
    print("merge_bridge selftest: PASS")
