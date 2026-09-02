"""Two-role relational transport: move one object relative to another role."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import sys
from typing import Any, Mapping

try:
    from experiments.guarded_roles import select_roles
    from experiments.object_correspondence import Correspondence, top_k_correspondences
    from experiments.object_deltas import Object, extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/relational_transport.py``
    from guarded_roles import select_roles
    from object_correspondence import Correspondence, top_k_correspondences
    from object_deltas import Object, extract_objects, normalize_grid


Shape = tuple[tuple[int, int], ...]
ColoredShape = tuple[tuple[int, int, int], ...]
Grid = tuple[tuple[int, ...], ...]
Predicate = tuple[Any, ...]


@dataclass(frozen=True)
class TransportProgram:
    mover_guard: frozenset[Predicate]
    reference_guard: frozenset[Predicate]
    relative_offset: tuple[int, int]


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _paint(
    result: list[list[int]],
    cells: ColoredShape,
    anchor: tuple[int, int],
    background: int,
) -> bool:
    height, width = len(result), len(result[0])
    absolute: list[tuple[int, int, int]] = []
    for row, col, color in cells:
        target = anchor[0] + row, anchor[1] + col
        if not (0 <= target[0] < height and 0 <= target[1] < width):
            return False
        absolute.append((target[0], target[1], color))
    if any(result[row][col] != background for row, col, _ in absolute):
        return False
    for row, col, color in absolute:
        result[row][col] = color
    return True


def execute_transport(program: TransportProgram, grid: Any) -> Grid | None:
    source = normalize_grid(grid)
    objects = extract_objects(source)
    movers = select_roles(source, program.mover_guard)
    references = select_roles(source, program.reference_guard)
    if len(movers) != 1 or len(references) != 1 or movers[0] == references[0]:
        return None
    mover, reference = objects[movers[0]], objects[references[0]]
    result = [list(row) for row in source]
    background = _background(source)
    for row, col, _ in mover.cells:
        result[row][col] = background
    anchor = (
        reference.anchor[0] + program.relative_offset[0],
        reference.anchor[1] + program.relative_offset[1],
    )
    if not _paint(result, mover.colored_shape, anchor, background):
        return None
    return tuple(tuple(row) for row in result)


def _move_reference_pairs(
    source_grid: Any,
    target_grid: Any,
    correspondence: Correspondence,
) -> tuple[tuple[int, int, int, int], ...]:
    source = extract_objects(normalize_grid(source_grid))
    target = extract_objects(normalize_grid(target_grid))
    moved: list[tuple[int, int]] = []
    stationary: list[tuple[int, int]] = []
    for source_index, target_index in correspondence.pairs:
        left, right = source[source_index], target[target_index]
        if left.shape != right.shape or left.colored_shape != right.colored_shape:
            continue
        if left.anchor != right.anchor:
            moved.append((source_index, target_index))
        else:
            stationary.append((source_index, target_index))
    result: list[tuple[int, int, int, int]] = []
    for mover_source, mover_target in moved:
        for ref_source, ref_target in stationary:
            if mover_source == ref_source:
                continue
            offset = (
                target[mover_target].anchor[0] - target[ref_target].anchor[0],
                target[mover_target].anchor[1] - target[ref_target].anchor[1],
            )
            result.append((mover_source, ref_source, offset[0], offset[1]))
    return tuple(result)


def compile_transport(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> TransportProgram | None:
    pairs = task.get("train", [])
    if not pairs:
        return None
    trace_pairs: list[tuple[int, int, int, int]] = []
    mover_guards: list[frozenset[Predicate]] = []
    reference_guards: list[frozenset[Predicate]] = []
    try:
        for pair in pairs:
            correspondence = top_k_correspondences(
                pair["input"], pair["output"], k=k, max_objects=max_objects
            )[0]
            candidates = _move_reference_pairs(
                pair["input"], pair["output"], correspondence
            )
            if len(candidates) != 1:
                return None
            mover, reference, dr, dc = candidates[0]
            from experiments.graph_lgg import role_predicates
            trace_pairs.append(candidates[0])
            mover_guards.append(role_predicates(pair["input"], mover))
            reference_guards.append(role_predicates(pair["input"], reference))
    except (IndexError, ValueError):
        return None
    if len({(dr, dc) for _, _, dr, dc in trace_pairs}) != 1:
        return None
    mover_guard = frozenset.intersection(*mover_guards)
    reference_guard = frozenset.intersection(*reference_guards)
    if not mover_guard or not reference_guard:
        return None
    program = TransportProgram(
        mover_guard=mover_guard,
        reference_guard=reference_guard,
        relative_offset=(trace_pairs[0][2], trace_pairs[0][3]),
    )
    for pair in pairs:
        if (len(select_roles(pair["input"], mover_guard)) != 1
                or len(select_roles(pair["input"], reference_guard)) != 1
                or execute_transport(program, pair["input"])
                != normalize_grid(pair["output"])):
            return None
    return program


def dataset_transport_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, list[Any]] | None = None,
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "compiled_tasks": 0,
        "candidate_outputs": 0,
        "correct_outputs": 0,
    })
    for task_id, task in challenges.items():
        summary["tasks"] += 1
        program = compile_transport(task, k=k, max_objects=max_objects)
        if program is None:
            continue
        summary["compiled_tasks"] += 1
        for index, test in enumerate(task.get("test", [])):
            prediction = execute_transport(program, test["input"])
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
        "input": [[0, 1, 0, 2], [0, 0, 0, 0]],
        "output": [[0, 0, 1, 2], [0, 0, 0, 0]],
    }]}
    program = compile_transport(task)
    assert program is not None
    print("relational_transport selftest: PASS")
