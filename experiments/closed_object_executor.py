"""Proof-gated executor for a deliberately closed object-action subset.

Only effects whose target cells are fully determined by a source fingerprint
or a fixed target anchor are executable.  Ambiguous role matches, shape
transforms, collisions, and out-of-bounds writes are rejected.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import sys
from typing import Any, Mapping

try:
    from experiments.object_correspondence import Correspondence, top_k_correspondences
    from experiments.object_deltas import Object, extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/closed_object_executor.py``
    from object_correspondence import Correspondence, top_k_correspondences
    from object_deltas import Object, extract_objects, normalize_grid


Shape = tuple[tuple[int, int], ...]
ColoredShape = tuple[tuple[int, int, int], ...]
Grid = tuple[tuple[int, ...], ...]


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


@dataclass(frozen=True)
class Operation:
    kind: str
    source_shape: Shape | None = None
    source_colored_shape: ColoredShape | None = None
    target_shape: Shape | None = None
    target_colored_shape: ColoredShape | None = None
    displacement: tuple[int, int] | None = None
    target_anchor: tuple[int, int] | None = None


@dataclass(frozen=True)
class ClosedProgram:
    operations: tuple[Operation, ...]


def _in_bounds(row: int, col: int, height: int, width: int) -> bool:
    return 0 <= row < height and 0 <= col < width


def _paint_cells(
    result: list[list[int]],
    cells: ColoredShape,
    anchor: tuple[int, int],
    background: int,
    *,
    occupied: set[tuple[int, int]],
) -> bool:
    height, width = len(result), len(result[0])
    absolute: list[tuple[int, int, int]] = []
    for row, col, color in cells:
        target = (anchor[0] + row, anchor[1] + col)
        if not _in_bounds(*target, height, width) or target in occupied:
            return False
        absolute.append((target[0], target[1], color))
    for row, col, color in absolute:
        if result[row][col] != background:
            return False
        result[row][col] = color
        occupied.add((row, col))
    return True


def execute_closed_program(program: ClosedProgram, grid: Any) -> Grid | None:
    """Execute only if every operation has an unambiguous source/target."""

    source_grid = normalize_grid(grid)
    result = [list(row) for row in source_grid]
    background = _background(source_grid)
    objects = extract_objects(source_grid)
    used_source: set[int] = set()
    edits: list[tuple[Operation, Object | None, tuple[int, int] | None]] = []

    for operation in program.operations:
        if operation.kind == "identity":
            continue
        if operation.kind == "add":
            if operation.target_colored_shape is None or operation.target_anchor is None:
                return None
            edits.append((operation, None, operation.target_anchor))
            continue
        if operation.source_shape is None or operation.source_colored_shape is None:
            return None
        matches = [
            (index, obj) for index, obj in enumerate(objects)
            if obj.shape == operation.source_shape
            and obj.colored_shape == operation.source_colored_shape
        ]
        if len(matches) != 1:
            return None
        index, obj = matches[0]
        if index in used_source:
            return None
        used_source.add(index)
        if operation.kind == "delete":
            edits.append((operation, obj, None))
        elif operation.kind in {"move", "recolor", "transform"}:
            edits.append((operation, obj, obj.anchor))
        else:
            return None

    # Clear all source objects first so a move can vacate its origin.  A source
    # cell may be cleared by at most one operation because matches are unique.
    for operation, obj, _ in edits:
        if obj is None or operation.kind == "add":
            continue
        for row, col, _ in obj.cells:
            result[row][col] = background

    occupied: set[tuple[int, int]] = set()
    for operation, obj, anchor in edits:
        if operation.kind == "delete":
            continue
        if operation.kind == "recolor":
            if obj is None or operation.target_colored_shape is None:
                return None
            target_anchor = obj.anchor
            target_cells = operation.target_colored_shape
        elif operation.kind == "move":
            if obj is None or operation.source_colored_shape is None:
                return None
            if operation.displacement is None:
                return None
            target_anchor = (
                obj.anchor[0] + operation.displacement[0],
                obj.anchor[1] + operation.displacement[1],
            )
            target_cells = operation.source_colored_shape
        elif operation.kind == "transform":
            if obj is None or operation.target_colored_shape is None:
                return None
            if operation.target_anchor is None:
                return None
            target_anchor = operation.target_anchor
            target_cells = operation.target_colored_shape
        elif operation.kind == "add":
            if operation.target_colored_shape is None or anchor is None:
                return None
            target_anchor = anchor
            target_cells = operation.target_colored_shape
        else:
            return None
        if not _paint_cells(
            result, target_cells, target_anchor, background, occupied=occupied
        ):
            return None
    return tuple(tuple(row) for row in result)


def infer_closed_program(
    source_grid: Any,
    target_grid: Any,
    correspondence: Correspondence,
    *,
    allow_shape_transform: bool = False,
) -> ClosedProgram | None:
    """Infer a closed program from one matching, or reject it.

    Shape-changing and move-plus-recolor effects remain opt-in because their
    target geometry must be grounded by a correspondence, not guessed.
    """

    source = normalize_grid(source_grid)
    target = normalize_grid(target_grid)
    if (len(source), len(source[0])) != (len(target), len(target[0])):
        return None
    source_objects = extract_objects(source)
    target_objects = extract_objects(target)
    if (any(i >= len(source_objects) or j >= len(target_objects)
            for i, j in correspondence.pairs)
            or any(i >= len(source_objects) for i in correspondence.unmatched_source)
            or any(j >= len(target_objects) for j in correspondence.unmatched_target)):
        return None
    operations: list[Operation] = []
    seen_sources: set[tuple[Shape, ColoredShape]] = set()
    for source_index, target_index in correspondence.pairs:
        left, right = source_objects[source_index], target_objects[target_index]
        fingerprint = (left.shape, left.colored_shape)
        if fingerprint in seen_sources:
            return None
        seen_sources.add(fingerprint)
        if left.shape != right.shape:
            if not allow_shape_transform:
                return None
            kind = "transform"
            operations.append(Operation(
                kind=kind,
                source_shape=left.shape,
                source_colored_shape=left.colored_shape,
                target_shape=right.shape,
                target_colored_shape=right.colored_shape,
                target_anchor=right.anchor,
                displacement=(right.anchor[0] - left.anchor[0],
                              right.anchor[1] - left.anchor[1]),
            ))
        elif left.colored_shape == right.colored_shape:
            if left.anchor == right.anchor:
                kind = "identity"
            else:
                kind = "move"
            operations.append(Operation(
                kind=kind,
                source_shape=left.shape,
                source_colored_shape=left.colored_shape,
                target_shape=right.shape,
                target_colored_shape=right.colored_shape,
                displacement=(right.anchor[0] - left.anchor[0],
                              right.anchor[1] - left.anchor[1]),
            ))
        elif left.anchor == right.anchor:
            operations.append(Operation(
                kind="recolor",
                source_shape=left.shape,
                source_colored_shape=left.colored_shape,
                target_shape=right.shape,
                target_colored_shape=right.colored_shape,
                displacement=(0, 0),
            ))
        elif allow_shape_transform:
            operations.append(Operation(
                kind="transform",
                source_shape=left.shape,
                source_colored_shape=left.colored_shape,
                target_shape=right.shape,
                target_colored_shape=right.colored_shape,
                target_anchor=right.anchor,
                displacement=(right.anchor[0] - left.anchor[0],
                              right.anchor[1] - left.anchor[1]),
            ))
        else:
            return None
    for source_index in correspondence.unmatched_source:
        left = source_objects[source_index]
        fingerprint = (left.shape, left.colored_shape)
        if fingerprint in seen_sources:
            return None
        seen_sources.add(fingerprint)
        operations.append(Operation(
            kind="delete",
            source_shape=left.shape,
            source_colored_shape=left.colored_shape,
        ))
    for target_index in correspondence.unmatched_target:
        right = target_objects[target_index]
        operations.append(Operation(
            kind="add",
            target_shape=right.shape,
            target_colored_shape=right.colored_shape,
            target_anchor=right.anchor,
        ))
    program = ClosedProgram(tuple(sorted(operations, key=repr)))
    if execute_closed_program(program, source) != target:
        return None
    return program


def verified_closed_programs(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    allow_shape_transform: bool = False,
    minimum_cost_only: bool = True,
) -> tuple[ClosedProgram, ...]:
    """Return programs from first-demo matchings that replay every demo."""

    train = task.get("train", [])
    if not train:
        return ()
    first = train[0]
    programs: list[ClosedProgram] = []
    seen: set[ClosedProgram] = set()
    try:
        correspondences = top_k_correspondences(
            first["input"], first["output"], k=k, max_objects=max_objects
        )
    except ValueError:
        return ()
    if not correspondences:
        return ()
    minimum_cost = correspondences[0].cost
    for correspondence in correspondences:
        # The conservative baseline keeps tied optima only.  The opt-in
        # lookahead mode lets cross-demo exact replay overrule local cost.
        if minimum_cost_only and correspondence.cost != minimum_cost:
            continue
        program = infer_closed_program(
            first["input"], first["output"], correspondence,
            allow_shape_transform=allow_shape_transform,
        )
        if program is None or program in seen:
            continue
        if all(execute_closed_program(program, pair["input"])
               == normalize_grid(pair["output"]) for pair in train):
            seen.add(program)
            programs.append(program)
    return tuple(programs)


def dataset_closed_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "verified_tasks": 0,
        "verified_programs": 0,
        "cap_or_unsupported_tasks": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        programs = verified_closed_programs(
            task, k=k, max_objects=max_objects
        )
        summary["verified_programs"] += len(programs)
        if programs:
            summary["verified_tasks"] += 1
        else:
            summary["cap_or_unsupported_tasks"] += 1
    return dict(summary)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1], "r", encoding="utf-8") as handle:
            print(dataset_closed_profile(json.load(handle)))
    else:
        program = verified_closed_programs({
            "train": [{
                "input": [[0, 1, 0], [0, 0, 0]],
                "output": [[0, 0, 1], [0, 0, 0]],
            }]
        })
        assert program
        print("closed_object_executor selftest: PASS")
