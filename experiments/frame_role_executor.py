"""Proof-gated multi-action role executor with an explicit frame condition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from experiments.graph_lgg import (
        ActionObservation,
        ActionSchema,
        _axis,
        _kind,
        _relation,
        lgg_observations,
        role_predicates,
    )
    from experiments.guarded_roles import select_roles
    from experiments.object_correspondence import Correspondence, top_k_correspondences
    from experiments.object_deltas import (
        Connectivity, Object, extract_objects, normalize_grid,
    )
except ModuleNotFoundError:  # direct ``python experiments/frame_role_executor.py``
    from graph_lgg import ActionObservation, ActionSchema, _axis, _kind, _relation, lgg_observations, role_predicates
    from guarded_roles import select_roles
    from object_correspondence import Correspondence, top_k_correspondences
    from object_deltas import Connectivity, Object, extract_objects, normalize_grid


Grid = tuple[tuple[int, ...], ...]
Shape = tuple[tuple[int, int], ...]
ColoredShape = tuple[tuple[int, int, int], ...]
Predicate = tuple[Any, ...]


@dataclass(frozen=True)
class LocalObservation:
    observation: ActionObservation
    source_index: int | None
    target_index: int | None


@dataclass(frozen=True)
class FrameClause:
    kind: str
    source_guard: frozenset[Predicate]
    displacement: tuple[int, int] | None = None
    source_shape: Shape | None = None
    recolor_target: ColoredShape | None = None


@dataclass(frozen=True)
class FrameProgram:
    clauses: tuple[FrameClause, ...]
    connectivity: Connectivity = 4


def _local_observations(
    source_grid: Any,
    target_grid: Any,
    correspondence: Correspondence,
    *,
    connectivity: Connectivity = 4,
) -> tuple[LocalObservation, ...]:
    source = extract_objects(
        normalize_grid(source_grid), connectivity=connectivity
    )
    target = extract_objects(
        normalize_grid(target_grid), connectivity=connectivity
    )
    if (any(i >= len(source) or j >= len(target)
            for i, j in correspondence.pairs)
            or any(i >= len(source) for i in correspondence.unmatched_source)
            or any(j >= len(target) for j in correspondence.unmatched_target)):
        raise ValueError("correspondence refers to an object outside its grids")
    result: list[LocalObservation] = []
    for source_index, target_index in correspondence.pairs:
        left, right = source[source_index], target[target_index]
        shape, area, palette = _relation(left, right)
        displacement = (
            right.anchor[0] - left.anchor[0],
            right.anchor[1] - left.anchor[1],
        )
        result.append(LocalObservation(ActionObservation(
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
        ), source_index, target_index))
    for source_index in correspondence.unmatched_source:
        result.append(LocalObservation(ActionObservation(
            kind="delete", motion_axis=None, shape_relation=None,
            area_relation=None, palette_relation=None,
            source_guard=role_predicates(
                source_grid, source_index, connectivity=connectivity
            ),
            target_guard=frozenset(),
        ), source_index, None))
    for target_index in correspondence.unmatched_target:
        result.append(LocalObservation(ActionObservation(
            kind="add", motion_axis=None, shape_relation=None,
            area_relation=None, palette_relation=None,
            source_guard=frozenset(),
            target_guard=role_predicates(
                target_grid, target_index, connectivity=connectivity
            ),
        ), None, target_index))
    return tuple(sorted(result, key=lambda item: repr(item.observation)))


def _paint(
    result: list[list[int]],
    cells: ColoredShape,
    anchor: tuple[int, int],
    background: int,
    occupied: set[tuple[int, int]],
) -> bool:
    height, width = len(result), len(result[0])
    absolute: list[tuple[int, int, int]] = []
    for row, col, color in cells:
        target = anchor[0] + row, anchor[1] + col
        if not (0 <= target[0] < height and 0 <= target[1] < width):
            return False
        if target in occupied:
            return False
        absolute.append((target[0], target[1], color))
    if any(result[row][col] != background for row, col, _ in absolute):
        return False
    for row, col, color in absolute:
        result[row][col] = color
        occupied.add((row, col))
    return True


def execute_frame_program(program: FrameProgram, grid: Any) -> Grid | None:
    source = normalize_grid(grid)
    objects = extract_objects(source, connectivity=program.connectivity)
    result = [list(row) for row in source]
    background = Counter(cell for row in source for cell in row).most_common(1)[0][0]
    selected: list[tuple[FrameClause, Object | None]] = []
    selected_indices: set[int] = set()
    for clause in program.clauses:
        if clause.kind == "identity":
            continue
        matches = select_roles(
            source, clause.source_guard, connectivity=program.connectivity
        )
        if len(matches) != 1 or matches[0] in selected_indices:
            return None
        index = matches[0]
        selected_indices.add(index)
        if clause.kind not in {"move", "recolor", "delete"}:
            return None
        selected.append((clause, objects[index]))
    for _, obj in selected:
        for row, col, _ in obj.cells:
            result[row][col] = background
    occupied: set[tuple[int, int]] = set()
    for clause, obj in selected:
        if clause.kind == "delete":
            continue
        if clause.kind == "move":
            if clause.displacement is None:
                return None
            anchor = (
                obj.anchor[0] + clause.displacement[0],
                obj.anchor[1] + clause.displacement[1],
            )
            cells = obj.colored_shape
        elif clause.kind == "recolor":
            if clause.recolor_target is None or clause.source_shape != obj.shape:
                return None
            anchor = obj.anchor
            cells = clause.recolor_target
        else:
            return None
        if not _paint(result, cells, anchor, background, occupied):
            return None
    return tuple(tuple(row) for row in result)


def compile_frame_program(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    connectivity: Connectivity = 4,
) -> FrameProgram | None:
    pairs = task.get("train", [])
    if not pairs:
        return None
    traces: list[tuple[ActionObservation, ...]] = []
    local_traces: list[tuple[LocalObservation, ...]] = []
    try:
        for pair in pairs:
            correspondence = top_k_correspondences(
                pair["input"], pair["output"], k=k,
                max_objects=max_objects, connectivity=connectivity
            )[0]
            local = _local_observations(
                pair["input"], pair["output"], correspondence,
                connectivity=connectivity,
            )
            local_traces.append(local)
            traces.append(tuple(item.observation for item in local))
    except (IndexError, ValueError):
        return None
    schema = lgg_observations(traces)
    if schema is None or not schema:
        return None
    allowed = {"identity", "move", "recolor", "delete"}
    if any(item.kind not in allowed for item in schema):
        return None
    # Guards must identify one object in every input/output scene.  Empty
    # guards are never promoted to a program clause.
    for schema_item in schema:
        if not schema_item.source_guard or not schema_item.target_guard:
            return None
        for pair in pairs:
            if (len(select_roles(
                    pair["input"], schema_item.source_guard,
                    connectivity=connectivity
                )) != 1
                    or len(select_roles(
                        pair["output"], schema_item.target_guard,
                        connectivity=connectivity
                    )) != 1):
                return None
    first_local = local_traces[0]
    if len(first_local) != len(schema):
        return None
    clauses: list[FrameClause] = []
    first_source = extract_objects(
        normalize_grid(pairs[0]["input"]), connectivity=connectivity
    )
    first_target = extract_objects(
        normalize_grid(pairs[0]["output"]), connectivity=connectivity
    )
    for schema_item, local in zip(schema, first_local):
        if schema_item.kind != local.observation.kind:
            return None
        if schema_item.kind == "identity":
            clauses.append(FrameClause("identity", schema_item.source_guard))
            continue
        if local.source_index is None:
            return None
        left = first_source[local.source_index]
        if schema_item.kind == "delete":
            clauses.append(FrameClause("delete", schema_item.source_guard))
            continue
        if local.target_index is None:
            return None
        right = first_target[local.target_index]
        if left.shape != right.shape:
            return None
        if schema_item.kind == "move" and left.colored_shape == right.colored_shape:
            clauses.append(FrameClause(
                "move", schema_item.source_guard,
                displacement=(right.anchor[0] - left.anchor[0],
                              right.anchor[1] - left.anchor[1]),
                source_shape=left.shape,
            ))
        elif schema_item.kind == "recolor" and left.anchor == right.anchor:
            clauses.append(FrameClause(
                "recolor", schema_item.source_guard,
                source_shape=left.shape,
                recolor_target=right.colored_shape,
            ))
        else:
            return None
    program = FrameProgram(tuple(clauses), connectivity=connectivity)
    if all(execute_frame_program(program, pair["input"])
           == normalize_grid(pair["output"]) for pair in pairs):
        return program
    return None


def dataset_frame_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, list[Any]] | None = None,
    *,
    k: int = 4,
    max_objects: int = 10,
    connectivity: Connectivity = 4,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "compiled_tasks": 0,
        "candidate_outputs": 0,
        "correct_outputs": 0,
    })
    for task_id, task in challenges.items():
        summary["tasks"] += 1
        program = compile_frame_program(
            task, k=k, max_objects=max_objects, connectivity=connectivity
        )
        if program is None:
            continue
        summary["compiled_tasks"] += 1
        for index, item in enumerate(task.get("test", [])):
            prediction = execute_frame_program(program, item["input"])
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
        "input": [[0, 1, 0, 0, 2]],
        "output": [[0, 2, 0, 0, 3]],
    }]}
    # This has two recolors and is an intentionally simple self-test.
    program = compile_frame_program(task)
    assert program is not None
    print("frame_role_executor selftest: PASS")
