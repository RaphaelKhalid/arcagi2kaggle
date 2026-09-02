"""Small complete-grid scene-graph rewrite compiler.

This is the next boundary after the strict frame executor.  It keeps the
existing proof gates but adds target-grounded shape transforms and generated
objects anchored relative to a uniquely selected source role.  The grammar is
deliberately finite; it is a falsifiable renderer, not an unconstrained
program synthesizer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from experiments.frame_role_executor import LocalObservation, _local_observations
    from experiments.graph_lgg import (
        ActionSchema, lgg_observations,
    )
    from experiments.guarded_roles import select_roles
    from experiments.object_correspondence import top_k_correspondences
    from experiments.object_deltas import (
        Connectivity, Object, extract_objects, normalize_grid,
    )
except ModuleNotFoundError:  # direct ``python experiments/scene_graph_rewrite.py``
    from frame_role_executor import LocalObservation, _local_observations
    from graph_lgg import ActionSchema, lgg_observations
    from guarded_roles import select_roles
    from object_correspondence import top_k_correspondences
    from object_deltas import Connectivity, Object, extract_objects, normalize_grid


Grid = tuple[tuple[int, ...], ...]
Shape = tuple[tuple[int, int], ...]
ColoredShape = tuple[tuple[int, int, int], ...]
Predicate = tuple[Any, ...]


@dataclass(frozen=True)
class RewriteClause:
    kind: str
    source_guard: frozenset[Predicate] = frozenset()
    displacement: tuple[int, int] | None = None
    target_colored_shape: ColoredShape | None = None
    reference_guard: frozenset[Predicate] = frozenset()


@dataclass(frozen=True)
class SceneRewriteProgram:
    clauses: tuple[RewriteClause, ...]
    connectivity: Connectivity = 4


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _cells(obj: Object) -> set[tuple[int, int]]:
    return {(row, col) for row, col, _ in obj.cells}


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
        target = (anchor[0] + row, anchor[1] + col)
        if not (0 <= target[0] < height and 0 <= target[1] < width):
            return False
        if target in occupied or result[target[0]][target[1]] != background:
            return False
        absolute.append((target[0], target[1], color))
    for row, col, color in absolute:
        result[row][col] = color
        occupied.add((row, col))
    return True


def execute_scene_rewrite(
    program: SceneRewriteProgram,
    grid: Any,
) -> Grid | None:
    """Execute a rewrite only when every role and write is unambiguous."""

    source = normalize_grid(grid)
    objects = extract_objects(source, connectivity=program.connectivity)
    result = [list(row) for row in source]
    background = _background(source)
    selected: list[tuple[RewriteClause, Object | None]] = []
    used_source: set[int] = set()
    for clause in program.clauses:
        if clause.kind == "identity":
            continue
        if clause.kind == "add":
            if (not clause.reference_guard or
                    not clause.target_colored_shape or
                    len(select_roles(
                        source, clause.reference_guard,
                        connectivity=program.connectivity,
                    )) != 1):
                return None
            selected.append((clause, None))
            continue
        if not clause.source_guard:
            return None
        matches = select_roles(
            source, clause.source_guard, connectivity=program.connectivity
        )
        if len(matches) != 1 or matches[0] in used_source:
            return None
        index = matches[0]
        used_source.add(index)
        if clause.kind not in {
            "move", "recolor", "move_recolor", "transform", "delete",
        }:
            return None
        selected.append((clause, objects[index]))

    for clause, obj in selected:
        if obj is not None:
            for row, col in _cells(obj):
                result[row][col] = background

    occupied: set[tuple[int, int]] = set()
    for clause, obj in selected:
        if clause.kind == "delete":
            continue
        if clause.kind == "add":
            refs = select_roles(
                source, clause.reference_guard,
                connectivity=program.connectivity,
            )
            if len(refs) != 1 or clause.displacement is None:
                return None
            reference = objects[refs[0]]
            anchor = (
                reference.anchor[0] + clause.displacement[0],
                reference.anchor[1] + clause.displacement[1],
            )
            cells = clause.target_colored_shape
        elif obj is None:
            return None
        elif clause.kind == "move":
            if clause.displacement is None:
                return None
            anchor = (
                obj.anchor[0] + clause.displacement[0],
                obj.anchor[1] + clause.displacement[1],
            )
            cells = obj.colored_shape
        elif clause.kind == "recolor":
            if clause.target_colored_shape is None:
                return None
            anchor = obj.anchor
            cells = clause.target_colored_shape
        elif clause.kind in {"move_recolor", "transform"}:
            if clause.displacement is None or clause.target_colored_shape is None:
                return None
            anchor = (
                obj.anchor[0] + clause.displacement[0],
                obj.anchor[1] + clause.displacement[1],
            )
            cells = clause.target_colored_shape
        else:
            return None
        if cells is None or not _paint(
            result, cells, anchor, background, occupied
        ):
            return None
    return tuple(tuple(row) for row in result)


def _stable_anchor_offset(
    local_traces: tuple[tuple[LocalObservation, ...], ...],
    source_objects: tuple[tuple[Object, ...], ...],
    target_objects: tuple[tuple[Object, ...], ...],
    active_index: int,
    reference_index: int,
) -> tuple[int, int] | None:
    offsets: list[tuple[int, int]] = []
    for trace, sources, targets in zip(
        local_traces, source_objects, target_objects
    ):
        active = trace[active_index]
        reference = trace[reference_index]
        if (active.target_index is None or reference.target_index is None or
                active.target_index >= len(targets) or
                reference.target_index >= len(targets)):
            return None
        active_anchor = targets[active.target_index].anchor
        reference_anchor = targets[reference.target_index].anchor
        offsets.append((
            active_anchor[0] - reference_anchor[0],
            active_anchor[1] - reference_anchor[1],
        ))
    if not offsets or len(set(offsets)) != 1:
        return None
    return offsets[0]


def compile_scene_rewrite(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    connectivity: Connectivity = 4,
) -> SceneRewriteProgram | None:
    """Compile the bounded rewrite and require exact replay of every demo."""

    pairs = task.get("train", [])
    if not pairs:
        return None
    local_traces: list[tuple[LocalObservation, ...]] = []
    source_objects: list[tuple[Object, ...]] = []
    target_objects: list[tuple[Object, ...]] = []
    try:
        for pair in pairs:
            correspondence = top_k_correspondences(
                pair["input"], pair["output"], k=k,
                max_objects=max_objects, connectivity=connectivity,
            )[0]
            local_traces.append(_local_observations(
                pair["input"], pair["output"], correspondence,
                connectivity=connectivity,
            ))
            source_objects.append(extract_objects(
                pair["input"], connectivity=connectivity
            ))
            target_objects.append(extract_objects(
                pair["output"], connectivity=connectivity
            ))
    except (IndexError, ValueError):
        return None
    if not local_traces or len({len(trace) for trace in local_traces}) != 1:
        return None
    schemas = lgg_observations(tuple(
        tuple(item.observation for item in trace)
        for trace in local_traces
    ))
    if schemas is None or len(schemas) != len(local_traces[0]):
        return None

    for schema, pair in zip(schemas, pairs):
        if schema.source_guard and len(select_roles(
            pair["input"], schema.source_guard, connectivity=connectivity
        )) != 1:
            return None
        if schema.target_guard and len(select_roles(
            pair["output"], schema.target_guard, connectivity=connectivity
        )) != 1:
            return None

    clauses: list[RewriteClause] = []
    for index, (schema, first_local) in enumerate(zip(
        schemas, local_traces[0]
    )):
        if schema.kind is None or schema.kind != first_local.observation.kind:
            return None
        kind = schema.kind
        if kind == "identity":
            if not schema.source_guard:
                return None
            clauses.append(RewriteClause("identity", schema.source_guard))
            continue
        if kind == "delete":
            if not schema.source_guard:
                return None
            clauses.append(RewriteClause("delete", schema.source_guard))
            continue
        if first_local.source_index is None:
            if kind != "add" or not schema.target_guard:
                return None
            reference_candidates = [
                ref for ref, ref_schema in enumerate(schemas)
                if ref != index and ref_schema.source_guard
                and ref_schema.kind not in {"add", "delete"}
                and all(
                    trace[ref].target_index is not None
                    for trace in local_traces
                )
            ]
            chosen_reference = None
            offset = None
            for ref in reference_candidates:
                candidate_offset = _stable_anchor_offset(
                    tuple(local_traces), tuple(source_objects),
                    tuple(target_objects), index, ref,
                )
                if candidate_offset is not None:
                    chosen_reference, offset = ref, candidate_offset
                    break
            if chosen_reference is None or offset is None:
                return None
            target = target_objects[0][first_local.target_index]
            if any(
                target_objects[demo][trace[index].target_index].colored_shape
                != target.colored_shape
                for demo, trace in enumerate(local_traces)
                if trace[index].target_index is not None
            ):
                return None
            clauses.append(RewriteClause(
                "add", target_colored_shape=target.colored_shape,
                reference_guard=schemas[chosen_reference].source_guard,
                displacement=offset,
            ))
            continue

        if not schema.source_guard or first_local.source_index is None:
            return None
        first_source = source_objects[0][first_local.source_index]
        first_target = target_objects[0][first_local.target_index]
        offsets: list[tuple[int, int]] = []
        target_shapes: list[ColoredShape] = []
        for demo, trace in enumerate(local_traces):
            local = trace[index]
            if local.source_index is None or local.target_index is None:
                return None
            source = source_objects[demo][local.source_index]
            target = target_objects[demo][local.target_index]
            offsets.append((
                target.anchor[0] - source.anchor[0],
                target.anchor[1] - source.anchor[1],
            ))
            target_shapes.append(target.colored_shape)
            if kind == "move" and (
                source.shape != target.shape or
                source.colored_shape != target.colored_shape
            ):
                return None
            if kind == "recolor" and (
                source.shape != target.shape or source.anchor != target.anchor
            ):
                return None
        if kind not in {"move", "recolor", "move_recolor", "transform"}:
            return None
        if kind == "recolor" and any(offset != (0, 0) for offset in offsets):
            return None
        if kind != "recolor" and len(set(offsets)) != 1:
            return None
        if kind in {"move_recolor", "transform"} and len(set(target_shapes)) != 1:
            return None
        if kind == "recolor" and len(set(target_shapes)) != 1:
            return None
        clauses.append(RewriteClause(
            kind=kind,
            source_guard=schema.source_guard,
            displacement=(0, 0) if kind == "recolor" else offsets[0],
            target_colored_shape=(
                target_shapes[0] if kind in {"recolor", "move_recolor", "transform"}
                else None
            ),
        ))

    program = SceneRewriteProgram(tuple(clauses), connectivity=connectivity)
    if all(
        execute_scene_rewrite(program, pair["input"])
        == normalize_grid(pair["output"])
        for pair in pairs
    ):
        return program
    return None


def dataset_scene_rewrite_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, list[Any]] | None = None,
    *,
    k: int = 4,
    max_objects: int = 10,
    connectivity: Connectivity = 4,
) -> dict[str, int]:
    """Measure complete candidate support, optionally on disjoint labels."""

    summary = Counter({
        "tasks": 0, "compiled_tasks": 0,
        "candidate_outputs": 0, "correct_outputs": 0,
    })
    for task_id, task in challenges.items():
        summary["tasks"] += 1
        program = compile_scene_rewrite(
            task, k=k, max_objects=max_objects, connectivity=connectivity
        )
        if program is None:
            continue
        summary["compiled_tasks"] += 1
        for index, test in enumerate(task.get("test", [])):
            prediction = execute_scene_rewrite(program, test["input"])
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
        "input": [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 2]],
        "output": [[0, 1, 0, 3, 0], [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 2]],
    }]}
    # New object is placed one cell to the right of the top source role.
    program = compile_scene_rewrite(task)
    assert program is not None
    print("scene_graph_rewrite selftest: PASS")
