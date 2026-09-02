"""Execute the proof-gated single-action subset using relational role guards."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
import sys
from typing import Any, Mapping

try:
    from experiments.graph_lgg import (
        lgg_observations,
        observations_for_correspondence,
    )
    from experiments.guarded_roles import select_roles, top1_lgg_for_task
    from experiments.object_correspondence import Correspondence, top_k_correspondences
    from experiments.object_deltas import Object, extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/role_effect_executor.py``
    from graph_lgg import lgg_observations, observations_for_correspondence
    from guarded_roles import select_roles, top1_lgg_for_task
    from object_correspondence import Correspondence, top_k_correspondences
    from object_deltas import Object, extract_objects, normalize_grid


ColoredShape = tuple[tuple[int, int, int], ...]
Shape = tuple[tuple[int, int], ...]
Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class RoleEffectProgram:
    kind: str
    source_guard: frozenset[tuple[Any, ...]]
    target_guard: frozenset[tuple[Any, ...]]
    displacement: tuple[int, int] | None = None
    source_shape: Shape | None = None
    recolor_target: ColoredShape | None = None


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
        target = (anchor[0] + row, anchor[1] + col)
        if not (0 <= target[0] < height and 0 <= target[1] < width):
            return False
        absolute.append((target[0], target[1], color))
    if any(result[row][col] != background for row, col, _ in absolute):
        return False
    for row, col, color in absolute:
        result[row][col] = color
    return True


def execute_role_effect(
    program: RoleEffectProgram,
    grid: Any,
) -> Grid | None:
    source = normalize_grid(grid)
    objects = extract_objects(source)
    selected = select_roles(source, program.source_guard)
    if len(selected) != 1:
        return None
    obj = objects[selected[0]]
    result = [list(row) for row in source]
    background = _background(source)
    if program.kind not in {"move", "recolor", "delete"}:
        return None
    for row, col, _ in obj.cells:
        result[row][col] = background
    if program.kind == "delete":
        return tuple(tuple(row) for row in result)
    if program.kind == "move":
        if program.displacement is None:
            return None
        anchor = (
            obj.anchor[0] + program.displacement[0],
            obj.anchor[1] + program.displacement[1],
        )
        cells = obj.colored_shape
    else:
        if program.recolor_target is None:
            return None
        anchor = obj.anchor
        cells = program.recolor_target
    if not _paint(result, cells, anchor, background):
        return None
    return tuple(tuple(row) for row in result)


def _first_effect(
    source_grid: Any,
    target_grid: Any,
    correspondence: Correspondence,
    kind: str,
) -> RoleEffectProgram | None:
    source_objects = extract_objects(normalize_grid(source_grid))
    target_objects = extract_objects(normalize_grid(target_grid))
    observations = observations_for_correspondence(
        source_grid, target_grid, correspondence
    )
    if len(observations) != 1 or observations[0].kind != kind:
        return None
    observation = observations[0]
    if kind == "delete":
        if len(correspondence.unmatched_source) != 1:
            return None
        return RoleEffectProgram(
            kind=kind,
            source_guard=observation.source_guard,
            target_guard=observation.target_guard,
        )
    if len(correspondence.pairs) != 1:
        return None
    source_index, target_index = correspondence.pairs[0]
    left, right = source_objects[source_index], target_objects[target_index]
    if left.shape != right.shape:
        return None
    if kind == "move" and left.colored_shape == right.colored_shape:
        return RoleEffectProgram(
            kind=kind,
            source_guard=observation.source_guard,
            target_guard=observation.target_guard,
            displacement=(right.anchor[0] - left.anchor[0],
                          right.anchor[1] - left.anchor[1]),
            source_shape=left.shape,
        )
    if kind == "recolor" and left.anchor == right.anchor:
        return RoleEffectProgram(
            kind=kind,
            source_guard=observation.source_guard,
            target_guard=observation.target_guard,
            source_shape=left.shape,
            recolor_target=right.colored_shape,
        )
    return None


def compile_role_effect(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> RoleEffectProgram | None:
    """Compile and verify one unique-role, single-action relational effect."""

    bounded = top1_lgg_for_task(task, k=k, max_objects=max_objects)
    if bounded is None:
        return None
    schemas, selections = bounded
    if len(schemas) != 1 or not selections:
        return None
    schema = schemas[0]
    if schema.kind not in {"move", "recolor", "delete"}:
        return None
    if not all(
        len(selection.source_indices) == 1
        and len(selection.target_indices) == 1
        and schema.source_guard and schema.target_guard
        for selection in selections
    ):
        return None
    first = task["train"][0]
    try:
        correspondence = top_k_correspondences(
            first["input"], first["output"], k=k, max_objects=max_objects
        )[0]
    except (IndexError, ValueError):
        return None
    program = _first_effect(
        first["input"], first["output"], correspondence, schema.kind
    )
    if program is None:
        return None
    # The first demo supplies effect parameters; the LGG supplies the
    # generalized role guard.  Never carry first-demo-only predicates forward.
    program = replace(
        program,
        source_guard=schema.source_guard,
        target_guard=schema.target_guard,
    )
    if all(execute_role_effect(program, pair["input"])
           == normalize_grid(pair["output"])
           for pair in task.get("train", [])):
        return program
    return None


def dataset_role_effect_profile(
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
        program = compile_role_effect(task, k=k, max_objects=max_objects)
        if program is None:
            continue
        summary["compiled_tasks"] += 1
        for index, test in enumerate(task.get("test", [])):
            prediction = execute_role_effect(program, test["input"])
            if prediction is None:
                continue
            summary["candidate_outputs"] += 1
            if solutions is not None and task_id in solutions:
                if prediction == normalize_grid(solutions[task_id][index]):
                    summary["correct_outputs"] += 1
    return dict(summary)


if __name__ == "__main__":
    task = {"train": [{
        "input": [[0, 1, 0], [0, 0, 0]],
        "output": [[0, 0, 1], [0, 0, 0]],
    }]}
    program = compile_role_effect(task)
    assert program is not None
    assert execute_role_effect(program, task["train"][0]["input"])
    print("role_effect_executor selftest: PASS")
