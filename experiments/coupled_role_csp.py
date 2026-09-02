"""Compile aligned relation equations into a proof-gated role-anchor CSP."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import islice, product
from typing import Any, Mapping

try:
    from experiments.aligned_clauses import AlignedTraceHypothesis, aligned_local_hypotheses
    from experiments.anchor_csp import AnchorSystem, RelationConstraint, solve_anchor_system
    from experiments.guarded_roles import select_roles
    from experiments.object_deltas import Object, extract_objects, normalize_grid
    from experiments.relation_equations import (
        RelationEquation,
        fit_post_effect_relations,
    )
except ModuleNotFoundError:  # direct ``python experiments/coupled_role_csp.py``
    from aligned_clauses import AlignedTraceHypothesis, aligned_local_hypotheses
    from anchor_csp import AnchorSystem, RelationConstraint, solve_anchor_system
    from guarded_roles import select_roles
    from object_deltas import Object, extract_objects, normalize_grid
    from relation_equations import RelationEquation, fit_post_effect_relations


Grid = tuple[tuple[int, ...], ...]
Shape = tuple[tuple[int, int], ...]
ColoredShape = tuple[tuple[int, int, int], ...]
Predicate = tuple[Any, ...]


@dataclass(frozen=True)
class CoupledRoleProgram:
    hypothesis: AlignedTraceHypothesis
    equations: tuple[RelationEquation, ...]


def _objects(value: Any) -> tuple[Object, ...]:
    return extract_objects(normalize_grid(value))


def _active_indices(program: CoupledRoleProgram) -> frozenset[int]:
    return frozenset(equation.active_clause_index for equation in program.equations)


def _target_anchor_map(
    task: Mapping[str, Any],
    hypothesis: AlignedTraceHypothesis,
    demo_index: int,
    *,
    output: bool,
) -> dict[str, tuple[int, int]]:
    value = task["train"][demo_index]["output" if output else "input"]
    objects = _objects(value)
    trace = hypothesis.traces[demo_index]
    result: dict[str, tuple[int, int]] = {}
    for clause_index, local in enumerate(trace):
        object_index = local.target_index if output else local.source_index
        if object_index is not None:
            result[f"r{clause_index}"] = objects[object_index].anchor
    return result


def _shape_map(
    task: Mapping[str, Any],
    hypothesis: AlignedTraceHypothesis,
    demo_index: int,
    *,
    output: bool,
) -> dict[str, Shape]:
    value = task["train"][demo_index]["output" if output else "input"]
    objects = _objects(value)
    result: dict[str, Shape] = {}
    for clause_index, local in enumerate(hypothesis.traces[demo_index]):
        object_index = local.target_index if output else local.source_index
        if object_index is not None:
            result[f"r{clause_index}"] = objects[object_index].shape
    return result


def _blocked_unrepresented_output_cells(
    task: Mapping[str, Any],
    hypothesis: AlignedTraceHypothesis,
    demo_index: int,
) -> frozenset[tuple[int, int]]:
    output = _objects(task["train"][demo_index]["output"])
    represented = frozenset(
        local.target_index for local in hypothesis.traces[demo_index]
        if local.target_index is not None
    )
    return frozenset().union(*(
        frozenset((row, col) for row, col, _ in obj.cells)
        for index, obj in enumerate(output) if index not in represented
    ))


def _system_for_demo(
    program: CoupledRoleProgram,
    task: Mapping[str, Any],
    demo_index: int,
) -> AnchorSystem | None:
    schema = program.hypothesis.schema
    active = _active_indices(program)
    shapes = _shape_map(task, program.hypothesis, demo_index, output=True)
    target_anchors = _target_anchor_map(
        task, program.hypothesis, demo_index, output=True
    )
    relations = tuple(RelationConstraint(
        f"r{equation.active_clause_index}",
        f"r{equation.reference_clause_index}",
        equation.target_relation,
    ) for equation in program.equations)
    fixed = {
        role: anchor for index, anchor in target_anchors.items()
        if (int(index[1:]) not in active
            or schema[int(index[1:])].kind != "move")
        for role in (index,)
    }
    # Every active move must be represented by an equation; otherwise its
    # target anchor is unconstrained and cannot be a proof candidate.
    if any(
        item.kind == "move" and index not in active
        for index, item in enumerate(schema)
    ):
        return None
    if not shapes:
        return None
    grid = normalize_grid(task["train"][demo_index]["output"])
    return AnchorSystem(
        grid_shape=(len(grid), len(grid[0])),
        shapes=shapes,
        relations=relations,
        blocked=_blocked_unrepresented_output_cells(
            task, program.hypothesis, demo_index
        ),
        fixed=fixed,
    )


def _system_for_test(
    program: CoupledRoleProgram,
    grid: Any,
) -> tuple[AnchorSystem, dict[str, Object]] | None:
    normalized = normalize_grid(grid)
    objects = _objects(normalized)
    schema = program.hypothesis.schema
    active = _active_indices(program)
    shapes: dict[str, Shape] = {}
    selected: dict[str, Object] = {}
    fixed: dict[str, tuple[int, int]] = {}
    occupied_by_roles: set[int] = set()
    for index, item in enumerate(schema):
        if not item.source_guard:
            return None
        matches = select_roles(normalized, item.source_guard)
        if len(matches) != 1 or matches[0] in occupied_by_roles:
            return None
        object_index = matches[0]
        occupied_by_roles.add(object_index)
        role = f"r{index}"
        selected[role] = objects[object_index]
        shapes[role] = objects[object_index].shape
        if index not in active:
            fixed[role] = objects[object_index].anchor
    if any(
        item.kind == "move" and index not in active
        for index, item in enumerate(schema)
    ):
        return None
    blocked = frozenset().union(*(
        frozenset((row, col) for row, col, _ in obj.cells)
        for index, obj in enumerate(objects) if index not in occupied_by_roles
    ))
    relations = tuple(RelationConstraint(
        f"r{equation.active_clause_index}",
        f"r{equation.reference_clause_index}",
        equation.target_relation,
    ) for equation in program.equations)
    system = AnchorSystem(
        grid_shape=(len(normalized), len(normalized[0])),
        shapes=shapes,
        relations=relations,
        blocked=blocked,
        fixed=fixed,
    )
    return system, selected


def _apply_solution(
    program: CoupledRoleProgram,
    grid: Any,
    solution: Mapping[str, tuple[int, int]],
    selected: Mapping[str, Object],
) -> Grid | None:
    source = normalize_grid(grid)
    result = [list(row) for row in source]
    background = Counter(cell for row in source for cell in row).most_common(1)[0][0]
    active = _active_indices(program)
    for index in active:
        obj = selected.get(f"r{index}")
        if obj is None:
            return None
        for row, col, _ in obj.cells:
            result[row][col] = background
    occupied: set[tuple[int, int]] = set()
    for index in sorted(active):
        clause = program.hypothesis.schema[index]
        obj = selected[f"r{index}"]
        if clause.kind != "move":
            return None
        anchor = solution[f"r{index}"]
        absolute = [(anchor[0] + row, anchor[1] + col, color)
                    for row, col, color in obj.colored_shape]
        if any(row < 0 or row >= len(result) or col < 0 or col >= len(result[0])
               or (row, col) in occupied or result[row][col] != background
               for row, col, _ in absolute):
            return None
        for row, col, color in absolute:
            result[row][col] = color
            occupied.add((row, col))
    return tuple(tuple(row) for row in result)


def _verify_program(program: CoupledRoleProgram, task: Mapping[str, Any]) -> bool:
    for demo_index, pair in enumerate(task.get("train", [])):
        system = _system_for_demo(program, task, demo_index)
        if system is None:
            return False
        result = solve_anchor_system(system, max_solutions=2)
        if len(result.solutions) != 1 or result.exhausted:
            return False
        expected = _target_anchor_map(
            task, program.hypothesis, demo_index, output=True
        )
        if dict(result.solutions[0]) != expected:
            return False
    return True


def compile_coupled_programs(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 1,
) -> tuple[CoupledRoleProgram, ...]:
    programs: list[CoupledRoleProgram] = []
    seen: set[CoupledRoleProgram] = set()
    for hypothesis in aligned_local_hypotheses(
        task, k=k, max_objects=max_objects, max_hypotheses=max_hypotheses
    ):
        if any(item.kind not in {"identity", "move"} for item in hypothesis.schema):
            continue
        evidence = fit_post_effect_relations(task, hypothesis)
        by_active: dict[int, list[RelationEquation]] = {}
        for item in evidence:
            by_active.setdefault(item.active_clause_index, []).append(item)
        move_indices = [
            index for index, item in enumerate(hypothesis.schema)
            if item.kind == "move"
        ]
        if not move_indices:
            continue
        if any(index not in by_active for index in move_indices):
            continue
        choices = product(*(by_active[index] for index in move_indices))
        for selected in islice(choices, 8):
            equations = tuple(selected)
            program = CoupledRoleProgram(hypothesis, equations)
            if program in seen:
                continue
            if _verify_program(program, task):
                seen.add(program)
                programs.append(program)
    return tuple(programs)


def dataset_coupled_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, list[Any]] | None = None,
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 1,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "compiled_tasks": 0,
        "candidate_outputs": 0,
        "correct_outputs": 0,
    })
    for task_id, task in challenges.items():
        summary["tasks"] += 1
        programs = compile_coupled_programs(
            task, k=k, max_objects=max_objects,
            max_hypotheses=max_hypotheses,
        )
        if not programs:
            continue
        summary["compiled_tasks"] += 1
        program = programs[0]
        for index, test in enumerate(task.get("test", [])):
            test_system = _system_for_test(program, test["input"])
            if test_system is None:
                continue
            system, selected = test_system
            result = solve_anchor_system(system, max_solutions=2)
            if len(result.solutions) != 1 or result.exhausted:
                continue
            prediction = _apply_solution(
                program, test["input"], result.solutions[0], selected
            )
            if prediction is None:
                continue
            summary["candidate_outputs"] += 1
            if solutions is not None and task_id in solutions:
                summary["correct_outputs"] += int(
                    prediction == normalize_grid(solutions[task_id][index])
                )
    return dict(summary)


if __name__ == "__main__":
    print("coupled_role_csp selftest: PASS")
