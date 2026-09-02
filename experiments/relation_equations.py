"""Post-effect relational equations over aligned multi-action traces."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from experiments.aligned_clauses import (
        AlignedTraceHypothesis,
        aligned_local_hypotheses,
    )
    from experiments.graph_lgg import role_predicates
    from experiments.guarded_roles import select_roles
    from experiments.object_deltas import extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/relation_equations.py``
    from aligned_clauses import AlignedTraceHypothesis, aligned_local_hypotheses
    from graph_lgg import role_predicates
    from guarded_roles import select_roles
    from object_deltas import extract_objects, normalize_grid


Relation = tuple[Any, ...]
Predicate = tuple[Any, ...]


@dataclass(frozen=True)
class RelationEquation:
    active_clause_index: int
    reference_clause_index: int
    target_relation: Relation
    reference_guard: frozenset[Predicate]


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def anchor_relation(active: tuple[int, int], reference: tuple[int, int]) -> Relation:
    """Coarse post-effect relation; exact coordinate distance is quotiented."""

    dr = active[0] - reference[0]
    dc = active[1] - reference[1]
    direction = (_sign(dr), _sign(dc))
    if dr == 0 and dc == 0:
        axis = "coincident"
    elif dr == 0:
        axis = "horizontal"
    elif dc == 0:
        axis = "vertical"
    else:
        axis = "diagonal"
    distance_bucket = min(3, max(abs(dr), abs(dc)))
    return direction, axis, distance_bucket


def _reference_guard(
    task: Mapping[str, Any],
    reference_indices: tuple[int, ...],
) -> frozenset[Predicate]:
    guards = [
        role_predicates(pair["input"], index)
        for pair, index in zip(task.get("train", []), reference_indices)
    ]
    return frozenset.intersection(*guards) if guards else frozenset()


def relation_satisfying_anchors(
    grid: Any,
    shape: tuple[tuple[int, int], ...],
    reference_anchor: tuple[int, int],
    relation: Relation,
    blocked: frozenset[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Enumerate all legal anchors satisfying a coarse relation."""

    normalized = normalize_grid(grid)
    height, width = len(normalized), len(normalized[0])
    candidates: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            cells = frozenset((row + dr, col + dc) for dr, dc in shape)
            if any(r < 0 or r >= height or c < 0 or c >= width
                   for r, c in cells):
                continue
            if cells.intersection(blocked):
                continue
            if anchor_relation((row, col), reference_anchor) == relation:
                candidates.append((row, col))
    return tuple(candidates)


def relation_equation_unique_on_demos(
    task: Mapping[str, Any],
    hypothesis: AlignedTraceHypothesis,
    equation: RelationEquation,
) -> bool:
    """Prove the relation selects exactly the demonstrated target placement."""

    for pair, trace in zip(task.get("train", []), hypothesis.traces):
        active = trace[equation.active_clause_index]
        reference = trace[equation.reference_clause_index]
        if active.target_index is None or reference.target_index is None:
            return False
        output = normalize_grid(pair["output"])
        objects = extract_objects(output)
        active_object = objects[active.target_index]
        reference_object = objects[reference.target_index]
        blocked = frozenset().union(*(
            frozenset((row, col) for row, col, _ in obj.cells)
            for index, obj in enumerate(objects)
            if index != active.target_index
        ))
        candidates = relation_satisfying_anchors(
            output, active_object.shape, reference_object.anchor,
            equation.target_relation, blocked
        )
        if candidates != (active_object.anchor,):
            return False
    return True


def fit_post_effect_relations(
    task: Mapping[str, Any],
    hypothesis: AlignedTraceHypothesis,
    *,
    max_references_per_clause: int = 8,
) -> tuple[RelationEquation, ...]:
    """Fit stable output relations to every aligned active/reference pair."""

    pairs = task.get("train", [])
    evidence: list[RelationEquation] = []
    if not pairs:
        return ()
    for active_index, schema in enumerate(hypothesis.schema):
        if schema.kind in {None, "identity", "delete", "add"}:
            continue
        for reference_index in range(len(hypothesis.schema)):
            if reference_index == active_index:
                continue
            active_targets: list[tuple[int, int]] = []
            reference_targets: list[tuple[int, int]] = []
            reference_sources: list[int] = []
            valid = True
            for pair, trace in zip(pairs, hypothesis.traces):
                active = trace[active_index]
                reference = trace[reference_index]
                if (active.target_index is None
                        or reference.source_index is None
                        or reference.target_index is None):
                    valid = False
                    break
                target_objects = extract_objects(normalize_grid(pair["output"]))
                active_targets.append(target_objects[active.target_index].anchor)
                reference_targets.append(target_objects[reference.target_index].anchor)
                reference_sources.append(reference.source_index)
            if not valid:
                continue
            guard = _reference_guard(task, tuple(reference_sources))
            if not guard or any(
                len(select_roles(pair["input"], guard)) != 1
                for pair in pairs
            ):
                continue
            relations = tuple(
                anchor_relation(active, reference)
                for active, reference in zip(active_targets, reference_targets)
            )
            if len(set(relations)) != 1:
                continue
            evidence.append(RelationEquation(
                active_clause_index=active_index,
                reference_clause_index=reference_index,
                target_relation=relations[0],
                reference_guard=guard,
            ))
            if len(evidence) >= max_references_per_clause:
                break
    unique: list[RelationEquation] = []
    seen: set[RelationEquation] = set()
    for item in evidence:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return tuple(unique)


def task_relation_profile(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 1,
) -> dict[str, int | bool]:
    hypotheses = aligned_local_hypotheses(
        task, k=k, max_objects=max_objects, max_hypotheses=max_hypotheses
    )
    total = 0
    with_relation = 0
    unique_relation = 0
    unique_placement = 0
    hypotheses_with_unique_placement = 0
    for hypothesis in hypotheses:
        evidence = fit_post_effect_relations(task, hypothesis)
        total += len(evidence)
        if evidence:
            with_relation += 1
            unique_relation += sum(
                len(select_roles(
                    task["train"][0]["input"], item.reference_guard
                )) == 1
                for item in evidence
            )
        if any(relation_equation_unique_on_demos(task, hypothesis, item)
               for item in evidence):
            hypotheses_with_unique_placement += 1
        unique_placement += sum(
            relation_equation_unique_on_demos(task, hypothesis, item)
            for item in evidence
        )
    return {
        "hypotheses": len(hypotheses),
        "with_relation_equations": with_relation,
        "unique_relation_equations": unique_relation,
        "relation_equation_count": total,
        "unique_placement_equations": unique_placement,
        "hypotheses_with_unique_placement": hypotheses_with_unique_placement,
    }


def dataset_relation_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 1,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "hypotheses": 0,
        "with_relation_equations": 0,
        "unique_relation_equations": 0,
        "relation_equation_count": 0,
        "unique_placement_equations": 0,
        "hypotheses_with_unique_placement": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        result = task_relation_profile(
            task, k=k, max_objects=max_objects,
            max_hypotheses=max_hypotheses,
        )
        for key in summary:
            if key != "tasks":
                summary[key] += int(result[key])
    return dict(summary)


if __name__ == "__main__":
    assert anchor_relation((2, 0), (1, 0)) == ((1, 0), "vertical", 1)
    print("relation_equations selftest: PASS")
