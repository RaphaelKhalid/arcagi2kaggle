"""Fit finite effect equations over aligned, guarded action clauses."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import islice, product
from typing import Any, Mapping

try:
    from experiments.aligned_clauses import (
        AlignedTraceHypothesis,
        aligned_local_hypotheses,
    )
    from experiments.effect_equations import (
        AnchorObservation,
        ConstantOffset,
        EffectEquation,
        RolePlusOffset,
        fit_equations,
    )
    from experiments.graph_lgg import role_predicates
    from experiments.guarded_roles import select_roles
    from experiments.object_deltas import extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/aligned_equations.py``
    from aligned_clauses import AlignedTraceHypothesis, aligned_local_hypotheses
    from effect_equations import AnchorObservation, ConstantOffset, EffectEquation, RolePlusOffset, fit_equations
    from graph_lgg import role_predicates
    from guarded_roles import select_roles
    from object_deltas import extract_objects, normalize_grid


@dataclass(frozen=True)
class EquationEvidence:
    clause_index: int
    equation: EffectEquation
    reference_guard: frozenset[tuple[Any, ...]]


def _stationary_indices(trace: tuple[Any, ...]) -> tuple[int, ...]:
    return tuple(
        item.source_index for item in trace
        if item.observation.kind == "identity"
        and item.source_index is not None
        and item.target_index is not None
    )


def _reference_guard(
    task: Mapping[str, Any],
    indices: tuple[int, ...],
) -> frozenset[tuple[Any, ...]]:
    guards = [
        role_predicates(pair["input"], index)
        for pair, index in zip(task.get("train", []), indices)
    ]
    if not guards:
        return frozenset()
    return frozenset.intersection(*guards)


def fit_move_equations(
    task: Mapping[str, Any],
    hypothesis: AlignedTraceHypothesis,
    clause_index: int,
    *,
    max_reference_choices: int = 8,
) -> tuple[EquationEvidence, ...]:
    """Fit constant/reference-plus-offset terms for one aligned move clause."""

    schema = hypothesis.schema[clause_index]
    if schema.kind != "move":
        return ()
    pairs = task.get("train", [])
    source_anchors: list[tuple[int, int]] = []
    target_anchors: list[tuple[int, int]] = []
    reference_options: list[tuple[int, ...]] = []
    for pair, trace in zip(pairs, hypothesis.traces):
        local = trace[clause_index]
        if local.source_index is None or local.target_index is None:
            return ()
        source = extract_objects(normalize_grid(pair["input"]))
        target = extract_objects(normalize_grid(pair["output"]))
        source_anchors.append(source[local.source_index].anchor)
        target_anchors.append(target[local.target_index].anchor)
        reference_options.append(_stationary_indices(trace))
    evidence: list[EquationEvidence] = []
    direct = fit_equations(tuple(
        AnchorObservation("A", target, {"A": source})
        for source, target in zip(source_anchors, target_anchors)
    ))
    evidence.extend(
        EquationEvidence(clause_index, equation, frozenset())
        for equation in direct if isinstance(equation, ConstantOffset)
    )
    if not all(reference_options):
        return tuple(evidence)
    for references in islice(product(*reference_options), max_reference_choices):
        guard = _reference_guard(task, tuple(references))
        if not guard:
            continue
        if any(len(select_roles(pair["input"], guard)) != 1
               for pair in pairs):
            continue
        observations: list[AnchorObservation] = []
        for pair, source_anchor, target_anchor, reference_index in zip(
            pairs, source_anchors, target_anchors, references
        ):
            source = extract_objects(normalize_grid(pair["input"]))
            observations.append(AnchorObservation(
                "A", target_anchor,
                {"A": source_anchor, "B": source[reference_index].anchor},
            ))
        for equation in fit_equations(tuple(observations)):
            if isinstance(equation, RolePlusOffset):
                evidence.append(EquationEvidence(
                    clause_index, equation, guard
                ))
    unique: list[EquationEvidence] = []
    seen: set[tuple[EffectEquation, frozenset[tuple[Any, ...]]]] = set()
    for item in sorted(evidence, key=repr):
        key = item.equation, item.reference_guard
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)


def hypothesis_equation_evidence(
    task: Mapping[str, Any],
    hypothesis: AlignedTraceHypothesis,
) -> tuple[EquationEvidence, ...]:
    evidence: list[EquationEvidence] = []
    for clause_index, schema in enumerate(hypothesis.schema):
        evidence.extend(fit_move_equations(task, hypothesis, clause_index))
    return tuple(evidence)


def task_aligned_equation_profile(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 1,
) -> dict[str, int | bool]:
    hypotheses = aligned_local_hypotheses(
        task, k=k, max_objects=max_objects, max_hypotheses=max_hypotheses
    )
    equation_hypotheses = [
        (hypothesis, hypothesis_equation_evidence(task, hypothesis))
        for hypothesis in hypotheses
    ]
    with_equations = [item for item in equation_hypotheses if item[1]]
    reference_equations = [
        item for item in with_equations
        if any(isinstance(evidence.equation, RolePlusOffset)
               for evidence in item[1])
    ]
    return {
        "hypotheses": len(hypotheses),
        "with_equations": len(with_equations),
        "with_reference_equations": len(reference_equations),
        "equation_count": sum(len(item[1]) for item in equation_hypotheses),
    }


def dataset_aligned_equation_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 1,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "hypotheses": 0,
        "with_equations": 0,
        "with_reference_equations": 0,
        "equation_count": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        result = task_aligned_equation_profile(
            task, k=k, max_objects=max_objects,
            max_hypotheses=max_hypotheses,
        )
        for key in summary:
            if key != "tasks":
                summary[key] += int(result[key])
    return dict(summary)


if __name__ == "__main__":
    print("aligned_equations selftest: PASS")
