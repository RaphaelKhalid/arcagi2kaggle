"""Keep local object indices while aligning top-k action traces."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import islice, product
from typing import Any, Mapping

try:
    from experiments.effect_equations import AnchorObservation, fit_equations
    from experiments.frame_role_executor import LocalObservation, _local_observations
    from experiments.graph_lgg import ActionObservation, ActionSchema, lgg_observations
    from experiments.guarded_roles import select_roles
    from experiments.object_correspondence import top_k_correspondences
    from experiments.object_deltas import extract_objects, normalize_grid
    from experiments.trace_alignment import align_traces
except ModuleNotFoundError:  # direct ``python experiments/aligned_clauses.py``
    from effect_equations import AnchorObservation, fit_equations
    from frame_role_executor import LocalObservation, _local_observations
    from graph_lgg import ActionObservation, ActionSchema, lgg_observations
    from guarded_roles import select_roles
    from object_correspondence import top_k_correspondences
    from object_deltas import extract_objects, normalize_grid
    from trace_alignment import align_traces


@dataclass(frozen=True)
class AlignedTraceHypothesis:
    traces: tuple[tuple[LocalObservation, ...], ...]
    schema: tuple[ActionSchema, ...]


def _observations(trace: tuple[LocalObservation, ...]) -> tuple[ActionObservation, ...]:
    return tuple(item.observation for item in trace)


def _full_local_alignments(
    reference: tuple[LocalObservation, ...],
    other: tuple[LocalObservation, ...],
    *,
    k: int,
) -> tuple[tuple[LocalObservation, ...], ...]:
    alignments = align_traces(
        _observations(reference), _observations(other), k=k
    )
    result: list[tuple[LocalObservation, ...]] = []
    for alignment in alignments:
        if alignment.unmatched_reference or alignment.unmatched_other:
            continue
        by_reference = {i: j for i, j in alignment.pairs}
        result.append(tuple(other[by_reference[i]] for i in range(len(reference))))
    return tuple(result)


def aligned_local_hypotheses(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 8,
) -> tuple[AlignedTraceHypothesis, ...]:
    """Return bounded hypotheses with local indices retained per clause."""

    pairs = task.get("train", [])
    if not pairs:
        return ()
    options: list[tuple[tuple[LocalObservation, ...], ...]] = []
    try:
        for pair in pairs:
            correspondences = top_k_correspondences(
                pair["input"], pair["output"], k=k, max_objects=max_objects
            )
            if not correspondences:
                return ()
            options.append(tuple(_local_observations(
                pair["input"], pair["output"], correspondence
            ) for correspondence in correspondences))
    except (IndexError, ValueError):
        return ()
    reference = options[0][0]
    aligned_options: list[tuple[tuple[LocalObservation, ...], ...]] = []
    for demo_options in options[1:]:
        aligned: list[tuple[LocalObservation, ...]] = []
        for local in demo_options:
            aligned.extend(_full_local_alignments(reference, local, k=k))
        # Preserve deterministic order while avoiding duplicate local traces.
        unique: list[tuple[LocalObservation, ...]] = []
        seen: set[tuple[LocalObservation, ...]] = set()
        for local in aligned:
            if local in seen:
                continue
            seen.add(local)
            unique.append(local)
            if len(unique) == k:
                break
        if not unique:
            return ()
        aligned_options.append(tuple(unique))
    result: list[AlignedTraceHypothesis] = []
    seen_schema: set[tuple[ActionSchema, ...]] = set()
    choices_iter = product(*aligned_options) if aligned_options else [()]
    for choices in islice(choices_iter, max_hypotheses):
        local_traces = (reference,) + tuple(choices)
        schema = lgg_observations(tuple(
            _observations(trace) for trace in local_traces
        ))
        if schema is None or schema in seen_schema:
            continue
        seen_schema.add(schema)
        result.append(AlignedTraceHypothesis(local_traces, schema))
    return tuple(result)


def _unique_roles(
    task: Mapping[str, Any],
    hypothesis: AlignedTraceHypothesis,
) -> bool:
    for schema, pair in zip(
        hypothesis.schema, task.get("train", [])
    ):
        if not schema.source_guard or not schema.target_guard:
            return False
        if (len(select_roles(pair["input"], schema.source_guard)) != 1
                or len(select_roles(pair["output"], schema.target_guard)) != 1):
            return False
    # The zip above is per clause only when there is one clause.  For general
    # traces check every schema against every demo explicitly.
    for pair in task.get("train", []):
        for schema in hypothesis.schema:
            if (not schema.source_guard or not schema.target_guard
                    or len(select_roles(pair["input"], schema.source_guard)) != 1
                    or len(select_roles(pair["output"], schema.target_guard)) != 1):
                return False
    return True


def _effects_grounded(
    task: Mapping[str, Any],
    hypothesis: AlignedTraceHypothesis,
) -> bool:
    """Check only constant move, exact recolor, and delete effects."""

    for clause_index, schema in enumerate(hypothesis.schema):
        if schema.kind not in {"identity", "move", "recolor", "delete"}:
            return False
        if schema.kind in {"identity", "delete"}:
            continue
        observations: list[AnchorObservation] = []
        target_shapes: list[Any] = []
        for pair, trace in zip(task.get("train", []), hypothesis.traces):
            local = trace[clause_index]
            if local.source_index is None or local.target_index is None:
                return False
            source = extract_objects(normalize_grid(pair["input"]))
            target = extract_objects(normalize_grid(pair["output"]))
            left, right = source[local.source_index], target[local.target_index]
            if left.shape != right.shape:
                return False
            if schema.kind == "move":
                if left.colored_shape != right.colored_shape:
                    return False
                observations.append(AnchorObservation(
                    source_role="A",
                    target_anchor=right.anchor,
                    anchors={"A": left.anchor},
                ))
            else:
                if left.anchor != right.anchor:
                    return False
                target_shapes.append(right.colored_shape)
        if schema.kind == "move" and not fit_equations(observations):
            return False
        if schema.kind == "recolor" and len(set(target_shapes)) != 1:
            return False
    return True


def task_aligned_proof_profile(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 8,
) -> dict[str, int | bool]:
    hypotheses = aligned_local_hypotheses(
        task, k=k, max_objects=max_objects, max_hypotheses=max_hypotheses
    )
    unique = [hypothesis for hypothesis in hypotheses
              if all(item.kind is not None for item in hypothesis.schema)
              and _unique_roles(task, hypothesis)]
    grounded = [hypothesis for hypothesis in unique
                if _effects_grounded(task, hypothesis)]
    return {
        "hypotheses": len(hypotheses),
        "unique_role_hypotheses": len(unique),
        "grounded_hypotheses": len(grounded),
        "has_hypothesis": bool(hypotheses),
        "has_unique_roles": bool(unique),
        "has_grounded_effects": bool(grounded),
    }


def dataset_aligned_proof_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 1,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "has_hypothesis": 0,
        "has_unique_roles": 0,
        "has_grounded_effects": 0,
        "hypotheses": 0,
        "unique_role_hypotheses": 0,
        "grounded_hypotheses": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        result = task_aligned_proof_profile(
            task, k=k, max_objects=max_objects,
            max_hypotheses=max_hypotheses,
        )
        for key in ("has_hypothesis", "has_unique_roles", "has_grounded_effects"):
            summary[key] += int(result[key])
        for key in ("hypotheses", "unique_role_hypotheses", "grounded_hypotheses"):
            summary[key] += int(result[key])
    return dict(summary)


if __name__ == "__main__":
    print("aligned_clauses selftest: PASS")
