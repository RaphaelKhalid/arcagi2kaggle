"""Bounded exact alignment of action traces before graph anti-unification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import islice, product
import json
import sys
from typing import Any, Iterable, Mapping

try:
    from experiments.graph_lgg import (
        ActionObservation,
        ActionSchema,
        lgg_observations,
        observations_for_correspondence,
    )
    from experiments.object_correspondence import top_k_correspondences
except ModuleNotFoundError:  # direct ``python experiments/trace_alignment.py``
    from graph_lgg import ActionObservation, ActionSchema, lgg_observations, observations_for_correspondence
    from object_correspondence import top_k_correspondences


@dataclass(frozen=True)
class TraceAlignment:
    pairs: tuple[tuple[int, int], ...]
    unmatched_reference: tuple[int, ...]
    unmatched_other: tuple[int, ...]
    cost: int


def _guard_distance(
    left: frozenset[tuple[Any, ...]],
    right: frozenset[tuple[Any, ...]],
) -> int:
    return len(left.symmetric_difference(right))


def action_alignment_cost(
    left: ActionObservation,
    right: ActionObservation,
) -> int:
    """Cost preserves action semantics while tolerating demo-specific guards."""

    cost = 0
    cost += 8 * (left.kind != right.kind)
    cost += 2 * (left.motion_axis != right.motion_axis)
    cost += 2 * (left.shape_relation != right.shape_relation)
    cost += 1 * (left.area_relation != right.area_relation)
    cost += 1 * (left.palette_relation != right.palette_relation)
    # Guards are evidence, not identity.  A small clipped penalty prevents a
    # wildly different role from winning a tie without requiring exact guards.
    cost += min(3, _guard_distance(left.source_guard, right.source_guard))
    return cost


def align_traces(
    reference: Iterable[ActionObservation],
    other: Iterable[ActionObservation],
    *,
    k: int = 4,
    unmatched_penalty: int = 7,
) -> tuple[TraceAlignment, ...]:
    """Return up to k minimum-cost partial action assignments."""

    reference = tuple(reference)
    other = tuple(other)
    if k <= 0 or unmatched_penalty < 0:
        raise ValueError("k must be positive and unmatched_penalty non-negative")
    states: dict[int, list[tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]]] = {
        0: [(0, (), ())]
    }
    for reference_index, reference_action in enumerate(reference):
        next_states: dict[int, list[tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]]] = {}
        for mask, candidates in states.items():
            for partial_cost, pairs, unmatched_reference in candidates:
                next_states.setdefault(mask, []).append((
                    partial_cost + unmatched_penalty,
                    pairs,
                    unmatched_reference + (reference_index,),
                ))
                for other_index, other_action in enumerate(other):
                    if mask & (1 << other_index):
                        continue
                    new_mask = mask | (1 << other_index)
                    next_states.setdefault(new_mask, []).append((
                        partial_cost + action_alignment_cost(
                            reference_action, other_action
                        ),
                        pairs + ((reference_index, other_index),),
                        unmatched_reference,
                    ))
        states = {
            mask: sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[:k]
            for mask, candidates in next_states.items()
        }
    result: list[TraceAlignment] = []
    all_other = set(range(len(other)))
    for mask, candidates in states.items():
        for cost, pairs, unmatched_reference in candidates:
            used = {other_index for _, other_index in pairs}
            result.append(TraceAlignment(
                pairs=pairs,
                unmatched_reference=unmatched_reference,
                unmatched_other=tuple(sorted(all_other - used)),
                cost=cost + unmatched_penalty * len(all_other - used),
            ))
    result.sort(key=lambda item: (
        item.cost, item.pairs, item.unmatched_reference, item.unmatched_other
    ))
    unique: list[TraceAlignment] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for alignment in result:
        if alignment.pairs in seen:
            continue
        seen.add(alignment.pairs)
        unique.append(alignment)
        if len(unique) == k:
            break
    return tuple(unique)


def _full_alignments(
    reference: tuple[ActionObservation, ...],
    other: tuple[ActionObservation, ...],
    *,
    k: int,
) -> tuple[TraceAlignment, ...]:
    return tuple(
        alignment for alignment in align_traces(reference, other, k=k)
        if (not alignment.unmatched_reference
            and not alignment.unmatched_other
            and len(alignment.pairs) == len(reference))
    )


def aligned_lgg_candidates(
    traces: Iterable[Iterable[ActionObservation]],
    *,
    k: int = 4,
    max_hypotheses: int = 32,
) -> tuple[tuple[ActionSchema, ...], ...]:
    """Anti-unify traces after bounded pairwise clause alignment."""

    traces = tuple(tuple(trace) for trace in traces)
    if not traces:
        return ()
    reference = traces[0]
    alignment_options: list[tuple[TraceAlignment, ...]] = []
    for trace in traces[1:]:
        options = _full_alignments(reference, trace, k=k)
        if not options:
            return ()
        alignment_options.append(options)
    schemas: list[tuple[ActionSchema, ...]] = []
    seen: set[tuple[ActionSchema, ...]] = set()
    for choices in product(*alignment_options):
        aligned: list[tuple[ActionObservation, ...]] = [reference]
        for trace, choice in zip(traces[1:], choices):
            by_reference = {i: j for i, j in choice.pairs}
            aligned.append(tuple(trace[by_reference[i]] for i in range(len(reference))))
        schema = lgg_observations(aligned)
        if schema is None or schema in seen:
            continue
        seen.add(schema)
        schemas.append(schema)
        if len(schemas) == max_hypotheses:
            break
    return tuple(schemas)


def task_alignment_profile(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 32,
) -> dict[str, int | bool]:
    """Compare sorted top-1 LGG with aligned top-1 and top-k alternatives."""

    pairs = task.get("train", [])
    if not pairs:
        return {"bounded": False, "sorted_lgg": False,
                "aligned_lgg": False, "topk_aligned_lgg": False,
                "aligned_schema_count": 0}
    top1_traces: list[tuple[ActionObservation, ...]] = []
    topk_traces: list[tuple[tuple[ActionObservation, ...], ...]] = []
    try:
        for pair in pairs:
            correspondences = top_k_correspondences(
                pair["input"], pair["output"], k=k, max_objects=max_objects
            )
            if not correspondences:
                return {"bounded": False, "sorted_lgg": False,
                        "aligned_lgg": False, "topk_aligned_lgg": False,
                        "aligned_schema_count": 0}
            traces = tuple(observations_for_correspondence(
                pair["input"], pair["output"], correspondence
            ) for correspondence in correspondences)
            top1_traces.append(traces[0])
            topk_traces.append(traces)
    except ValueError:
        return {"bounded": False, "sorted_lgg": False,
                "aligned_lgg": False, "topk_aligned_lgg": False,
                "aligned_schema_count": 0}
    sorted_schema = lgg_observations(top1_traces)
    aligned = aligned_lgg_candidates(
        top1_traces, k=k, max_hypotheses=max_hypotheses
    )
    # Keep the first demo fixed and choose one of the bounded alternatives for
    # every other demo.  The product is capped so this remains an anytime pass.
    topk_options: list[tuple[tuple[ActionObservation, ...], ...]] = []
    reference = topk_traces[0][0]
    for traces in topk_traces[1:]:
        valid: list[tuple[ActionObservation, ...]] = []
        for trace in traces:
            if _full_alignments(reference, trace, k=k):
                valid.append(trace)
        topk_options.append(tuple(valid))
    topk_schemas: list[tuple[ActionSchema, ...]] = []
    if all(topk_options):
        for choices in islice(product(*topk_options), max_hypotheses):
            candidates = aligned_lgg_candidates(
                (reference,) + choices, k=k, max_hypotheses=max_hypotheses
            )
            topk_schemas.extend(candidates)
            if len(topk_schemas) >= max_hypotheses:
                break
    return {
        "bounded": True,
        "sorted_lgg": sorted_schema is not None,
        "aligned_lgg": bool(aligned),
        "topk_aligned_lgg": bool(topk_schemas),
        "aligned_schema_count": len(set(aligned)),
    }


def dataset_alignment_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 32,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "bounded": 0,
        "sorted_lgg": 0,
        "aligned_lgg": 0,
        "topk_aligned_lgg": 0,
        "aligned_schema_count": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        result = task_alignment_profile(
            task, k=k, max_objects=max_objects,
            max_hypotheses=max_hypotheses,
        )
        for key in ("bounded", "sorted_lgg", "aligned_lgg", "topk_aligned_lgg"):
            summary[key] += int(result[key])
        summary["aligned_schema_count"] += int(result["aligned_schema_count"])
    return dict(summary)


if __name__ == "__main__":
    left = (ActionObservation(
        "move", "horizontal", "same", "same", "same",
        frozenset({("area", 1)}), frozenset({("area", 1)}),
    ),)
    right = (ActionObservation(
        "move", "horizontal", "same", "same", "same",
        frozenset({("area", 1)}), frozenset({("area", 1)}),
    ),)
    assert align_traces(left, right)[0].pairs == ((0, 0),)
    assert aligned_lgg_candidates((left, right))
    print("trace_alignment selftest: PASS")
