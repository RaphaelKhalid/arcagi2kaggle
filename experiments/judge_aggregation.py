"""Deterministic council aggregation for holistic ARC candidate judges.

The MDS result motivates three independent judges that each return a ranked
pair.  This module contains only the final, auditable aggregation step: a
judge's first choice receives two points and its second choice one point.
Candidate traces and synthesis remain outside this CPU-only seam.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def aggregate_ranked_pairs(
    judge_pairs: Iterable[tuple[str, ...]],
    *,
    first_weight: float = 2.0,
    second_weight: float = 1.0,
    judge_groups: Iterable[str | None] | None = None,
) -> tuple[str, ...]:
    """Return the two highest-scoring distinct output classes.

    Each input pair is interpreted as ranked: position 0 is the judge's first
    choice and position 1 is its second.  Extra entries are rejected so that a
    malformed judge cannot silently change the pass@2 objective.  Ties are
    deterministic and prefer the class with broader judge support, then its
    lexical id.
    """

    if first_weight < 0 or second_weight < 0:
        raise ValueError("judge weights must be non-negative")
    pairs = tuple(judge_pairs)
    if judge_groups is None:
        group_weights = [1.0] * len(pairs)
    else:
        groups = tuple(judge_groups)
        if len(groups) != len(pairs):
            raise ValueError("judge_groups must match judge_pairs")
        counts: dict[str, int] = {}
        for index, group in enumerate(groups):
            key = group if group is not None else f"__judge_{index}"
            counts[key] = counts.get(key, 0) + 1
        group_weights = [
            1.0 / counts[group if group is not None else f"__judge_{index}"]
            for index, group in enumerate(groups)
        ]
    scores: dict[str, float] = defaultdict(float)
    support: dict[str, float] = defaultdict(float)
    for pair, group_weight in zip(pairs, group_weights):
        if len(pair) != 2:
            raise ValueError("each judge must return exactly two classes")
        if pair[0] == pair[1]:
            raise ValueError("a judge pair must contain distinct classes")
        scores[pair[0]] += group_weight * first_weight
        scores[pair[1]] += group_weight * second_weight
        support[pair[0]] += group_weight
        support[pair[1]] += group_weight
    ranked = sorted(scores, key=lambda output: (
        -scores[output], -support[output], output
    ))
    return tuple(ranked[:2])


def position_debiased_aggregation(
    judge_pairs: Iterable[tuple[str, ...]],
    *,
    judge_groups: Iterable[str | None] | None = None,
) -> tuple[str, ...]:
    """Aggregate pairs after ignoring judge-specific list position.

    This is a diagnostic control, not the MDS method: it gives each class one
    vote per appearance and exposes whether first-choice weighting is driving
    the result.  It is useful when checking position/verbosity bias.
    """

    return aggregate_ranked_pairs(
        judge_pairs,
        first_weight=1.0,
        second_weight=1.0,
        judge_groups=judge_groups,
    )


if __name__ == "__main__":
    assert set(aggregate_ranked_pairs(
        [("minority", "modal"), ("minority", "modal"), ("modal", "other")]
    )) == {"minority", "modal"}
    print("judge_aggregation selftest: PASS")
