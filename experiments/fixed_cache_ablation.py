"""Fixed-cache ablation metrics for multi-lineage ARC proposals.

The report is deliberately cache-only: it never generates a candidate and it
can be run without solutions for hidden-input inventory.  When a disjoint
labeled fold is supplied, it measures candidate-class recall separately from
the selected pass@2 score.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

try:
    from experiments.candidate_records import CandidateRecord, normalize_grid
    from experiments.replay_harness import replay_score
except ModuleNotFoundError:  # direct ``python experiments/fixed_cache_ablation.py``
    from candidate_records import CandidateRecord, normalize_grid
    from replay_harness import replay_score


Position = tuple[str, int]


@dataclass(frozen=True)
class FixedCacheReport:
    raw_records: int
    positions_with_records: int
    output_classes: int
    family_records: dict[str, int]
    family_pairwise_class_jaccard: dict[str, float]
    candidate_recall: tuple[int, int] | None
    selected_score: float | None


def _class_keys(records: list[CandidateRecord]) -> dict[str, set[tuple[Position, str]]]:
    by_family: dict[str, set[tuple[Position, str]]] = defaultdict(set)
    for record in records:
        by_family[record.family].add(
            ((record.task_id, record.test_index), record.output_hash)
        )
    return dict(by_family)


def fixed_cache_report(
    records: list[CandidateRecord],
    *,
    solutions: Mapping[str, list[Any]] | None = None,
) -> FixedCacheReport:
    """Summarize a frozen cache and optionally score it on a labeled fold."""

    positions = {
        (record.task_id, record.test_index)
        for record in records
    }
    classes = {
        (record.task_id, record.test_index, record.output_hash)
        for record in records
    }
    family_records = Counter(record.family for record in records)
    family_classes = _class_keys(records)
    pairwise: dict[str, float] = {}
    for left, right in combinations(sorted(family_classes), 2):
        left_values, right_values = family_classes[left], family_classes[right]
        union = left_values | right_values
        pairwise[f"{left}|{right}"] = (
            len(left_values & right_values) / len(union) if union else 0.0
        )

    candidate_recall: tuple[int, int] | None = None
    selected_score: float | None = None
    if solutions is not None:
        by_position: dict[Position, set[str]] = defaultdict(set)
        for record in records:
            by_position[(record.task_id, record.test_index)].add(
                record.output_hash
            )
        hits = total = 0
        for task_id, outputs in solutions.items():
            for test_index, output in enumerate(outputs):
                total += 1
                truth = CandidateRecord.from_output(
                    task_id=task_id,
                    test_index=test_index,
                    family="truth",
                    candidate_id="truth",
                    output=normalize_grid(output),
                )
                hits += truth.output_hash in by_position.get(
                    (task_id, test_index), set()
                )
        candidate_recall = (hits, total)
        selected_score = replay_score(records, solutions=solutions)

    return FixedCacheReport(
        raw_records=len(records),
        positions_with_records=len(positions),
        output_classes=len(classes),
        family_records=dict(sorted(family_records.items())),
        family_pairwise_class_jaccard=dict(sorted(pairwise.items())),
        candidate_recall=candidate_recall,
        selected_score=selected_score,
    )


if __name__ == "__main__":
    records = [CandidateRecord.from_output(
        task_id="t", test_index=0, family="a", candidate_id="a",
        output=[[1]],
    )]
    assert fixed_cache_report(records).output_classes == 1
    print("fixed_cache_ablation selftest: PASS")
