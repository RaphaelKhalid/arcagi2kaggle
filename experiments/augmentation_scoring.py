"""Invariant likelihood scoring under a finite augmentation/nuisance group."""

from __future__ import annotations

from math import exp, log
from typing import Iterable

try:
    from experiments.candidate_records import CandidateRecord
    from experiments.pass2_selector import Candidate
except ModuleNotFoundError:  # direct ``python experiments/augmentation_scoring.py``
    from candidate_records import CandidateRecord
    from pass2_selector import Candidate


def log_mean_exp(values: Iterable[float]) -> float:
    """Stable log(mean(exp(values))) for a non-empty finite sequence."""

    values = tuple(float(value) for value in values)
    if not values:
        raise ValueError("log_mean_exp needs at least one value")
    pivot = max(values)
    return pivot + log(sum(exp(value - pivot) for value in values) / len(values))


def augmentation_log_marginal(nlls: Iterable[float]) -> float:
    """Marginal log likelihood when one augmentation is uniformly sampled."""

    return log_mean_exp(-float(nll) for nll in nlls)


def record_log_marginal(record: CandidateRecord) -> float:
    """Score a cache record using preserved per-augmentation NLLs."""

    if not record.augmentation_nlls:
        if record.augmentation_score is None:
            raise ValueError("record has no augmentation likelihood")
        return -float(record.augmentation_score)
    return augmentation_log_marginal(record.augmentation_nlls)


def best_class_by_augmentation_marginal(
    records: Iterable[CandidateRecord],
) -> tuple[str, ...]:
    """Rank output classes by the best representative's marginal."""

    best: dict[str, float] = {}
    for record in records:
        score = record_log_marginal(record)
        best[record.output_hash] = max(best.get(record.output_hash, float("-inf")), score)
    return tuple(sorted(best, key=lambda output: (-best[output], output)))


def records_to_marginal_candidates(
    records: Iterable[CandidateRecord],
) -> list[Candidate]:
    """Collapse records to one likelihood-weighted candidate per family/class."""

    grouped: dict[str, dict[str, tuple[float, CandidateRecord]]] = {}
    for record in records:
        score = record_log_marginal(record)
        family = grouped.setdefault(record.family, {})
        old = family.get(record.output_hash)
        if old is None or score > old[0]:
            family[record.output_hash] = (score, record)
    result: list[Candidate] = []
    for family, classes in grouped.items():
        if not classes:
            continue
        pivot = max(score for score, _ in classes.values())
        for output_hash, (score, record) in classes.items():
            result.append(Candidate(
                output_hash=output_hash,
                family=family,
                weight=exp(score - pivot),
                mdl_length=record.mdl_length,
                hard_valid=record.hard_valid,
            ))
    return result


if __name__ == "__main__":
    assert abs(augmentation_log_marginal((0.0, 0.0))) < 1e-12
    print("augmentation_scoring selftest: PASS")
