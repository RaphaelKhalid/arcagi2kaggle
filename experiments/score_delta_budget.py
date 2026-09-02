"""Exact decomposition of pass@2 score changes into coverage and recovery."""

from __future__ import annotations

from dataclasses import dataclass


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def score_from_coverage(coverage: float, selector_recovery: float) -> float:
    """Return output-level score under coverage/recovery decomposition."""

    return _probability(coverage, "coverage") * _probability(
        selector_recovery, "selector_recovery"
    )


@dataclass(frozen=True)
class ScoreDelta:
    baseline_score: float
    candidate_score: float
    coverage_component: float
    recovery_component: float
    interaction_component: float

    @property
    def total(self) -> float:
        return self.candidate_score - self.baseline_score


def decompose_score_delta(
    baseline_coverage: float,
    baseline_recovery: float,
    candidate_coverage: float,
    candidate_recovery: float,
) -> ScoreDelta:
    """Decompose an exact score delta using the baseline/candidate path."""

    c0 = _probability(baseline_coverage, "baseline_coverage")
    r0 = _probability(baseline_recovery, "baseline_recovery")
    c1 = _probability(candidate_coverage, "candidate_coverage")
    r1 = _probability(candidate_recovery, "candidate_recovery")
    return ScoreDelta(
        baseline_score=c0 * r0,
        candidate_score=c1 * r1,
        coverage_component=(c1 - c0) * r0,
        recovery_component=c0 * (r1 - r0),
        interaction_component=(c1 - c0) * (r1 - r0),
    )


def minimum_recovery_for_gain(
    coverage: float,
    baseline_coverage: float,
    baseline_recovery: float,
    target_gain: float,
) -> float:
    """Return the recovery required at fixed coverage for a target gain."""

    coverage = _probability(coverage, "coverage")
    baseline_coverage = _probability(baseline_coverage, "baseline_coverage")
    baseline_recovery = _probability(baseline_recovery, "baseline_recovery")
    if target_gain < 0.0:
        raise ValueError("target_gain must be non-negative")
    if coverage == 0.0:
        return float("inf")
    return (baseline_coverage * baseline_recovery + target_gain) / coverage


if __name__ == "__main__":
    delta = decompose_score_delta(0.8, 0.5, 0.9, 0.6)
    assert abs(delta.total - (
        delta.coverage_component + delta.recovery_component
        + delta.interaction_component
    )) < 1e-12
    print("score_delta_budget selftest: PASS", delta.total)
