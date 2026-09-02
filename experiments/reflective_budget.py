"""Budget calculus for verifier-guided ARC candidate repair.

The module is deliberately model- and executor-independent.  It answers one
question for a fixed two-stage budget: after an initial proposal fails, is a
targeted repair more valuable per second than a fresh proposal?  Rates must be
estimated from leakage-safe folds or a shadow run; no rate is inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SecondStageDecision:
    """Marginal rescue comparison after the first proposal has failed."""

    fresh_gain: float
    repair_gain: float
    fresh_gain_per_second: float
    repair_gain_per_second: float
    use_repair: bool


def validate_rate(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


def second_stage_decision(
    fresh_success: float,
    repair_success: float,
    *,
    fresh_seconds: float,
    repair_seconds: float,
    repair_novelty: float = 1.0,
) -> SecondStageDecision:
    """Choose repair when its conditional rescue per second is larger.

    ``fresh_success`` is p: the success rate of a new proposal.  ``repair_success``
    is q: the success rate conditional on an actionable failed candidate and its
    bounded first-divergence feedback. ``repair_novelty`` is the fraction of
    repair successes expected to contribute a distinct useful output class rather
    than a correlated copy of an existing basin. Both are marginal gains after
    failure, so the common failure factor (1-p) cancels from the decision.
    """

    p = validate_rate(fresh_success, "fresh_success")
    q = validate_rate(repair_success, "repair_success")
    novelty = validate_rate(repair_novelty, "repair_novelty")
    if fresh_seconds <= 0.0 or repair_seconds <= 0.0:
        raise ValueError("stage costs must be positive")
    return SecondStageDecision(
        fresh_gain=p,
        repair_gain=q * novelty,
        fresh_gain_per_second=p / fresh_seconds,
        repair_gain_per_second=(q * novelty) / repair_seconds,
        use_repair=((q * novelty) / repair_seconds) > (p / fresh_seconds),
    )


def paired_success_probability(
    fresh_success: float, repair_success: float
) -> float:
    """Success probability for one proposal followed by conditional repair."""

    p = validate_rate(fresh_success, "fresh_success")
    q = validate_rate(repair_success, "repair_success")
    return p + (1.0 - p) * q


def independent_pair_probability(fresh_success: float) -> float:
    """Success probability for two independent fresh proposals."""

    p = validate_rate(fresh_success, "fresh_success")
    return 1.0 - (1.0 - p) ** 2


if __name__ == "__main__":
    decision = second_stage_decision(0.2, 0.4, fresh_seconds=2, repair_seconds=2)
    assert decision.use_repair
    assert paired_success_probability(0.2, 0.4) > independent_pair_probability(0.2)
    print("reflective_budget selftest: PASS")
