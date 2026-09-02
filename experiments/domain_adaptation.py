"""Leakage-safe covariate-shift calibration for the private test distribution.

The hidden test *inputs* are present in the competition challenge file while
their outputs are not.  If solver success is approximately stable conditional
on an input-only structural group, training-fold outcomes can be reweighted to
the hidden group's frequency.  This module provides the estimator and makes
its uncertainty visible; it never consumes hidden solutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Mapping


def smoothed_distribution(
    counts: Mapping[str, int], *, alpha: float = 0.5
) -> dict[str, float]:
    """Dirichlet-smoothed distribution over the observed union of groups."""

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    keys = tuple(sorted(counts))
    if any(counts[key] < 0 for key in keys):
        raise ValueError("counts must be non-negative")
    if not keys:
        return {}
    denominator = fsum(float(counts[key]) + alpha for key in keys)
    if denominator <= 0.0:
        raise ValueError("counts and alpha cannot both be zero")
    return {key: (counts[key] + alpha) / denominator for key in keys}


def importance_weights(
    source_counts: Mapping[str, int],
    target_counts: Mapping[str, int],
    *,
    alpha: float = 0.5,
    max_weight: float = 5.0,
) -> dict[str, float]:
    """Return capped target/source group-frequency ratios."""

    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")
    keys = tuple(sorted(set(source_counts) | set(target_counts)))
    source = smoothed_distribution({key: source_counts.get(key, 0) for key in keys}, alpha=alpha)
    target = smoothed_distribution({key: target_counts.get(key, 0) for key in keys}, alpha=alpha)
    return {key: min(max_weight, target[key] / source[key]) for key in keys}


@dataclass(frozen=True)
class GroupOutcome:
    successes: int
    failures: int

    def __post_init__(self) -> None:
        if self.successes < 0 or self.failures < 0:
            raise ValueError("outcomes must be non-negative")


def target_success_rate(
    outcomes: Mapping[str, GroupOutcome],
    target_counts: Mapping[str, int],
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> float:
    """Estimate target success using groupwise Beta posteriors.

    Groups absent from the labeled source use the pooled Beta posterior.  This
    is deliberately conservative: it avoids inventing a zero or one rate for
    a private group unseen during calibration.
    """

    if prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("Beta priors must be positive")
    target = smoothed_distribution(target_counts, alpha=0.0)
    total_successes = sum(item.successes for item in outcomes.values())
    total_failures = sum(item.failures for item in outcomes.values())
    pooled = (prior_alpha + total_successes) / (
        prior_alpha + prior_beta + total_successes + total_failures
    )
    result = 0.0
    for group, probability in target.items():
        observation = outcomes.get(group)
        if observation is None or observation.successes + observation.failures == 0:
            estimate = pooled
        else:
            estimate = (prior_alpha + observation.successes) / (
                prior_alpha + prior_beta + observation.successes + observation.failures
            )
        result += probability * estimate
    return result


def effective_sample_size(weights: Mapping[str, float]) -> float:
    """Kish effective sample size for capped importance weights."""

    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError("weights must be non-negative")
    total = fsum(weights.values())
    squares = fsum(weight * weight for weight in weights.values())
    return 0.0 if squares == 0.0 else total * total / squares


if __name__ == "__main__":
    weights = importance_weights({"a": 9, "b": 1}, {"a": 1, "b": 9})
    assert weights["b"] > weights["a"]
    print("domain_adaptation selftest: PASS", weights)
