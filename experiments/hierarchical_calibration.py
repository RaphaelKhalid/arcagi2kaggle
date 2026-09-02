"""Hierarchical featurewise calibration for sparse ARC task groups.

Full composite task keys overfit because most feature combinations are rare.
This module uses a global Beta posterior and shrunk one-feature corrections in
log-odds space.  It is a calibration proposal: training-fold outcomes supply
the observations, while deployment features may come from hidden test inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import Mapping


@dataclass(frozen=True)
class Outcome:
    successes: int
    failures: int

    def __post_init__(self) -> None:
        if self.successes < 0 or self.failures < 0:
            raise ValueError("outcomes must be non-negative")

    @property
    def trials(self) -> int:
        return self.successes + self.failures


def _clip_probability(value: float, epsilon: float = 1e-6) -> float:
    return min(1.0 - epsilon, max(epsilon, value))


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = exp(-value)
        return 1.0 / (1.0 + inverse)
    inverse = exp(value)
    return inverse / (1.0 + inverse)


def posterior_mean(
    outcome: Outcome, *, prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> float:
    if prior_alpha <= 0.0 or prior_beta <= 0.0:
        raise ValueError("Beta prior parameters must be positive")
    return (prior_alpha + outcome.successes) / (
        prior_alpha + prior_beta + outcome.trials
    )


def hierarchical_rate(
    global_outcome: Outcome,
    feature_outcomes: Mapping[tuple[str, str], Outcome],
    features: Mapping[str, str],
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    shrinkage: float = 8.0,
) -> float:
    """Estimate a rate with additive shrunk feature corrections.

    For feature value `v`, its posterior is blended toward the global rate by
    `n_v/(n_v + shrinkage)`.  Corrections are averaged in log-odds space so
    contradictory features cannot push the estimate outside [0, 1].
    """

    if shrinkage <= 0.0:
        raise ValueError("shrinkage must be positive")
    global_rate = posterior_mean(
        global_outcome, prior_alpha=prior_alpha, prior_beta=prior_beta
    )
    base_logit = _logit(global_rate)
    corrections: list[float] = []
    for name, value in sorted(features.items()):
        observation = feature_outcomes.get((name, value))
        if observation is None or observation.trials == 0:
            continue
        local_rate = posterior_mean(
            observation, prior_alpha=prior_alpha, prior_beta=prior_beta
        )
        strength = observation.trials / (observation.trials + shrinkage)
        corrections.append(strength * (_logit(local_rate) - base_logit))
    if corrections:
        base_logit += sum(corrections) / len(corrections)
    return _sigmoid(base_logit)


if __name__ == "__main__":
    rate = hierarchical_rate(
        Outcome(5, 5), {("shape", "small"): Outcome(8, 2)}, {"shape": "small"}
    )
    assert rate > 0.5
    print("hierarchical_calibration selftest: PASS", rate)
