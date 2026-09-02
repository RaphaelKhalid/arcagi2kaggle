"""Distribution-free calibration for length-normalized decode thresholds."""

from __future__ import annotations

from math import ceil, isfinite
from typing import Iterable


def upper_conformal_quantile(
    scores: Iterable[float],
    *,
    miscoverage: float = 0.1,
) -> float:
    """Return a finite-sample upper conformal quantile.

    For exchangeable calibration scores, the order statistic at rank
    ``ceil((n+1)*(1-alpha))`` has future coverage at least ``1-alpha``.  When
    that rank lies beyond the available sample, returning infinity is the
    honest abstention rather than pretending a tiny fold supports a threshold.
    """

    if not 0.0 < miscoverage < 1.0:
        raise ValueError("miscoverage must lie in (0, 1)")
    values = tuple(float(score) for score in scores)
    if not values or any(not isfinite(score) or score < 0.0 for score in values):
        raise ValueError("scores must be a non-empty finite non-negative sequence")
    rank = ceil((len(values) + 1) * (1.0 - miscoverage))
    if rank > len(values):
        return float("inf")
    return sorted(values)[rank - 1]


def conformal_union_budget(
    absolute_budget: float,
    target_length: int,
    calibration_mean_nll: Iterable[float],
    *,
    miscoverage: float = 0.1,
) -> float:
    """Return `max(absolute, calibrated mean-NLL * length)` budget."""

    if absolute_budget <= 0.0 or target_length < 1:
        raise ValueError("absolute budget and target length must be positive")
    tau = upper_conformal_quantile(
        calibration_mean_nll, miscoverage=miscoverage
    )
    return max(absolute_budget, tau * target_length)


if __name__ == "__main__":
    assert upper_conformal_quantile((0.01, 0.02, 0.03), miscoverage=0.25) == 0.03
    assert conformal_union_budget(1.609, 900, (0.01, 0.02, 0.03), miscoverage=0.25) == 27.0
    print("conformal_threshold selftest: PASS")
