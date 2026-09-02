"""Adaptive candidate-generation routing for the ARC pass@2 objective.

This module turns early stopping into a value-of-information calculation.  A
target task has no labels, so the rates passed here must come from leakage-safe
fold calibration; the module itself only consumes current output-class mass and
lane estimates.  It is intentionally independent of any model or executor.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import exp, fsum, log


@dataclass(frozen=True)
class DistributionStats:
    top_two_mass: float
    entropy: float
    effective_classes: float
    observed_classes: int


@dataclass(frozen=True)
class RouteDecision:
    current_utility: float
    unresolved_mass: float
    expected_gain: float
    gain_per_second: float
    continue_search: bool


def distribution_stats(
    masses: Mapping[str, float], *, unknown_mass: float = 0.0
) -> DistributionStats:
    """Summarize known output classes plus an unobserved-class reserve.

    ``unknown_mass`` is not a selectable output.  It represents calibrated
    probability that the correct class is absent from the current candidate
    set, so it contributes to uncertainty/entropy but not to top-two utility.
    """

    if not 0.0 <= unknown_mass <= 1.0:
        raise ValueError("unknown_mass must lie in [0, 1]")
    raw = [max(0.0, float(value)) for value in masses.values()]
    total = fsum(raw)
    if total <= 0.0:
        if unknown_mass <= 0.0:
            return DistributionStats(0.0, 0.0, 0.0, 0)
        return DistributionStats(0.0, 0.0, 1.0, 0)
    known = [value / total * (1.0 - unknown_mass)
             for value in raw if value > 0.0]
    known.sort(reverse=True)
    entropy_terms = known + ([unknown_mass] if unknown_mass > 0.0 else [])
    entropy = -fsum(probability * log(probability) for probability in entropy_terms)
    return DistributionStats(
        top_two_mass=min(1.0, fsum(known[:2])),
        entropy=entropy,
        effective_classes=exp(entropy),
        observed_classes=len(known),
    )


def decide_route(
    positions: Iterable[Mapping[str, float]],
    *,
    novelty_rate: float,
    selector_recovery: float = 1.0,
    lane_seconds: float,
    min_gain_per_second: float = 0.0,
    unknown_mass: float | Iterable[float] = 0.0,
) -> RouteDecision:
    """Decide whether another candidate lane has positive expected value.

    For each output position the current Bayes pass@2 utility is the mass of
    its two best classes.  The unresolved mass is the remainder.  If the next
    lane discovers a useful new class with calibrated rate ``novelty_rate`` and
    the selector retains it with probability ``selector_recovery``, its
    expected rescued-output count is their product times unresolved mass.
    """

    if not 0.0 <= novelty_rate <= 1.0:
        raise ValueError("novelty_rate must lie in [0, 1]")
    if not 0.0 <= selector_recovery <= 1.0:
        raise ValueError("selector_recovery must lie in [0, 1]")
    positions = tuple(positions)
    return decide_route_positionwise(
        positions,
        novelty_rates=[novelty_rate] * len(positions),
        selector_recovery=[selector_recovery] * len(positions),
        lane_seconds=lane_seconds,
        min_gain_per_second=min_gain_per_second,
        unknown_mass=unknown_mass,
    )


def decide_route_positionwise(
    positions: Iterable[Mapping[str, float]],
    *,
    novelty_rates: Iterable[float],
    selector_recovery: float | Iterable[float] = 1.0,
    lane_seconds: float,
    min_gain_per_second: float = 0.0,
    unknown_mass: float | Iterable[float] = 0.0,
) -> RouteDecision:
    """Route using position-conditioned discovery and recovery rates.

    ``novelty_rates[j]`` is the calibrated probability that this lane adds the
    correct class for position ``j`` conditional on that position remaining
    unresolved.  This is strictly more expressive than one global rate: a
    lane can be valuable on a structural subgroup while being redundant
    elsewhere.  Rates must be estimated from leakage-safe folds.
    """

    positions = tuple(positions)
    rates = tuple(float(value) for value in novelty_rates)
    if len(rates) != len(positions):
        raise ValueError("novelty_rates iterable must match positions")
    if isinstance(selector_recovery, (int, float)):
        recoveries = (float(selector_recovery),) * len(positions)
    else:
        recoveries = tuple(float(value) for value in selector_recovery)
        if len(recoveries) != len(positions):
            raise ValueError("selector_recovery iterable must match positions")
    if isinstance(unknown_mass, (int, float)):
        unknowns = (float(unknown_mass),) * len(positions)
    else:
        unknowns = tuple(float(value) for value in unknown_mass)
        if len(unknowns) != len(positions):
            raise ValueError("unknown_mass iterable must match positions")
    if any(not 0.0 <= value <= 1.0 for value in rates):
        raise ValueError("novelty_rates must lie in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in recoveries):
        raise ValueError("selector_recovery must lie in [0, 1]")
    if lane_seconds <= 0.0:
        raise ValueError("lane_seconds must be positive")
    if min_gain_per_second < 0.0:
        raise ValueError("min_gain_per_second must be non-negative")
    stats = [distribution_stats(position, unknown_mass=unknown)
             for position, unknown in zip(positions, unknowns)]
    current = fsum(stat.top_two_mass for stat in stats)
    unresolved = fsum(1.0 - stat.top_two_mass for stat in stats)
    expected = fsum(
        (1.0 - stat.top_two_mass) * rate * recovery
        for stat, rate, recovery in zip(stats, rates, recoveries)
    )
    gain_per_second = expected / lane_seconds
    return RouteDecision(
        current_utility=current,
        unresolved_mass=unresolved,
        expected_gain=expected,
        gain_per_second=gain_per_second,
        continue_search=gain_per_second > min_gain_per_second,
    )


if __name__ == "__main__":
    decision = decide_route(
        [{"a": 0.8, "b": 0.1, "c": 0.1}],
        novelty_rate=0.5, lane_seconds=10,
    )
    assert decision.expected_gain > 0.0
    print("adaptive_routing selftest: PASS", decision)
