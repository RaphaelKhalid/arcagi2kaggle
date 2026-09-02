"""Conservative solver-lane routing under covariate shift.

The router is CPU-only and consumes leakage-safe fold counts.  A Wilson lower
bound describes uncertainty in a source-fold hit rate; a declared total
variation radius then gives a worst-case target lower bound because every
event probability can change by at most that radius.  An observed train/test
TV distance is not automatically a certified radius and must be supplied
explicitly by the caller as a stress assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, sqrt
from typing import Iterable, Mapping


def wilson_lower_bound(
    successes: int, trials: int, *, z: float = 1.96
) -> float:
    """Return a two-sided Wilson lower confidence bound for a Bernoulli rate."""

    if successes < 0 or trials < 0 or successes > trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if z <= 0.0:
        raise ValueError("z must be positive")
    if trials == 0:
        return 0.0
    n = float(trials)
    p = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    radius = z * sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, min(1.0, (center - radius) / denominator))


@dataclass(frozen=True)
class ShiftRobustLane:
    """A proposal lane with a conservative target-success estimate."""

    name: str
    successes: int
    trials: int
    cost_seconds: float
    shift_radius: float = 0.0
    selector_recovery: float = 1.0
    z: float = 1.96

    def __post_init__(self) -> None:
        if not self.name or self.successes < 0 or self.trials < 0:
            raise ValueError("invalid lane identity or counts")
        if self.successes > self.trials:
            raise ValueError("successes cannot exceed trials")
        if self.cost_seconds <= 0.0:
            raise ValueError("cost_seconds must be positive")
        if not 0.0 <= self.shift_radius <= 1.0:
            raise ValueError("shift_radius must lie in [0, 1]")
        if not 0.0 <= self.selector_recovery <= 1.0:
            raise ValueError("selector_recovery must lie in [0, 1]")
        if self.z <= 0.0:
            raise ValueError("z must be positive")

    @property
    def source_lower_rate(self) -> float:
        return wilson_lower_bound(self.successes, self.trials, z=self.z)

    @property
    def target_lower_rate(self) -> float:
        """Worst-case rate under the caller-declared TV shift radius."""

        return max(0.0, self.source_lower_rate - self.shift_radius)

    @property
    def lower_gain_per_second(self) -> float:
        """Conservative useful-gain rate after selector recovery."""

        return (
            self.target_lower_rate * self.selector_recovery
            / self.cost_seconds
        )


@dataclass(frozen=True)
class RobustRoute:
    lane_names: tuple[str, ...]
    lower_gain_per_second: tuple[float, ...]


def groupwise_target_lower_rate(
    group_lanes: Mapping[str, ShiftRobustLane],
    target_mass: Mapping[str, float],
) -> float:
    """Return a target-weighted lower rate from conditional group bounds.

    A target-visible group with no source evidence receives zero lower mass.
    This is conservative and keeps unsupported target strata explicit.
    """

    if not target_mass:
        raise ValueError("target_mass must be non-empty")
    if any(mass < 0.0 for mass in target_mass.values()):
        raise ValueError("target masses must be non-negative")
    if abs(fsum(target_mass.values()) - 1.0) > 1e-9:
        raise ValueError("target masses must sum to one")
    return fsum(
        mass * group_lanes[group].target_lower_rate
        if group in group_lanes
        else 0.0
        for group, mass in target_mass.items()
    )


def rank_shift_robust_lanes(
    lanes: Iterable[ShiftRobustLane],
    *,
    unresolved_mass: float = 1.0,
) -> RobustRoute:
    """Rank lanes by conservative unresolved-output gain per second."""

    if not 0.0 <= unresolved_mass <= 1.0:
        raise ValueError("unresolved_mass must lie in [0, 1]")
    lanes = tuple(lanes)
    ranked = sorted(
        lanes,
        key=lambda lane: (
            -unresolved_mass * lane.lower_gain_per_second,
            -lane.target_lower_rate,
            lane.cost_seconds,
            lane.name,
        ),
    )
    return RobustRoute(
        tuple(lane.name for lane in ranked),
        tuple(unresolved_mass * lane.lower_gain_per_second for lane in ranked),
    )


if __name__ == "__main__":
    lane = ShiftRobustLane("program", successes=8, trials=10,
                           cost_seconds=2.0, shift_radius=0.1)
    assert 0.0 < lane.target_lower_rate < lane.source_lower_rate
    route = rank_shift_robust_lanes([lane])
    assert route.lane_names == ("program",)
    print("shift_robust_router selftest: PASS")
