"""Complementarity-aware proposal-lane allocation.

Independent geometric decay is a useful cold-start approximation, but it does
not distinguish a genuinely complementary family from a duplicate checkpoint.
This module uses the monotone submodular coverage surrogate

    F(S) = sum_i [1 - product_{f in S} (1 - q[f, i])]

where ``i`` indexes held-out task positions and ``q`` is a calibrated chance
that a lane yields a useful class.  Greedy marginal-gain-per-cost selection is
deterministic and respects a hard budget.  The model is still an approximation:
rates should be estimated after deduplicating output classes and can be replaced
by measured conditional rates when a real cache exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Iterable


@dataclass(frozen=True)
class ProposalLane:
    name: str
    success_rates: tuple[float, ...]
    cost: float


def _validate_lane(lane: ProposalLane) -> None:
    if lane.cost <= 0.0:
        raise ValueError("lane cost must be positive")
    if any(not 0.0 <= rate <= 1.0 for rate in lane.success_rates):
        raise ValueError("success rates must lie in [0, 1]")


def expected_coverage(
    lanes: Iterable[ProposalLane],
) -> float:
    """Return expected number of covered task positions under the surrogate."""

    lanes = tuple(lanes)
    for lane in lanes:
        _validate_lane(lane)
    lengths = {len(lane.success_rates) for lane in lanes}
    if len(lengths) > 1:
        raise ValueError("all lanes must cover the same number of positions")
    if not lanes:
        return 0.0
    return fsum(
        1.0 - _residual_probability(lanes, index)
        for index in range(len(lanes[0].success_rates))
    )


def _residual_probability(lanes: Iterable[ProposalLane], index: int) -> float:
    residual = 1.0
    for lane in lanes:
        residual *= 1.0 - lane.success_rates[index]
    return residual


def marginal_coverage(
    lane: ProposalLane,
    selected: Iterable[ProposalLane],
) -> float:
    """Return the expected new coverage from adding ``lane``."""

    _validate_lane(lane)
    selected = tuple(selected)
    for other in selected:
        _validate_lane(other)
    if any(len(other.success_rates) != len(lane.success_rates) for other in selected):
        raise ValueError("all lanes must cover the same number of positions")
    return fsum(
        lane.success_rates[index] * _residual_probability(selected, index)
        for index in range(len(lane.success_rates))
    )


def greedy_plan(
    lanes: Iterable[ProposalLane],
    budget: float,
) -> tuple[ProposalLane, ...]:
    """Greedily select lanes by residual marginal coverage per unit cost."""

    if budget < 0.0:
        raise ValueError("budget must be non-negative")
    remaining = list(lanes)
    for lane in remaining:
        _validate_lane(lane)
    selected: list[ProposalLane] = []
    spent = 0.0
    while remaining:
        affordable = [lane for lane in remaining if spent + lane.cost <= budget]
        if not affordable:
            break
        best = min(
            affordable,
            key=lambda lane: (
                -marginal_coverage(lane, selected) / lane.cost,
                -marginal_coverage(lane, selected),
                lane.name,
            ),
        )
        selected.append(best)
        remaining.remove(best)
        spent += best.cost
    return tuple(selected)


if __name__ == "__main__":
    lanes = [
        ProposalLane("text", (0.8, 0.2), 1.0),
        ProposalLane("image", (0.2, 0.8), 1.0),
    ]
    assert expected_coverage(greedy_plan(lanes, 2.0)) > 0.0
    print("submodular_allocator selftest: PASS")
