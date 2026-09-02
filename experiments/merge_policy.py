"""Posterior-gated replacement for a two-slot ARC submission pair.

Adding a candidate to a pass@2 position is not monotone: one of the existing
two guesses must be displaced.  This helper makes promotion depend on a
calibrated output-class mass rather than on demo verification alone.  It is an
offline policy seam; the notebook must supply leakage-safe masses before using
it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import fsum


@dataclass(frozen=True)
class MergeDecision:
    attempts: tuple[str, ...]
    baseline_mass: float
    proposed_mass: float
    promoted: bool


def gated_pair_merge(
    baseline_pair: Iterable[str],
    candidate_masses: Mapping[str, float],
    *,
    min_gain: float = 0.0,
) -> MergeDecision:
    """Replace the baseline pair only when posterior mass strictly improves.

    Missing baseline masses are treated as zero, which makes the function safe
    for a staged rollout where only newly generated classes have calibration
    records. Ties preserve the baseline pair, avoiding gratuitous slot churn.
    """

    if min_gain < 0.0:
        raise ValueError("min_gain must be non-negative")
    baseline = tuple(baseline_pair)
    if len(baseline) > 2 or len(set(baseline)) != len(baseline):
        raise ValueError("baseline_pair must contain at most two distinct classes")
    masses = {str(key): float(value) for key, value in candidate_masses.items()}
    if any(value < 0.0 for value in masses.values()):
        raise ValueError("candidate masses must be non-negative")
    options = set(masses) | set(baseline)
    ranked = tuple(sorted(options, key=lambda key: (-masses.get(key, 0.0), key)))[:2]
    baseline_mass = fsum(masses.get(key, 0.0) for key in baseline)
    proposed_mass = fsum(masses.get(key, 0.0) for key in ranked)
    promoted = proposed_mass > baseline_mass + min_gain
    return MergeDecision(
        attempts=ranked if promoted else baseline,
        baseline_mass=baseline_mass,
        proposed_mass=proposed_mass,
        promoted=promoted,
    )


if __name__ == "__main__":
    decision = gated_pair_merge(("a", "b"), {"a": 0.45, "b": 0.40, "g": 0.10})
    assert decision.attempts == ("a", "b")
    print("merge_policy selftest: PASS")
