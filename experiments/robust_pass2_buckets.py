"""Distributionally robust pass@2 selection over visible task buckets.

The official scorer accepts two exact output attempts per test position.  This
module selects those classes against bucket-conditional uncertainty: a source
class mass is reduced by a declared total-variation radius independently in
each visible structural bucket, and the resulting lower masses are mixed by
the unlabeled target bucket distribution.
"""

from __future__ import annotations

from itertools import combinations
from math import fsum
from typing import Iterable, Mapping


def robust_pair_mass(
    bucket_class_mass: Mapping[str, Mapping[str, float]],
    target_mass: Mapping[str, float],
    pair: Iterable[str],
    *,
    shift_radius: Mapping[str, float] | None = None,
) -> float:
    """Return a conservative target lower bound for one distinct class pair."""

    if not target_mass:
        raise ValueError("target_mass must be non-empty")
    if any(mass < 0.0 for mass in target_mass.values()):
        raise ValueError("target masses must be non-negative")
    if abs(fsum(target_mass.values()) - 1.0) > 1e-9:
        raise ValueError("target masses must sum to one")
    pair = tuple(dict.fromkeys(pair))
    if not 1 <= len(pair) <= 2:
        raise ValueError("pair must contain one or two distinct classes")
    radii = shift_radius or {}
    if any(not 0.0 <= radius <= 1.0 for radius in radii.values()):
        raise ValueError("shift radii must lie in [0, 1]")

    total = 0.0
    for bucket, mass in target_mass.items():
        distribution = bucket_class_mass.get(bucket)
        if distribution is None:
            lower = 0.0
        else:
            if any(value < 0.0 for value in distribution.values()):
                raise ValueError("class masses must be non-negative")
            if abs(fsum(distribution.values()) - 1.0) > 1e-9:
                raise ValueError("each source bucket must be normalized")
            event_mass = fsum(distribution.get(output, 0.0) for output in pair)
            lower = max(0.0, event_mass - radii.get(bucket, 0.0))
        total += mass * lower
    return min(1.0, max(0.0, total))


def select_robust_pass2(
    bucket_class_mass: Mapping[str, Mapping[str, float]],
    target_mass: Mapping[str, float],
    *,
    shift_radius: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    """Choose up to two observed classes by robust target lower mass."""

    outputs = sorted({
        output
        for distribution in bucket_class_mass.values()
        for output in distribution
    })
    if not outputs:
        return ()
    actions = tuple(combinations(outputs, min(2, len(outputs))))
    return max(
        actions,
        key=lambda pair: (
            robust_pair_mass(
                bucket_class_mass, target_mass, pair,
                shift_radius=shift_radius,
            ),
            tuple(reversed(pair)),
        ),
    )


if __name__ == "__main__":
    selected = select_robust_pass2(
        {"compact": {"a": 0.8, "b": 0.2}}, {"compact": 1.0}
    )
    assert selected == ("a", "b")
    print("robust_pass2_buckets selftest: PASS")
