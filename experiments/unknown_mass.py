"""Capture--recapture estimate for unseen ARC output classes.

This is a diagnostic for proposal routing, not a calibrated probability
unless the two panels are approximately independent samples from the same
class population.  It is useful precisely because a panel's observed entropy
cannot reveal output classes that no candidate emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable

from experiments.cegis_version_space import freeze_output


@dataclass(frozen=True)
class CaptureRecaptureEstimate:
    """Estimated unseen class count and fraction for two proposal panels."""

    first_count: int
    second_count: int
    overlap_count: int
    union_count: int
    estimated_population: float
    estimated_unseen: float
    unseen_fraction: float
    reliable: bool


def estimate_unseen_fraction(
    first_panel: Iterable[Any],
    second_panel: Iterable[Any],
) -> CaptureRecaptureEstimate:
    """Estimate unseen class fraction with the Chapman correction.

    Repeated syntactic candidates are removed before estimation.  With no
    overlap, the estimator deliberately returns an infinite population and a
    full unknown reserve: the panels provide no evidence that they cover the
    same class population.  ``reliable`` requires at least two recaptures;
    callers should shrink or cap the reserve when it is false.
    """

    first = {freeze_output(value) for value in first_panel}
    second = {freeze_output(value) for value in second_panel}
    n_first, n_second = len(first), len(second)
    overlap = len(first & second)
    union = len(first | second)

    if not first or not second or overlap == 0:
        return CaptureRecaptureEstimate(
            n_first, n_second, overlap, union,
            float("inf"), float("inf"), 1.0, False,
        )

    # Chapman: (n1+1)(n2+1)/(m+1) - 1, less biased than n1*n2/m for
    # small panels.  The unseen count is population minus observed union.
    population = ((n_first + 1) * (n_second + 1) / (overlap + 1)) - 1.0
    unseen = max(0.0, population - union)
    fraction = min(1.0, max(0.0, unseen / population))
    return CaptureRecaptureEstimate(
        n_first, n_second, overlap, union,
        population, unseen, fraction, overlap >= 2 and isfinite(population),
    )


if __name__ == "__main__":
    estimate = estimate_unseen_fraction(["a", "b"], ["b", "c"])
    assert estimate.unseen_fraction > 0.0
    print("unknown_mass selftest: PASS")
