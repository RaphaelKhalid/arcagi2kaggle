"""Correlation-aware allocation of neural program samples.

The usual ``n``-sample intuition counts programs as independent evidence.  In
ARC induction, samples sharing a checkpoint, prompt, and decoding batch are
often correlated.  This module exposes the design equation used to choose a
number of independently seeded/prompted lineages before spending GPU time.

For lineage sample counts ``n_i`` and an exchangeable within-lineage
correlation ``rho``, the effective sample size proxy is

    ESS = (sum_i n_i)^2 / sum_i (n_i + rho*n_i*(n_i-1)).

For fixed total samples and fixed non-empty lineage count, balanced counts
maximize this proxy because they minimize the sum of ``n_i**2``.  The proxy is
not a correctness estimate; it is a planning diagnostic that must be checked
against held-out output-class discovery and selector recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum


def lineage_effective_sample_size(
    sample_counts: tuple[int, ...], *, rho: float = 0.0
) -> float:
    """Return the exchangeable-correlation ESS proxy for lineage counts."""

    if not sample_counts or any(count <= 0 for count in sample_counts):
        raise ValueError("sample_counts must contain only positive integers")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    total = sum(sample_counts)
    variance = fsum(
        count + rho * count * (count - 1) for count in sample_counts
    )
    return total * total / variance


@dataclass(frozen=True)
class LineagePlan:
    """A deterministic allocation of a fixed sample budget across lineages."""

    sample_counts: tuple[int, ...]
    rho: float

    @property
    def total_samples(self) -> int:
        return sum(self.sample_counts)

    @property
    def effective_sample_size(self) -> float:
        return lineage_effective_sample_size(self.sample_counts, rho=self.rho)


def balanced_lineage_plan(
    total_samples: int,
    lineage_count: int,
    *,
    rho: float = 0.0,
) -> LineagePlan:
    """Split samples as evenly as possible across a fixed lineage count.

    The first ``remainder`` lineages receive one extra sample.  This is the
    unique allocation shape (up to permutation) that maximizes ESS for the
    exchangeable-correlation proxy when the number of lineages is fixed.
    """

    if total_samples < 1 or lineage_count < 1:
        raise ValueError("total_samples and lineage_count must be positive")
    if lineage_count > total_samples:
        raise ValueError("lineage_count cannot exceed total_samples")
    # Validate rho through the public formula before constructing the plan.
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    quotient, remainder = divmod(total_samples, lineage_count)
    counts = tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(lineage_count)
    )
    return LineagePlan(counts, rho)


def max_lineage_ess(
    total_samples: int, *, rho: float = 0.0
) -> LineagePlan:
    """Return the ESS-maximizing plan when each sample may be its own lineage."""

    return balanced_lineage_plan(total_samples, total_samples, rho=rho)


if __name__ == "__main__":
    plan = balanced_lineage_plan(8, 4, rho=0.6)
    assert plan.sample_counts == (2, 2, 2, 2)
    assert plan.effective_sample_size > lineage_effective_sample_size((5, 1, 1, 1), rho=0.6)
    print("lineage_sampling selftest: PASS", plan)
