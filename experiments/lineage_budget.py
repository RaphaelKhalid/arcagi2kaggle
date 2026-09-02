"""Choose the number of independent proposal lineages under setup cost."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum


@dataclass(frozen=True)
class LineageBudgetPlan:
    lineage_count: int
    sample_counts: tuple[int, ...]
    total_samples: int
    effective_sample_size: float


def _ess(sample_counts: tuple[int, ...], rho: float) -> float:
    total = sum(sample_counts)
    variance = fsum(
        count + rho * count * (count - 1) for count in sample_counts
    )
    return total * total / variance


def choose_lineage_budget(
    budget_seconds: float,
    sample_seconds: float,
    setup_seconds: float,
    *,
    rho: float = 0.0,
    max_lineages: int = 64,
) -> LineageBudgetPlan:
    """Return the ESS-maximizing balanced plan under setup/sample costs.

    ``rho`` is an exchangeable within-lineage correlation proxy. This is a
    planning statistic, not a correctness probability. At least one sample is
    reserved per selected lineage.
    """

    if budget_seconds <= 0.0 or sample_seconds <= 0.0 or setup_seconds < 0.0:
        raise ValueError("invalid budget or cost")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if max_lineages < 1:
        raise ValueError("max_lineages must be positive")

    plans: list[LineageBudgetPlan] = []
    for lineages in range(1, max_lineages + 1):
        remaining = budget_seconds - lineages * setup_seconds
        total_samples = int(remaining // sample_seconds)
        if total_samples < lineages:
            continue
        quotient, remainder = divmod(total_samples, lineages)
        counts = tuple(
            quotient + (1 if index < remainder else 0)
            for index in range(lineages)
        )
        plans.append(LineageBudgetPlan(
            lineages, counts, total_samples, _ess(counts, rho)
        ))
    if not plans:
        raise ValueError("budget cannot fund one sample per lineage")
    return max(plans, key=lambda plan: (
        plan.effective_sample_size,
        plan.total_samples,
        -plan.lineage_count,
    ))


if __name__ == "__main__":
    plan = choose_lineage_budget(10.0, 1.0, 0.5, rho=0.8)
    assert plan.lineage_count > 1
    print("lineage_budget selftest: PASS", plan)
