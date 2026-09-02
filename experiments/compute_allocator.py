"""Anytime candidate-budget allocation from calibrated family hit rates.

For a family with independent effective hit rate ``q`` per additional
candidate, after ``n`` candidates the probability of still missing the target
is ``(1-q)**n``.  The next candidate's expected discovery gain is therefore
``q * (1-q)**n``.  Dividing by candidate cost gives a principled greedy
priority.  In practice ``q`` must be estimated from held-out folds and should
already be an effective rate that discounts within-family correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, lgamma


@dataclass(frozen=True)
class FamilyBudget:
    name: str
    hit_rate: float
    cost_seconds: float
    max_candidates: int = 10**9

    def __post_init__(self) -> None:
        if not 0.0 <= self.hit_rate <= 1.0:
            raise ValueError("hit_rate must be in [0, 1]")
        if self.cost_seconds <= 0.0:
            raise ValueError("cost_seconds must be positive")
        if self.max_candidates < 0:
            raise ValueError("max_candidates must be non-negative")


def marginal_discovery_gain(hit_rate: float, count: int) -> float:
    """Expected new-hit probability of candidate ``count + 1``."""

    if not 0.0 <= hit_rate <= 1.0 or count < 0:
        raise ValueError("invalid hit rate or count")
    return hit_rate * (1.0 - hit_rate) ** count


def greedy_plan(
    families: tuple[FamilyBudget, ...], *, budget_seconds: float
) -> tuple[str, ...]:
    """Return a cost-aware sequence of family names under a hard budget."""

    if budget_seconds < 0.0:
        raise ValueError("budget_seconds must be non-negative")
    counts = {family.name: 0 for family in families}
    elapsed = 0.0
    plan: list[str] = []
    while True:
        eligible = [family for family in families
                    if counts[family.name] < family.max_candidates
                    and elapsed + family.cost_seconds <= budget_seconds]
        if not eligible:
            return tuple(plan)
        family = max(
            eligible,
            key=lambda item: (
                marginal_discovery_gain(item.hit_rate, counts[item.name])
                / item.cost_seconds,
                marginal_discovery_gain(item.hit_rate, counts[item.name]),
                item.name,
            ),
        )
        plan.append(family.name)
        counts[family.name] += 1
        elapsed += family.cost_seconds


@dataclass(frozen=True)
class PosteriorFamily:
    """A family arm with a Beta posterior for its effective hit rate."""

    name: str
    successes: int
    failures: int
    cost_seconds: float
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    max_candidates: int = 10**9

    def __post_init__(self) -> None:
        if self.successes < 0 or self.failures < 0:
            raise ValueError("observations must be non-negative")
        if self.prior_alpha <= 0.0 or self.prior_beta <= 0.0:
            raise ValueError("Beta prior parameters must be positive")
        if self.cost_seconds <= 0.0 or self.max_candidates < 0:
            raise ValueError("invalid cost or candidate bound")

    @property
    def alpha(self) -> float:
        return self.prior_alpha + self.successes

    @property
    def beta(self) -> float:
        return self.prior_beta + self.failures


def beta_marginal_gain(alpha: float, beta: float, count: int) -> float:
    """E[q(1-q)^count] for q ~ Beta(alpha, beta)."""

    if alpha <= 0.0 or beta <= 0.0 or count < 0:
        raise ValueError("invalid Beta parameters or count")
    return exp(
        lgamma(alpha + 1.0) + lgamma(beta + count)
        - lgamma(alpha + beta + count + 1.0)
        - lgamma(alpha) - lgamma(beta) + lgamma(alpha + beta)
    )


def greedy_posterior_plan(
    families: tuple[PosteriorFamily, ...], *, budget_seconds: float
) -> tuple[str, ...]:
    """Budget candidates by posterior expected discovery gain per second."""

    if budget_seconds < 0.0:
        raise ValueError("budget_seconds must be non-negative")
    counts = {family.name: 0 for family in families}
    elapsed = 0.0
    plan: list[str] = []
    while True:
        eligible = [family for family in families
                    if counts[family.name] < family.max_candidates
                    and elapsed + family.cost_seconds <= budget_seconds]
        if not eligible:
            return tuple(plan)
        family = max(
            eligible,
            key=lambda item: (
                beta_marginal_gain(item.alpha, item.beta, counts[item.name])
                / item.cost_seconds,
                item.name,
            ),
        )
        plan.append(family.name)
        counts[family.name] += 1
        elapsed += family.cost_seconds


if __name__ == "__main__":
    plan = greedy_plan((FamilyBudget("cheap", 0.2, 1),
                        FamilyBudget("slow", 0.6, 5)), budget_seconds=3)
    assert plan == ("cheap", "cheap", "cheap")
    assert abs(beta_marginal_gain(1.0, 1.0, 0) - 0.5) < 1e-12
    print("compute_allocator selftest: PASS", plan)
