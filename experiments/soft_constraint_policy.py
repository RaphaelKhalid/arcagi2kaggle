"""Recall-preserving mixture policy for uncertain grid constraints.

An inferred size or palette is evidence about a test output, not a theorem.
This module makes the safe decoder contract executable without depending on a
model implementation: retain an unconstrained branch, allocate constrained
branches by posterior mass, and merge duplicate output hypotheses with
log-sum-exp.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ConstraintHypothesis:
    """A decoder branch and its calibrated prior probability."""

    name: str
    prior: float
    constrained: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hypothesis name must be non-empty")
        if not isfinite(float(self.prior)) or self.prior < 0.0:
            raise ValueError("hypothesis prior must be finite and non-negative")


def normalized_priors(
    hypotheses: Sequence[ConstraintHypothesis],
) -> tuple[ConstraintHypothesis, ...]:
    """Normalize non-negative priors without changing branch identity."""

    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    total = sum(float(item.prior) for item in hypotheses)
    if total <= 0.0:
        raise ValueError("hypothesis priors must have positive total mass")
    return tuple(
        ConstraintHypothesis(item.name, item.prior / total, item.constrained)
        for item in hypotheses
    )


def hard_gate_allowed(
    fired: int,
    false_negatives: int,
    *,
    minimum_fired: int = 1,
) -> bool:
    """Return whether an empirical hard gate passes the zero-FN contract.

    This is deliberately a release gate, not a statistical claim that future
    false negatives are impossible.  The caller must audit a disjoint labeled
    fold and record the counts.
    """

    if fired < 0 or false_negatives < 0 or false_negatives > fired:
        raise ValueError("invalid fired/false-negative counts")
    if minimum_fired < 1:
        raise ValueError("minimum_fired must be positive")
    return fired >= minimum_fired and false_negatives == 0


def largest_remainder_allocation(
    hypotheses: Sequence[ConstraintHypothesis],
    total_beams: int,
    *,
    baseline_name: str = "unconstrained",
    baseline_minimum: int = 1,
) -> dict[str, int]:
    """Allocate integer beams by prior while preserving an incumbent branch.

    The largest-remainder rule minimizes rounding error.  The baseline branch
    is guaranteed at least ``baseline_minimum`` beams, so adding uncertain
    constraints cannot remove the old candidate-support path.
    """

    if total_beams < 1 or baseline_minimum < 1:
        raise ValueError("beam counts must be positive")
    normalized = normalized_priors(hypotheses)
    names = [item.name for item in normalized]
    if len(set(names)) != len(names):
        raise ValueError("hypothesis names must be unique")
    if baseline_name not in names:
        raise ValueError("baseline hypothesis is missing")
    if total_beams < baseline_minimum:
        raise ValueError("total_beams cannot satisfy baseline_minimum")

    raw = [total_beams * item.prior for item in normalized]
    allocation = {item.name: int(value) for item, value in zip(normalized, raw)}
    remaining = total_beams - sum(allocation.values())
    order = sorted(
        range(len(normalized)),
        key=lambda index: (raw[index] - int(raw[index]), normalized[index].prior),
        reverse=True,
    )
    for index in order[:remaining]:
        allocation[normalized[index].name] += 1

    if allocation[baseline_name] < baseline_minimum:
        deficit = baseline_minimum - allocation[baseline_name]
        donors = sorted(
            (name for name in names if name != baseline_name),
            key=lambda name: allocation[name],
            reverse=True,
        )
        for donor in donors:
            moved = min(deficit, max(0, allocation[donor]))
            allocation[donor] -= moved
            allocation[baseline_name] += moved
            deficit -= moved
            if deficit == 0:
                break
        if deficit:
            raise ValueError("could not preserve baseline beam minimum")
    return allocation


def logsumexp(values: Iterable[float]) -> float:
    """Stable log-sum-exp for a non-empty finite sequence."""

    values = tuple(float(value) for value in values)
    if not values or any(not isfinite(value) for value in values):
        raise ValueError("values must be a non-empty finite sequence")
    maximum = max(values)
    return maximum + log(sum(exp(value - maximum) for value in values))


def mixture_output_score(
    witnesses: Iterable[tuple[float, float]],
) -> float:
    """Score one deduplicated output from ``(prior, log_likelihood)`` pairs.

    Summing branch mass is the Bayesian model average.  Taking only the best
    branch would overstate evidence when several correlated hypotheses produce
    the same grid.
    """

    terms = []
    for prior, log_likelihood in witnesses:
        prior = float(prior)
        log_likelihood = float(log_likelihood)
        if prior <= 0.0 or not isfinite(prior) or not isfinite(log_likelihood):
            raise ValueError("witness priors must be positive and finite")
        terms.append(log(prior) + log_likelihood)
    return logsumexp(terms)


if __name__ == "__main__":
    branches = normalized_priors(
        (
            ConstraintHypothesis("unconstrained", 0.4, constrained=False),
            ConstraintHypothesis("paranoid-size", 0.6),
        )
    )
    allocation = largest_remainder_allocation(branches, 10)
    assert allocation == {"unconstrained": 4, "paranoid-size": 6}
    assert hard_gate_allowed(109, 0)
    assert not hard_gate_allowed(56, 19)
    assert mixture_output_score(((0.4, -1.0), (0.6, -1.0))) == -1.0
    print("soft_constraint_policy selftest: PASS")
