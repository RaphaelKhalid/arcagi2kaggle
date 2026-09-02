"""Worst-case pass@2 selection under uncertain family priors.

Family hit rates and priors are estimated from sparse folds, so a point-mixture
selector can be overconfident.  For a proposed output pair, expected coverage
is linear in family weights.  This module optimizes that linear function over a
box-constrained simplex exactly, then chooses the pair with the best worst-case
mass.  It is a conservative calibration ablation, not a replacement for
held-out posterior estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import fsum
from typing import Iterable, Mapping

from experiments.pass2_selector import (
    Candidate,
    TaskCandidate,
    _family_conditional_mass,
    _family_vector_mass,
)


Interval = tuple[float, float]


@dataclass(frozen=True)
class RobustClassScore:
    output_hash: str
    lower_mass: float
    upper_mass: float


@dataclass(frozen=True)
class RobustSelection:
    outputs: tuple[str, ...]
    worst_case_mass: float


@dataclass(frozen=True)
class RobustTaskSelection:
    """Independent output pairs optimized against one shared family mixture."""

    pairs: tuple[tuple[str, ...], ...]
    worst_case_mass: float


def _validate_intervals(
    family_names: Iterable[str],
    intervals: Mapping[str, Interval],
) -> tuple[str, ...]:
    names = tuple(sorted(set(family_names)))
    if set(intervals) != set(names):
        raise ValueError("an interval is required for every active family")
    for family in names:
        low, high = intervals[family]
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("family intervals must satisfy 0 <= low <= high <= 1")
    if fsum(intervals[family][0] for family in names) > 1.0 + 1e-12:
        raise ValueError("family lower bounds exceed one")
    if fsum(intervals[family][1] for family in names) < 1.0 - 1e-12:
        raise ValueError("family upper bounds cannot reach one")
    return names


def _extreme_mass(
    coefficients: Mapping[str, float],
    intervals: Mapping[str, Interval],
    *,
    maximize: bool,
) -> float:
    """Optimize a linear family mixture over interval-constrained weights."""

    names = tuple(sorted(coefficients))
    _validate_intervals(names, intervals)
    weights = {family: intervals[family][0] for family in names}
    remaining = 1.0 - fsum(weights.values())
    order = sorted(
        names,
        key=lambda family: coefficients[family],
        reverse=maximize,
    )
    for family in order:
        capacity = intervals[family][1] - intervals[family][0]
        amount = min(remaining, capacity)
        weights[family] += amount
        remaining -= amount
    if remaining > 1e-10:
        raise ValueError("intervals do not define a feasible simplex")
    return fsum(weights[family] * coefficients[family] for family in names)


def _conditional_distributions(
    candidates: Iterable[Candidate],
    *,
    alpha: float,
    collapse_correlated: bool,
) -> tuple[tuple[str, ...], dict[str, dict[str, float]]]:
    by_family: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        if candidate.hard_valid:
            by_family.setdefault(candidate.family, []).append(candidate)
    if not by_family:
        return (), {}
    distributions = {
        family: _family_conditional_mass(
            family_candidates, alpha, collapse_correlated
        )
        for family, family_candidates in by_family.items()
    }
    return tuple(sorted(by_family)), distributions


def score_robust_classes(
    candidates: Iterable[Candidate],
    *,
    family_prior_intervals: Mapping[str, Interval],
    alpha: float = 0.25,
    collapse_correlated: bool = False,
) -> tuple[RobustClassScore, ...]:
    """Return exact lower/upper mass for every observed output class."""

    families, distributions = _conditional_distributions(
        candidates, alpha=alpha, collapse_correlated=collapse_correlated
    )
    if not families:
        return ()
    _validate_intervals(families, family_prior_intervals)
    outputs = sorted({
        output
        for distribution in distributions.values()
        for output in distribution
    })
    result = []
    for output in outputs:
        coefficients = {
            family: distributions[family].get(output, 0.0)
            for family in families
        }
        result.append(RobustClassScore(
            output,
            _extreme_mass(coefficients, family_prior_intervals, maximize=False),
            _extreme_mass(coefficients, family_prior_intervals, maximize=True),
        ))
    return tuple(sorted(result, key=lambda item: (
        -item.lower_mass, -item.upper_mass, item.output_hash
    )))


def robust_select_pass2(
    candidates: Iterable[Candidate],
    *,
    family_prior_intervals: Mapping[str, Interval],
    alpha: float = 0.25,
    collapse_correlated: bool = False,
) -> RobustSelection:
    """Choose up to two classes maximizing worst-case total family mass."""

    families, distributions = _conditional_distributions(
        candidates, alpha=alpha, collapse_correlated=collapse_correlated
    )
    if not families:
        return RobustSelection((), 0.0)
    _validate_intervals(families, family_prior_intervals)
    outputs = sorted({
        output
        for distribution in distributions.values()
        for output in distribution
    })
    if len(outputs) <= 2:
        selected = tuple(outputs)
        coefficients = {
            family: sum(distributions[family].get(output, 0.0) for output in selected)
            for family in families
        }
        return RobustSelection(
            selected,
            _extreme_mass(coefficients, family_prior_intervals, maximize=False),
        )

    best: tuple[float, float, tuple[str, str]] | None = None
    for pair in combinations(outputs, 2):
        coefficients = {
            family: sum(distributions[family].get(output, 0.0) for output in pair)
            for family in families
        }
        lower = _extreme_mass(coefficients, family_prior_intervals, maximize=False)
        upper = _extreme_mass(coefficients, family_prior_intervals, maximize=True)
        candidate = (lower, upper, pair)
        if best is None or (lower, upper, tuple(reversed(pair))) > (
            best[0], best[1], tuple(reversed(best[2]))
        ):
            best = candidate
    assert best is not None
    return RobustSelection(best[2], best[0])


def robust_select_task_output_pairs(
    candidates: Iterable[TaskCandidate],
    *,
    family_prior_intervals: Mapping[str, Interval],
    alpha: float = 0.25,
    collapse_correlated: bool = False,
    max_joint_actions: int = 100_000,
) -> RobustTaskSelection:
    """Optimize the additive task objective under one shared prior interval.

    Complete output vectors induce a joint family-conditional distribution.
    For a chosen pair at each position, family ``f`` contributes the sum of
    its per-position pair probabilities.  The shared-prior adversary then
    minimizes that sum once, rather than independently at every position.
    Exhaustive enumeration is exact while the Cartesian action count is below
    ``max_joint_actions``; exceeding the cap fails loudly so callers can
    pre-trim classes with a documented approximation.
    """

    candidates = tuple(candidate for candidate in candidates)
    if not candidates:
        return RobustTaskSelection((), 0.0)
    lengths = {len(candidate.output_vector) for candidate in candidates}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("all task candidates must have the same non-empty vector")

    by_family: dict[str, list[TaskCandidate]] = {}
    for candidate in candidates:
        by_family.setdefault(candidate.family, []).append(candidate)
    families = tuple(sorted(by_family))
    _validate_intervals(families, family_prior_intervals)
    distributions = {
        family: _family_vector_mass(
            family_candidates, alpha, collapse_correlated
        )
        for family, family_candidates in by_family.items()
    }
    n_positions = next(iter(lengths))
    position_classes: list[tuple[str, ...]] = []
    for position in range(n_positions):
        classes = sorted({
            vector[position]
            for distribution in distributions.values()
            for vector in distribution
        })
        if not classes:
            return RobustTaskSelection((), 0.0)
        position_classes.append(tuple(classes))

    action_lists = [
        tuple(combinations(classes, min(2, len(classes))))
        for classes in position_classes
    ]
    joint_count = 1
    for actions in action_lists:
        joint_count *= len(actions)
    if max_joint_actions < 1 or joint_count > max_joint_actions:
        raise ValueError("joint robust action space exceeds configured cap")

    best: tuple[float, float, tuple[tuple[str, ...], ...]] | None = None
    for action_tuple in product(*action_lists):
        coefficients = {
            family: sum(
                sum(
                    distribution.get(vector, 0.0)
                    for vector in distribution
                    if vector[position] in action
                )
                for position, action in enumerate(action_tuple)
            )
            for family, distribution in distributions.items()
        }
        lower = _extreme_mass(coefficients, family_prior_intervals, maximize=False)
        upper = _extreme_mass(coefficients, family_prior_intervals, maximize=True)
        candidate = (lower, upper, tuple(action_tuple))
        if best is None or (lower, upper, tuple(action_tuple)) > best:
            best = candidate
    assert best is not None
    return RobustTaskSelection(best[2], best[0])


if __name__ == "__main__":
    result = robust_select_pass2(
        [Candidate("a", "fast"), Candidate("b", "slow")],
        family_prior_intervals={"fast": (0.4, 0.6), "slow": (0.4, 0.6)},
    )
    assert result.outputs == ("a", "b") and result.worst_case_mass > 0.99
    print("robust_selector selftest: PASS")
