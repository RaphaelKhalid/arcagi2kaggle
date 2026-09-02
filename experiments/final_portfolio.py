"""Scenario-based selection of two whole final submissions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import fsum
from typing import Iterable, Mapping


def _validate(scores: Mapping[str, tuple[float, ...]]) -> tuple[str, ...]:
    if not scores:
        raise ValueError("at least one submission is required")
    names = tuple(sorted(scores))
    lengths = {len(scores[name]) for name in names}
    if not lengths or 0 in lengths or len(lengths) != 1:
        raise ValueError("every submission needs the same non-empty scenarios")
    if any(not 0.0 <= value <= 1.0
           for values in scores.values() for value in values):
        raise ValueError("scenario scores must lie in [0, 1]")
    return names


@dataclass(frozen=True)
class PortfolioScore:
    submissions: tuple[str, ...]
    mean_best_score: float
    worst_case_best_score: float


def score_portfolio(
    scores: Mapping[str, tuple[float, ...]],
    submissions: Iterable[str],
) -> PortfolioScore:
    """Score a portfolio as best whole-file score per validation scenario."""

    names = _validate(scores)
    pair = tuple(dict.fromkeys(submissions))
    if not 1 <= len(pair) <= 2 or any(name not in scores for name in pair):
        raise ValueError("portfolio must contain one or two known submissions")
    scenario_best = tuple(max(scores[name][i] for name in pair)
                          for i in range(len(scores[names[0]])))
    return PortfolioScore(
        pair,
        fsum(scenario_best) / len(scenario_best),
        min(scenario_best),
    )


def select_portfolio(
    scores: Mapping[str, tuple[float, ...]],
    *,
    require_two: bool = False,
) -> PortfolioScore:
    """Select the pair with maximum mean scenario-best score.

    Scenario scores must come from disjoint labeled folds or bootstrap
    replicates. This function does not assume that Kaggle ORs predictions from
    separate final submissions; it models the conservative ``best whole-file
    score`` interpretation only.
    """

    names = _validate(scores)
    actions = list(combinations(names, 2)) if len(names) > 1 else [(names[0],)]
    if not require_two:
        actions.extend((name,) for name in names)
    if not actions:
        raise ValueError("no portfolio action is available")
    ranked = [score_portfolio(scores, action) for action in actions]
    return max(
        ranked,
        key=lambda item: (
            item.mean_best_score,
            item.worst_case_best_score,
            tuple(reversed(item.submissions)),
        ),
    )


if __name__ == "__main__":
    result = select_portfolio({
        "base": (0.8, 0.8),
        "explore": (0.6, 0.95),
    }, require_two=True)
    assert result.submissions == ("base", "explore")
    print("final_portfolio selftest: PASS", result.mean_best_score)
