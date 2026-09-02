from __future__ import annotations

import pytest

from experiments.final_portfolio import score_portfolio, select_portfolio


def _scores() -> dict[str, tuple[float, ...]]:
    return {
        "base": (0.8, 0.8),
        "explore": (0.6, 0.95),
        "duplicate": (0.8, 0.8),
    }


def test_portfolio_uses_best_whole_file_per_scenario() -> None:
    result = score_portfolio(_scores(), ("base", "explore"))
    assert result.mean_best_score == pytest.approx((0.8 + 0.95) / 2)
    assert result.worst_case_best_score == pytest.approx(0.8)


def test_complementary_pair_beats_each_singleton() -> None:
    result = select_portfolio(_scores(), require_two=True)
    assert result.submissions == ("base", "explore")


def test_single_submission_is_allowed_when_not_required() -> None:
    result = select_portfolio({"base": (0.8, 0.8)}, require_two=False)
    assert result.submissions == ("base",)


def test_invalid_scenarios_are_rejected() -> None:
    with pytest.raises(ValueError):
        select_portfolio({"a": (0.8,), "b": (0.7, 0.6)})
    with pytest.raises(ValueError):
        score_portfolio(_scores(), ("unknown",))

