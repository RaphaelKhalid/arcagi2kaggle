from __future__ import annotations

import pytest

from experiments.score_delta_budget import (
    decompose_score_delta,
    minimum_recovery_for_gain,
    score_from_coverage,
)


def test_score_is_coverage_times_conditional_recovery() -> None:
    assert score_from_coverage(0.8, 0.75) == pytest.approx(0.6)


def test_delta_components_recombine_exactly() -> None:
    delta = decompose_score_delta(0.8, 0.5, 0.9, 0.6)
    assert delta.total == pytest.approx(
        delta.coverage_component
        + delta.recovery_component
        + delta.interaction_component
    )


def test_fixed_coverage_target_is_invertible() -> None:
    required = minimum_recovery_for_gain(0.9, 0.8, 0.5, 0.04)
    assert required == pytest.approx((0.8 * 0.5 + 0.04) / 0.9)


def test_invalid_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError):
        score_from_coverage(1.1, 0.5)
    with pytest.raises(ValueError):
        minimum_recovery_for_gain(0.9, 0.8, 0.5, -0.1)

