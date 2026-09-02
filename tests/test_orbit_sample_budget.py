from __future__ import annotations

import pytest

from experiments.orbit_sample_budget import (
    projection_error_bound,
    required_sample_count,
)


def test_required_count_meets_bound_and_is_minimal() -> None:
    count = required_sample_count(100, 0.1, 0.05)
    assert projection_error_bound(count, 100, 0.05) <= 0.1
    assert projection_error_bound(count - 1, 100, 0.05) > 0.1


def test_more_coordinates_or_confidence_costs_samples() -> None:
    base = required_sample_count(10, 0.1, 0.05)
    assert required_sample_count(100, 0.1, 0.05) > base
    assert required_sample_count(10, 0.1, 0.01) > base


def test_bound_decreases_with_samples() -> None:
    assert projection_error_bound(100, 10, 0.05) < projection_error_bound(
        10, 10, 0.05
    )


def test_invalid_budget_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        required_sample_count(0, 0.1, 0.05)
    with pytest.raises(ValueError):
        required_sample_count(10, 0.0, 0.05)
    with pytest.raises(ValueError):
        projection_error_bound(10, 10, 1.0)

