from __future__ import annotations

import pytest

from experiments.output_cost_calibration import (
    demo_count_bucket,
    fit_output_cost_calibration,
    input_area_bucket,
)


def test_bucket_boundaries_are_deterministic() -> None:
    assert input_area_bucket(25) == 0
    assert input_area_bucket(26) == 1
    assert input_area_bucket(225) == 2
    assert demo_count_bucket(2) == 0
    assert demo_count_bucket(5) == 2


def test_large_buckets_get_their_own_mean() -> None:
    calibration = fit_output_cost_calibration(
        [(10, 2, 5.0)] * 20 + [(200, 4, 20.0)] * 20
    )
    assert calibration.estimate(10, 2) == pytest.approx(5.0)
    assert calibration.estimate(200, 4) == pytest.approx(20.0)


def test_sparse_bucket_falls_back_and_known_shape_wins() -> None:
    calibration = fit_output_cost_calibration([(10, 2, 5.0)], min_bucket_samples=20)
    assert calibration.estimate(10, 2) == pytest.approx(5.0)
    assert calibration.estimate(10, 2, known_shape_cap=7.0) == pytest.approx(7.0)


def test_invalid_calibration_is_rejected() -> None:
    with pytest.raises(ValueError):
        fit_output_cost_calibration(())
    with pytest.raises(ValueError):
        fit_output_cost_calibration([(10, 2, 0.0)])
    with pytest.raises(ValueError):
        input_area_bucket(0)
