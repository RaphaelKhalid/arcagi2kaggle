from __future__ import annotations

import math

import pytest

from experiments.conformal_threshold import (
    conformal_union_budget,
    upper_conformal_quantile,
)


def test_upper_quantile_uses_finite_sample_rank() -> None:
    assert upper_conformal_quantile((0.01, 0.02, 0.03), miscoverage=0.25) == pytest.approx(0.03)


def test_small_calibration_fold_abstains() -> None:
    assert math.isinf(upper_conformal_quantile((0.01, 0.02), miscoverage=0.1))


def test_union_budget_preserves_absolute_cutoff() -> None:
    assert conformal_union_budget(1.609, 900, (0.01, 0.02, 0.03), miscoverage=0.25) == pytest.approx(27.0)
    assert conformal_union_budget(2.0, 1, (0.01, 0.02, 0.03), miscoverage=0.25) == pytest.approx(2.0)


def test_invalid_scores_are_rejected() -> None:
    with pytest.raises(ValueError):
        upper_conformal_quantile((), miscoverage=0.1)
    with pytest.raises(ValueError):
        upper_conformal_quantile((float("nan"),), miscoverage=0.1)
    with pytest.raises(ValueError):
        conformal_union_budget(0.0, 10, (0.1,))
