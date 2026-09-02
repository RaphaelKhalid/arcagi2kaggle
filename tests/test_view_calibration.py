from __future__ import annotations

import pytest

from experiments.view_calibration import (
    calibrated_poe_score,
    estimate_view_offsets,
    rank_candidates,
)


def test_offsets_remove_consistent_representation_bias() -> None:
    offsets = estimate_view_offsets({"row": (1.0, 2.0), "column": (4.0, 5.0)})
    assert offsets == {"column": 1.5, "row": -1.5}
    assert calibrated_poe_score(
        {"row": 1.0, "column": 4.0}, offsets
    ) == pytest.approx(calibrated_poe_score(
        {"row": 2.0, "column": 5.0}, offsets
    ))


def test_shrinkage_is_bounded_and_zero_is_raw() -> None:
    values = {"a": (1.0, 3.0), "b": (2.0, 2.0)}
    assert estimate_view_offsets(values, shrinkage=1.0) == {"a": 0.0, "b": 0.0}
    with pytest.raises(ValueError):
        estimate_view_offsets(values, shrinkage=1.1)


def test_missing_view_penalty_is_explicit() -> None:
    offsets = {"a": 0.0, "b": 0.0}
    unpenalized = calibrated_poe_score({"a": 1.0}, offsets)
    penalized = calibrated_poe_score(
        {"a": 1.0}, offsets, missing_view_penalty=2.0
    )
    assert unpenalized == pytest.approx(-1.0)
    assert penalized == pytest.approx(-2.0)


def test_rank_is_deterministic_after_calibration() -> None:
    offsets = {"row": 0.0, "column": 3.0}
    ranked = rank_candidates(
        {
            "candidate-b": {"row": 2.0, "column": 5.0},
            "candidate-a": {"row": 1.0, "column": 4.0},
        },
        offsets,
    )
    assert ranked == ("candidate-a", "candidate-b")
