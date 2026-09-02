from __future__ import annotations

import pytest

from experiments.length_frontier import UnionLengthFrontier


def test_union_budget_preserves_absolute_branch() -> None:
    frontier = UnionLengthFrontier(1.609, 0.02, 900)
    assert frontier.union_budget == pytest.approx(18.0)
    assert frontier.accepts_complete(1.5)


def test_normalized_branch_admits_longer_completion() -> None:
    frontier = UnionLengthFrontier(1.609, 0.02, 900)
    assert frontier.accepts_complete(10.0)
    assert frontier.mean_nll(10.0) == pytest.approx(10.0 / 900)


def test_partial_admission_is_optimistic_and_shape_agnostic() -> None:
    frontier = UnionLengthFrontier(1.609, 0.02, 900)
    assert frontier.accepts_partial(17.9, 100)
    assert not frontier.accepts_partial(18.1, 100)


def test_invalid_frontier_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        UnionLengthFrontier(0.0, 0.02, 900)
    with pytest.raises(ValueError):
        UnionLengthFrontier(1.609, 0.02, 0)
    with pytest.raises(ValueError):
        UnionLengthFrontier(1.609, 0.02, 900).accepts_partial(1.0, 901)
