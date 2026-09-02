from __future__ import annotations

import pytest

from experiments.soft_constraint_policy import (
    ConstraintHypothesis,
    hard_gate_allowed,
    largest_remainder_allocation,
    mixture_output_score,
    normalized_priors,
)


def test_priors_normalize_without_changing_names() -> None:
    branches = normalized_priors(
        (ConstraintHypothesis("unconstrained", 1), ConstraintHypothesis("shape", 3))
    )
    assert [item.name for item in branches] == ["unconstrained", "shape"]
    assert [item.prior for item in branches] == pytest.approx([0.25, 0.75])


def test_allocation_preserves_baseline_branch() -> None:
    allocation = largest_remainder_allocation(
        (ConstraintHypothesis("unconstrained", 0.01), ConstraintHypothesis("shape", 0.99)),
        8,
    )
    assert allocation["unconstrained"] >= 1
    assert sum(allocation.values()) == 8


def test_hard_gate_requires_zero_false_negatives() -> None:
    assert hard_gate_allowed(109, 0, minimum_fired=100)
    assert not hard_gate_allowed(109, 1, minimum_fired=100)


def test_mixture_score_sums_duplicate_branch_mass() -> None:
    one = mixture_output_score(((1.0, -2.0),))
    two = mixture_output_score(((0.25, -2.0), (0.75, -2.0)))
    assert two == pytest.approx(one)


def test_invalid_branch_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        normalized_priors(())
    with pytest.raises(ValueError):
        largest_remainder_allocation(
            (ConstraintHypothesis("shape", 1),), 4
        )
    with pytest.raises(ValueError):
        mixture_output_score(((0.0, -1.0),))
