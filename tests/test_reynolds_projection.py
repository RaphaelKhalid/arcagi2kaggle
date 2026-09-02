from __future__ import annotations

import pytest

from experiments.reynolds_projection import orbit_residual, reynolds_projection


def test_inverse_alignment_recovers_same_equivariant_prediction() -> None:
    vectors = {"identity": (1.0, 3.0), "swap": (3.0, 1.0)}
    permutations = {"identity": (0, 1), "swap": (1, 0)}
    assert reynolds_projection(vectors, permutations) == (1.0, 3.0)
    assert orbit_residual(vectors, permutations) == pytest.approx(0.0)


def test_projection_averages_non_equivariant_views() -> None:
    vectors = {"identity": (0.0, 4.0), "swap": (0.0, 4.0)}
    permutations = {"identity": (0, 1), "swap": (1, 0)}
    assert reynolds_projection(vectors, permutations) == (2.0, 2.0)
    assert orbit_residual(vectors, permutations) == pytest.approx(2.0)


def test_projection_is_idempotent_for_already_aligned_values() -> None:
    vectors = {"a": (2.0, 2.0), "b": (2.0, 2.0)}
    permutations = {"a": (0, 1), "b": (1, 0)}
    projected = reynolds_projection(vectors, permutations)
    assert projected == reynolds_projection(
        {"a": projected, "b": projected}, permutations
    )


def test_invalid_alignment_is_rejected() -> None:
    with pytest.raises(ValueError):
        reynolds_projection({"a": (1.0, 2.0)}, {"a": (0, 0)})

