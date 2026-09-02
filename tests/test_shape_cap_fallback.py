from __future__ import annotations

import pytest

from experiments.shape_cap_fallback import primary_or_fallback, token_cap


def test_primary_prediction_is_never_replaced() -> None:
    assert primary_or_fallback((3, 4), (9, 9)) == (3, 4)


def test_fallback_is_used_only_on_primary_abstention() -> None:
    assert primary_or_fallback(None, (9, 9)) == (9, 9)
    assert primary_or_fallback(None, None) is None


def test_token_cap_matches_grid_serialization() -> None:
    assert token_cap((2, 3)) == 2 * 3 + 2 + 2


def test_invalid_sizes_and_slack_are_rejected() -> None:
    with pytest.raises(ValueError):
        primary_or_fallback((0, 3), None)
    with pytest.raises(ValueError):
        token_cap((3, 4), -1)
