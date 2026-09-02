from __future__ import annotations

import pytest

from experiments.behavioral_partition import (
    behavioral_partition,
    greedy_probe_selection,
    separated_pairs,
)


def _probes() -> dict[str, dict[str, int]]:
    return {
        "weak": {"a": 0, "b": 0, "c": 1},
        "strong": {"a": 0, "b": 1, "c": 1},
    }


def test_behavioral_partition_quotients_exact_probe_signatures() -> None:
    classes = behavioral_partition(_probes())
    assert len(classes) == 3
    assert all(len(item.members) == 1 for item in classes)


def test_separated_pairs_are_monotone() -> None:
    probes = _probes()
    one = separated_pairs(probes, ["weak"])
    two = separated_pairs(probes, ["weak", "strong"])
    assert one < two
    assert one <= two


def test_greedy_selection_uses_pair_separation_gain() -> None:
    assert greedy_probe_selection(_probes(), 1) == ("strong",)
    assert greedy_probe_selection(_probes(), 2) == ("strong", "weak")


def test_invalid_probe_panels_are_rejected() -> None:
    with pytest.raises(ValueError):
        behavioral_partition({"a": {"p": 1}, "b": {"q": 2}})
    with pytest.raises(ValueError):
        greedy_probe_selection(_probes(), -1)

