"""Label-free ablation of 4- versus 8-connected ARC scene parsing.

Connectivity is a latent representation choice, not a property that should be
silently fixed by the object executor.  This module keeps the two parsers
separate and reports only structural changes; it never reads output solutions
for hidden inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:
    from experiments.object_deltas import (
        Connectivity, extract_objects,
    )
except ModuleNotFoundError:  # direct ``python experiments/connectivity_ablation.py``
    from object_deltas import Connectivity, extract_objects


@dataclass(frozen=True)
class ConnectivityProfile:
    grids: int
    changed_grids: int
    four_objects: int
    eight_objects: int
    diagonal_merges: int


def profile_grids(grids: list[Any] | tuple[Any, ...]) -> ConnectivityProfile:
    """Summarize parser disagreement without consulting target labels."""

    changed = 0
    four_objects = 0
    eight_objects = 0
    diagonal_merges = 0
    for value in grids:
        four = extract_objects(value, connectivity=4)
        eight = extract_objects(value, connectivity=8)
        four_count, eight_count = len(four), len(eight)
        four_objects += four_count
        eight_objects += eight_count
        if four_count != eight_count:
            changed += 1
            diagonal_merges += four_count - eight_count
    return ConnectivityProfile(
        grids=len(grids), changed_grids=changed,
        four_objects=four_objects, eight_objects=eight_objects,
        diagonal_merges=diagonal_merges,
    )


def dataset_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    include_train_inputs: bool = True,
) -> dict[str, int]:
    """Count 4/8 parser disagreement for a challenge mapping.

    For a hidden challenge mapping, callers should pass only inputs from the
    test grids.  The function itself has no solution-file input by design.
    """

    grids: list[Any] = []
    for task in challenges.values():
        if include_train_inputs:
            grids.extend(pair["input"] for pair in task.get("train", []))
        grids.extend(test["input"] for test in task.get("test", []))
    profile = profile_grids(grids)
    return {
        "tasks": len(challenges),
        "grids": profile.grids,
        "changed_grids": profile.changed_grids,
        "four_objects": profile.four_objects,
        "eight_objects": profile.eight_objects,
        "diagonal_merges": profile.diagonal_merges,
    }


if __name__ == "__main__":
    diagonal = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    assert len(extract_objects(diagonal, connectivity=4)) == 2
    assert len(extract_objects(diagonal, connectivity=8)) == 1
    print("connectivity_ablation selftest: PASS")
