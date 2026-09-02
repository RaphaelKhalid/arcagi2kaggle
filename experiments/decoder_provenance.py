"""Lineage-safe aggregation for decoded ARC output classes.

This is a shadow selector contract for the Kaggle decoder. It deliberately
does not infer independence from sample count or filenames: missing metadata
is collapsed into one unknown group.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import fsum
from typing import Iterable


UNKNOWN = "__unknown__"


@dataclass(frozen=True)
class DecodedCandidate:
    output_hash: str
    weight: float = 1.0
    lineage: str | None = None
    view_family: str | None = None

    @property
    def provenance_group(self) -> tuple[str, str]:
        return (self.lineage or UNKNOWN, self.view_family or UNKNOWN)


@dataclass(frozen=True)
class DecoderClassMass:
    output_hash: str
    mass: float
    witness_groups: tuple[tuple[str, str], ...]


def rank_decoder_classes(
    candidates: Iterable[DecodedCandidate],
) -> tuple[DecoderClassMass, ...]:
    """Rank classes with one normalized distribution per provenance group."""

    candidates = tuple(candidates)
    if not candidates:
        return ()
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for candidate in candidates:
        if not candidate.output_hash:
            raise ValueError("output_hash is required")
        if candidate.weight < 0.0:
            raise ValueError("candidate weights must be non-negative")
        group = candidate.provenance_group
        grouped[group][candidate.output_hash] = max(
            grouped[group].get(candidate.output_hash, 0.0), candidate.weight
        )

    masses: dict[str, float] = defaultdict(float)
    witnesses: dict[str, set[tuple[str, str]]] = defaultdict(set)
    group_weight = 1.0 / len(grouped)
    for group, class_weights in sorted(grouped.items()):
        total = fsum(class_weights.values())
        if total <= 0.0:
            total = float(len(class_weights))
            class_weights = {output: 1.0 for output in class_weights}
        for output, weight in class_weights.items():
            masses[output] += group_weight * weight / total
            witnesses[output].add(group)

    return tuple(
        DecoderClassMass(output, masses[output], tuple(sorted(witnesses[output])))
        for output in sorted(masses, key=lambda item: (-masses[item], item))
    )


def select_decoder_pass2(
    candidates: Iterable[DecodedCandidate],
) -> tuple[str, ...]:
    """Return the top two provenance-normalized output classes."""

    return tuple(item.output_hash for item in rank_decoder_classes(candidates)[:2])


if __name__ == "__main__":
    selected = select_decoder_pass2([
        DecodedCandidate("wrong", lineage="a", view_family="rot"),
        DecodedCandidate("right", lineage="b", view_family="rot"),
    ])
    assert selected == ("right", "wrong")
    print("decoder_provenance selftest: PASS")
