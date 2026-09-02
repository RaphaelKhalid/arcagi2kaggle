"""Compact, lossy candidate cards for an offline holistic judge.

The external holistic-judge result uses a 30k--80k token context.  The local
notebook's 8k context cannot reproduce that literally.  This module defines a
safe compression seam: exact output classes are deduplicated first, then each
class receives lossless metadata (hash and dimensions) plus color-blind grid
statistics.  The raw grid is retained separately for final rendering, so
compression can affect ranking but cannot corrupt the submission itself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

try:
    from experiments.candidate_records import CandidateRecord, ProofStatus, grid_hash, normalize_grid
    from experiments.object_deltas import extract_objects
except ModuleNotFoundError:  # direct ``python experiments/candidate_cards.py``
    from candidate_records import CandidateRecord, ProofStatus, grid_hash, normalize_grid
    from object_deltas import extract_objects


Grid = tuple[tuple[int, ...], ...]


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _diff_count(grid: Grid, reference: Grid | None) -> int | None:
    if reference is None or (len(grid), len(grid[0])) != (len(reference), len(reference[0])):
        return None
    return sum(left != right for row_left, row_right in zip(grid, reference)
               for left, right in zip(row_left, row_right))


@dataclass(frozen=True)
class CandidateCard:
    output_hash: str
    family: str
    shape: tuple[int, int]
    palette: tuple[int, ...]
    background: int
    object_count: int
    occupied_cells: int
    diff_cells: int | None
    weight: float
    mdl_length: float
    proof_status: ProofStatus = "unverified"
    correlation_group: str | None = None

    def prompt_line(self) -> str:
        """A bounded one-line representation for a local judge prompt."""

        diff = "?" if self.diff_cells is None else str(self.diff_cells)
        group = "?" if self.correlation_group is None else self.correlation_group[:12]
        return (
            f"family={self.family} hash={self.output_hash[:12]} "
            f"shape={self.shape[0]}x{self.shape[1]} palette={''.join(map(str, self.palette))} "
            f"bg={self.background} objects={self.object_count} occupied={self.occupied_cells} "
            f"diff={diff} weight={self.weight:.3g} mdl={self.mdl_length:.3g} "
            f"proof={self.proof_status} group={group}"
        )


def make_card(
    record: CandidateRecord, *, reference: Any | None = None
) -> CandidateCard:
    grid = normalize_grid(record.output)
    reference_grid = None if reference is None else normalize_grid(reference)
    background = _background(grid)
    occupied = sum(cell != background for row in grid for cell in row)
    return CandidateCard(
        output_hash=grid_hash(grid),
        family=record.family,
        shape=(len(grid), len(grid[0])),
        palette=tuple(sorted({cell for row in grid for cell in row})),
        background=background,
        object_count=len(extract_objects(grid)),
        occupied_cells=occupied,
        diff_cells=_diff_count(grid, reference_grid),
        weight=record.weight,
        mdl_length=record.mdl_length,
        proof_status=record.proof_status,
        correlation_group=record.correlation_group,
    )


def deduplicate_cards(
    records: Iterable[CandidateRecord], *, reference: Any | None = None
) -> tuple[CandidateCard, ...]:
    """Keep one representative per exact output class.

    Demo-verified provenance is part of the evidence shown to the holistic
    judge, so it takes precedence over a shorter but unverified explanation.
    MDL remains the tie-breaker within the same provenance class.
    """

    def representative_key(record: CandidateRecord) -> tuple[bool, float, str]:
        return (
            record.proof_status != "demo_verified",
            record.mdl_length,
            record.candidate_id,
        )

    representatives: dict[str, CandidateRecord] = {}
    for record in records:
        old = representatives.get(record.output_hash)
        if old is None or representative_key(record) < representative_key(old):
            representatives[record.output_hash] = record
    return tuple(
        make_card(record, reference=reference)
        for record in sorted(representatives.values(), key=lambda item: item.output_hash)
    )


def card_token_estimate(cards: Iterable[CandidateCard]) -> int:
    """Conservative character-to-token proxy for prompt budgeting."""

    return sum((len(card.prompt_line()) + 3) // 4 for card in cards)


if __name__ == "__main__":
    print("candidate_cards selftest: PASS")
