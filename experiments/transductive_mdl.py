"""A conservative program-plus-output MDL tie-break for ARC candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, log2
from typing import Any, Iterable

from experiments.candidate_records import CandidateRecord, normalize_grid


COLOR_ALPHABET = 10


@dataclass(frozen=True)
class TransductiveScore:
    candidate_id: str
    output_hash: str
    program_bits: float
    output_bits: float
    total_bits: float
    proof_status: str


def _shape_bits(height: int, width: int) -> float:
    return log2(max(2, height + 1)) + log2(max(2, width + 1))


def output_description_bits(reference: Any, output: Any) -> float:
    """Approximate a prefix code for output conditional on the input grid.

    Equal-shape outputs use a sparse delta code.  Shape-changing outputs pay a
    full-grid code, so visual simplicity cannot hide a structural change.
    """

    source = normalize_grid(reference)
    target = normalize_grid(output)
    source_shape = (len(source), len(source[0]))
    target_shape = (len(target), len(target[0]))
    if source_shape != target_shape:
        cells = len(target) * len(target[0])
        return _shape_bits(*target_shape) + cells * log2(COLOR_ALPHABET)
    height, width = source_shape
    flat_source = [cell for row in source for cell in row]
    flat_target = [cell for row in target for cell in row]
    changed = sum(left != right for left, right in zip(flat_source, flat_target))
    positions_bits = log2(comb(height * width, changed)) if changed else 0.0
    count_bits = log2(height * width + 1)
    color_bits = changed * log2(COLOR_ALPHABET - 1)
    return count_bits + positions_bits + color_bits


def score_candidate(record: CandidateRecord, reference: Any) -> TransductiveScore:
    """Score one structurally valid candidate; proof is metadata, not a bonus."""

    if not record.hard_valid:
        raise ValueError("transductive MDL requires a structurally valid candidate")
    output_bits = output_description_bits(reference, record.output)
    return TransductiveScore(
        candidate_id=record.candidate_id,
        output_hash=record.output_hash,
        program_bits=record.mdl_length,
        output_bits=output_bits,
        total_bits=record.mdl_length + output_bits,
        proof_status=record.proof_status,
    )


def rank_transductive(
    records: Iterable[CandidateRecord],
    reference: Any,
    *,
    verified_only: bool = False,
    unique_outputs: bool = True,
) -> tuple[TransductiveScore, ...]:
    """Rank candidates by joint MDL, optionally retaining one output class."""

    scores = []
    for record in records:
        if verified_only and record.proof_status != "demo_verified":
            continue
        try:
            scores.append(score_candidate(record, reference))
        except (TypeError, ValueError, IndexError):
            continue
    if not unique_outputs:
        return tuple(sorted(scores, key=lambda item: (
            item.total_bits, item.program_bits, item.candidate_id
        )))
    representatives: dict[str, TransductiveScore] = {}
    for score in scores:
        old = representatives.get(score.output_hash)
        if old is None or (
            score.proof_status != "demo_verified",
            score.total_bits,
            score.candidate_id,
        ) < (
            old.proof_status != "demo_verified",
            old.total_bits,
            old.candidate_id,
        ):
            representatives[score.output_hash] = score
    return tuple(sorted(representatives.values(), key=lambda item: (
        item.total_bits, item.program_bits, item.candidate_id
    )))
