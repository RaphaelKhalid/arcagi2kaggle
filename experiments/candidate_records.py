"""Common candidate-record schema for offline solver replay.

The Kaggle notebook currently stores NVARC samples, TRM outputs, and Leg-C
results in different shapes.  This module normalizes only the evidence needed
for selection; it does not assign correctness from hidden labels.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

try:
    from experiments.pass2_selector import Candidate, TaskCandidate
except ModuleNotFoundError:  # direct ``python experiments/candidate_records.py``
    from pass2_selector import Candidate, TaskCandidate


Grid = tuple[tuple[int, ...], ...]
ProofStatus = Literal["unverified", "demo_verified"]


def normalize_grid(value: Any) -> Grid:
    """Convert a nested grid or array-like object into a validated immutable grid."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    rows = tuple(tuple(int(cell) for cell in row) for row in value)
    if not rows or not rows[0] or len(rows) > 30:
        raise ValueError("grid must be a non-empty grid of at most 30 rows")
    width = len(rows[0])
    if width > 30 or any(len(row) != width for row in rows):
        raise ValueError("grid must be rectangular and at most 30 columns")
    if any(cell < 0 or cell > 9 for row in rows for cell in row):
        raise ValueError("grid cells must be ARC colors 0..9")
    return rows


def grid_hash(grid: Grid) -> str:
    """Stable content hash for exact output-class deduplication."""

    payload = json.dumps(grid, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class CandidateRecord:
    """Selection evidence emitted by one solver family."""

    task_id: str
    test_index: int
    family: str
    candidate_id: str
    output: Grid
    output_hash: str
    hard_valid: bool = True
    weight: float = 1.0
    mdl_length: float = 0.0
    program_id: str | None = None
    beam_score: float | None = None
    augmentation_score: float | None = None
    augmentation_nlls: tuple[float, ...] = ()
    correlation_group: str | None = None
    proof_status: ProofStatus = "unverified"

    def __post_init__(self) -> None:
        if self.proof_status not in {"unverified", "demo_verified"}:
            raise ValueError("unknown proof status")

    @classmethod
    def from_output(
        cls,
        *,
        task_id: str,
        test_index: int,
        family: str,
        candidate_id: str,
        output: Any,
        hard_valid: bool = True,
        weight: float = 1.0,
        mdl_length: float = 0.0,
        program_id: str | None = None,
        beam_score: float | None = None,
        augmentation_score: float | None = None,
        augmentation_nlls: Iterable[float] | None = None,
        correlation_group: str | None = None,
        proof_status: ProofStatus = "unverified",
    ) -> "CandidateRecord":
        grid = normalize_grid(output)
        return cls(
            task_id=task_id,
            test_index=int(test_index),
            family=family,
            candidate_id=candidate_id,
            output=grid,
            output_hash=grid_hash(grid),
            hard_valid=bool(hard_valid),
            weight=float(weight),
            mdl_length=float(mdl_length),
            program_id=program_id,
            beam_score=beam_score,
            augmentation_score=augmentation_score,
            augmentation_nlls=tuple(float(score) for score in (augmentation_nlls or ())),
            correlation_group=correlation_group,
            proof_status=proof_status,
        )

    def as_selector_candidate(self) -> Candidate:
        return Candidate(
            output_hash=self.output_hash,
            family=self.family,
            weight=self.weight,
            mdl_length=self.mdl_length,
            hard_valid=self.hard_valid,
            correlation_group=self.correlation_group,
            # Structural validity is carried to the selector separately from
            # proof status; unverified neural candidates remain eligible.
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record for append-only notebook logging."""

        result = asdict(self)
        result["output"] = [list(row) for row in self.output]
        return result


def to_selector_candidates(records: Iterable[CandidateRecord]) -> list[Candidate]:
    """Convert records to output-level selector inputs."""

    return [record.as_selector_candidate() for record in records]


def from_nvarc_sample(
    *,
    task_id: str,
    test_index: int,
    candidate_id: str,
    sample: Any,
) -> CandidateRecord:
    """Adapt one decoded NVARC sample without inventing a posterior weight."""

    aug_scores = sample.get("score_aug", [])
    if hasattr(aug_scores, "tolist"):
        aug_scores = aug_scores.tolist()
    aug_scores = tuple(float(score) for score in aug_scores)
    augmentation_score = (
        sum(aug_scores) / len(aug_scores) if aug_scores else None
    )
    beam_score = sample.get("beam_score")
    return CandidateRecord.from_output(
        task_id=task_id,
        test_index=test_index,
        family="nvarc",
        candidate_id=candidate_id,
        output=sample["solution"],
        weight=1.0,
        beam_score=None if beam_score is None else float(beam_score),
        augmentation_score=augmentation_score,
        augmentation_nlls=aug_scores,
        correlation_group=sample.get("correlation_group"),
        proof_status="unverified",
    )


def from_legc_result(
    *,
    task_id: str,
    result: Any,
) -> list[CandidateRecord]:
    """Adapt train-verified Leg-C outputs, including a distinct alternate."""

    if not isinstance(result, dict) or not result.get("verified"):
        return []
    records: list[CandidateRecord] = []
    for test_index, output in enumerate(result.get("outputs", [])):
        if not isinstance(output, dict) or "attempt" not in output:
            continue
        source = output.get("program")
        program_id = None if source is None else text_hash(str(source))
        records.append(
            CandidateRecord.from_output(
                task_id=task_id,
                test_index=test_index,
                family="legc",
                candidate_id=f"{task_id}:{test_index}:primary",
                output=output["attempt"],
                program_id=program_id,
                mdl_length=float(len(source)) if source is not None else 0.0,
                correlation_group=program_id,
                proof_status="demo_verified",
            )
        )
        if output.get("alt") is not None:
            records.append(
                CandidateRecord.from_output(
                    task_id=task_id,
                    test_index=test_index,
                    family="legc",
                    candidate_id=f"{task_id}:{test_index}:alternate",
                output=output["alt"],
                program_id=None,
                correlation_group=program_id,
                proof_status="demo_verified",
                )
            )
    return records


def text_hash(value: str) -> str:
    """Stable hash for program/source identifiers without storing source text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def to_task_candidates(
    records: Iterable[CandidateRecord],
    *,
    task_id: str,
    n_test: int,
) -> list[TaskCandidate]:
    """Build complete program vectors, dropping incomplete program records.

    A program is eligible only when it supplies exactly one candidate for every
    test position.  This prevents accidental mixing of unrelated programs
    while estimating the joint posterior.  The final official action should
    still be obtained by marginalizing these vectors.
    """

    grouped: dict[tuple[str, str], dict[int, CandidateRecord]] = defaultdict(dict)
    invalid_keys: set[tuple[str, str]] = set()
    for record in records:
        if record.task_id != task_id or record.program_id is None:
            continue
        key = (record.family, record.program_id)
        if not record.hard_valid:
            invalid_keys.add(key)
            continue
        if record.test_index in grouped[key]:
            raise ValueError("duplicate test index for one program record")
        grouped[key][record.test_index] = record

    result: list[TaskCandidate] = []
    expected = set(range(n_test))
    for (family, program_id), by_index in grouped.items():
        if (family, program_id) in invalid_keys:
            continue
        if set(by_index) != expected:
            continue
        ordered = tuple(by_index[index] for index in range(n_test))
        groups = {
            record.correlation_group
            for record in ordered
            if record.correlation_group is not None
        }
        result.append(
            TaskCandidate(
                program_hash=program_id,
                output_vector=tuple(record.output_hash for record in ordered),
                family=family,
                weight=sum(record.weight for record in ordered) / n_test,
                mdl_length=max(record.mdl_length for record in ordered),
                correlation_group=next(iter(groups)) if len(groups) == 1 else None,
            )
        )
    return result


if __name__ == "__main__":
    first = CandidateRecord.from_output(
        task_id="task", test_index=0, family="program", candidate_id="p-0",
        program_id="p", output=[[0, 1], [1, 0]],
    )
    second = CandidateRecord.from_output(
        task_id="task", test_index=1, family="program", candidate_id="p-1",
        program_id="p", output=[[1]],
    )
    vectors = to_task_candidates([first, second], task_id="task", n_test=2)
    assert len(vectors) == 1 and vectors[0].program_hash == "p"
    print("candidate_records selftest: PASS", vectors[0].output_vector)
