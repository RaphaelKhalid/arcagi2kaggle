"""Offline replay from normalized candidate records to an exact submission."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

try:
    from experiments.candidate_records import CandidateRecord
    from experiments.official_metric import official_pass2_score
    from experiments.pass2_selector import select_pass2
except ModuleNotFoundError:  # direct ``python experiments/replay_harness.py``
    from candidate_records import CandidateRecord
    from official_metric import official_pass2_score
    from pass2_selector import select_pass2


def build_submission(
    records: list[CandidateRecord],
    *,
    n_test_by_task: Mapping[str, int],
    family_priors: Mapping[str, float] | None = None,
    collapse_correlated: bool = False,
) -> dict[str, list[dict[str, object]]]:
    """Select two outputs per position and fill uncovered positions safely.

    The fallback is deliberately a valid placeholder, not an implicit claim
    that ``[[0]]`` is correct.  Every task ID and every test position is
    emitted, so the official scorer can expose coverage gaps rather than
    silently dropping them.
    """

    grouped: dict[tuple[str, int], list[CandidateRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.task_id, record.test_index)].append(record)

    submission: dict[str, list[dict[str, object]]] = {}
    for task_id, n_test in n_test_by_task.items():
        outputs: list[dict[str, object]] = []
        for test_index in range(n_test):
            position_records = grouped[(task_id, test_index)]
            selected_hashes = select_pass2(
                [record.as_selector_candidate() for record in position_records],
                family_priors=family_priors,
                collapse_correlated=collapse_correlated,
            )
            representatives: dict[str, CandidateRecord] = {}
            for record in position_records:
                old = representatives.get(record.output_hash)
                if old is None or (record.mdl_length, record.candidate_id) < (
                    old.mdl_length, old.candidate_id
                ):
                    representatives[record.output_hash] = record
            selected = [representatives[output_hash] for output_hash in selected_hashes]
            guesses = [
                [list(row) for row in record.output]
                for record in selected
            ]
            if not guesses:
                guesses = [[[0]]]
            while len(guesses) < 2:
                guesses.append([row[:] for row in guesses[0]])
            outputs.append({"attempt_1": guesses[0], "attempt_2": guesses[1]})
        submission[task_id] = outputs
    return submission


def replay_score(
    records: list[CandidateRecord],
    *,
    solutions: Mapping[str, list[object]],
    family_priors: Mapping[str, float] | None = None,
    collapse_correlated: bool = False,
) -> float:
    """Build a complete submission and score it with the official oracle."""

    n_test_by_task = {task_id: len(outputs) for task_id, outputs in solutions.items()}
    submission = build_submission(
        records,
        n_test_by_task=n_test_by_task,
        family_priors=family_priors,
        collapse_correlated=collapse_correlated,
    )
    return official_pass2_score(submission, solutions)


if __name__ == "__main__":
    records = [
        CandidateRecord.from_output(
            task_id="t", test_index=0, family="dense", candidate_id="wrong",
            output=[[0]], weight=10,
        ),
        CandidateRecord.from_output(
            task_id="t", test_index=0, family="program", candidate_id="right",
            output=[[1]],
        ),
    ]
    score = replay_score(records, solutions={"t": [[[1]]]})
    assert score == 1.0, score
    print("replay_harness selftest: PASS", score)
