"""Exact ARC-AGI-2 submission scoring for offline replay."""

from __future__ import annotations

from typing import Any, Mapping


def official_pass2_score(
    submission: Mapping[str, list[Mapping[str, Any]]],
    solutions: Mapping[str, list[Any]],
) -> float:
    """Return correct test outputs divided by total test outputs.

    ARC-AGI-2 awards one point per test output when either attempt is an exact
    grid match.  This deliberately rejects task-weighted surrogates and
    incomplete submissions, which can otherwise make selector ablations look
    better than they are.
    """

    if set(submission) != set(solutions):
        raise ValueError("submission and solutions must contain identical task IDs")
    total = correct = 0
    for task_id, expected_outputs in solutions.items():
        attempts = submission[task_id]
        if len(attempts) != len(expected_outputs):
            raise ValueError(f"wrong number of outputs for task {task_id}")
        for index, expected in enumerate(expected_outputs):
            entry = attempts[index]
            if "attempt_1" not in entry or "attempt_2" not in entry:
                raise ValueError(f"missing attempt for {task_id}[{index}]")
            total += 1
            if entry["attempt_1"] == expected or entry["attempt_2"] == expected:
                correct += 1
    return correct / total if total else 0.0


if __name__ == "__main__":
    score = official_pass2_score(
        {"a": [{"attempt_1": [[1]], "attempt_2": [[0]]},
               {"attempt_1": [[0]], "attempt_2": [[0]]}],
         "b": [{"attempt_1": [[2]], "attempt_2": [[0]]}]},
        {"a": [[[1]], [[2]]], "b": [[[2]]]},
    )
    assert score == 2 / 3, score
    print("official_metric selftest: PASS", score)
