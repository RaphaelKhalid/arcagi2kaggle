"""Official-metric scoring for the ARC Prize 2026 harness.

For each test output, a submission earns 1 point if ``attempt_1`` OR
``attempt_2`` exactly equals the ground truth (same dimensions, every
cell equal).  The final score is ``correct outputs / total outputs``.

``score_submission`` assumes the submission has already passed
:func:`arc.validate.validate_submission`; scoring is nonetheless
defensive -- a missing task, entry, or attempt simply scores 0 for the
affected outputs rather than raising.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arc.tasks import Grid, grids_equal, load_challenges, load_solutions


@dataclass(frozen=True)
class OutputScore:
    """Result for a single test output."""

    task_id: str
    index: int
    attempt_1_correct: bool
    attempt_2_correct: bool

    @property
    def correct(self) -> bool:
        return self.attempt_1_correct or self.attempt_2_correct


@dataclass(frozen=True)
class TaskScore:
    """Aggregated results for one task."""

    task_id: str
    outputs: list[OutputScore]

    @property
    def total(self) -> int:
        return len(self.outputs)

    @property
    def correct(self) -> int:
        return sum(1 for o in self.outputs if o.correct)

    @property
    def solved(self) -> bool:
        """True iff every test output of the task is correct."""
        return self.correct == self.total


@dataclass(frozen=True)
class ScoreReport:
    """Full scoring breakdown for a submission."""

    per_task: dict[str, TaskScore] = field(default_factory=dict)

    @property
    def total_outputs(self) -> int:
        return sum(t.total for t in self.per_task.values())

    @property
    def correct_outputs(self) -> int:
        return sum(t.correct for t in self.per_task.values())

    @property
    def score(self) -> float:
        """The official metric: correct outputs / total outputs."""
        total = self.total_outputs
        return self.correct_outputs / total if total else 0.0

    @property
    def attempt_breakdown(self) -> dict[str, int]:
        """Correct outputs split by which attempt(s) matched.

        Keys: ``attempt_1_only``, ``attempt_2_only``, ``both``.
        """
        a1_only = a2_only = both = 0
        for task in self.per_task.values():
            for out in task.outputs:
                if out.attempt_1_correct and out.attempt_2_correct:
                    both += 1
                elif out.attempt_1_correct:
                    a1_only += 1
                elif out.attempt_2_correct:
                    a2_only += 1
        return {"attempt_1_only": a1_only, "attempt_2_only": a2_only, "both": both}

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        bd = self.attempt_breakdown
        solved_tasks = sum(1 for t in self.per_task.values() if t.solved)
        lines = [
            f"score: {self.score:.6f}",
            f"correct outputs: {self.correct_outputs}/{self.total_outputs}",
            f"fully solved tasks: {solved_tasks}/{len(self.per_task)}",
            f"solved by attempt_1 only: {bd['attempt_1_only']}",
            f"solved by attempt_2 only: {bd['attempt_2_only']}",
            f"solved by both attempts:  {bd['both']}",
        ]
        return "\n".join(lines)


def _attempt_matches(entry: Any, key: str, truth: Grid) -> bool:
    """True iff ``entry[key]`` exists and exactly equals ``truth``."""
    if not isinstance(entry, dict) or key not in entry:
        return False
    return grids_equal(entry[key], truth)


def score_submission(
    submission: Any,
    challenges: dict[str, Any],
    solutions: dict[str, list[Grid]],
) -> ScoreReport:
    """Score ``submission`` on every test output listed in ``challenges``.

    ``solutions`` must cover every task in ``challenges`` with one ground
    truth grid per test input (a ``ValueError`` is raised otherwise --
    that is a harness misuse, not a submission defect).
    """
    per_task: dict[str, TaskScore] = {}
    for task_id in sorted(challenges):
        if task_id not in solutions:
            raise ValueError(f"solutions file missing task {task_id}")
        truths = solutions[task_id]
        num_test = len(challenges[task_id].get("test", []))
        if len(truths) != num_test:
            raise ValueError(
                f"{task_id}: {len(truths)} solutions for {num_test} test inputs"
            )

        entries = submission.get(task_id, []) if isinstance(submission, dict) else []
        outputs: list[OutputScore] = []
        for i, truth in enumerate(truths):
            entry = entries[i] if isinstance(entries, list) and i < len(entries) else None
            outputs.append(
                OutputScore(
                    task_id,
                    i,
                    _attempt_matches(entry, "attempt_1", truth),
                    _attempt_matches(entry, "attempt_2", truth),
                )
            )
        per_task[task_id] = TaskScore(task_id, outputs)
    return ScoreReport(per_task)


def score_submission_files(
    submission_path: str | Path,
    challenges_path: str | Path,
    solutions_path: str | Path,
) -> ScoreReport:
    """File-based wrapper around :func:`score_submission`."""
    with open(submission_path, "r", encoding="utf-8") as fh:
        submission = json.load(fh)
    return score_submission(
        submission,
        load_challenges(challenges_path),
        load_solutions(solutions_path),
    )
