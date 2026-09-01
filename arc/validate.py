"""Strict submission validation for the ARC Prize 2026 harness.

A valid submission is a JSON object mapping EVERY task id in the
challenges file to a list with exactly one entry per test input, in
order, where each entry is ``{"attempt_1": grid, "attempt_2": grid}``
and both attempts are well-formed ARC grids.

``validate_submission`` never raises on malformed input; it returns a
list of structured :class:`ValidationIssue` objects (empty == valid).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from arc.tasks import grid_error, load_challenges

REQUIRED_ATTEMPTS = ("attempt_1", "attempt_2")


@dataclass(frozen=True)
class ValidationIssue:
    """One problem found in a submission.

    Attributes:
        code:    machine-readable issue category (see ``validate_submission``).
        message: human-readable description.
        task_id: offending task id, if the issue is task-scoped.
        index:   offending test-entry index within the task, if applicable.
        attempt: offending attempt key ("attempt_1"/"attempt_2"), if applicable.
    """

    code: str
    message: str
    task_id: Optional[str] = None
    index: Optional[int] = None
    attempt: Optional[str] = None

    def __str__(self) -> str:
        loc = ".".join(
            str(part)
            for part in (self.task_id, self.index, self.attempt)
            if part is not None
        )
        prefix = f"[{self.code}] "
        return prefix + (f"{loc}: {self.message}" if loc else self.message)


def validate_submission(
    submission: Any, challenges: dict[str, Any]
) -> list[ValidationIssue]:
    """Check ``submission`` against ``challenges``; return all issues found.

    Issue codes:
        not_object        -- submission is not a JSON object.
        missing_task      -- a challenge task id is absent.
        unknown_task      -- a submission task id is not in the challenges.
        bad_task_type     -- a task's value is not a list.
        wrong_test_count  -- entry count != number of test inputs.
        bad_entry_type    -- a test entry is not an object.
        missing_attempt   -- attempt_1 or attempt_2 absent from an entry.
        extra_key         -- an entry carries a key besides the two attempts.
        bad_grid          -- an attempt is not a well-formed ARC grid.
    """
    issues: list[ValidationIssue] = []

    if not isinstance(submission, dict):
        return [
            ValidationIssue(
                "not_object",
                f"submission must be a JSON object mapping task_id to "
                f"attempts, got {type(submission).__name__}",
            )
        ]

    for task_id in sorted(challenges):
        if task_id not in submission:
            issues.append(
                ValidationIssue(
                    "missing_task", "task missing from submission", task_id
                )
            )
    for task_id in sorted(submission):
        if task_id not in challenges:
            issues.append(
                ValidationIssue(
                    "unknown_task",
                    "task not present in challenges file",
                    task_id,
                )
            )

    for task_id in sorted(challenges):
        if task_id not in submission:
            continue
        entries = submission[task_id]
        expected = len(challenges[task_id].get("test", []))

        if not isinstance(entries, list):
            issues.append(
                ValidationIssue(
                    "bad_task_type",
                    f"expected a list of attempt dicts, "
                    f"got {type(entries).__name__}",
                    task_id,
                )
            )
            continue
        if len(entries) != expected:
            issues.append(
                ValidationIssue(
                    "wrong_test_count",
                    f"expected {expected} test entr"
                    f"{'y' if expected == 1 else 'ies'}, got {len(entries)}",
                    task_id,
                )
            )

        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                issues.append(
                    ValidationIssue(
                        "bad_entry_type",
                        f"expected an object with attempt_1/attempt_2, "
                        f"got {type(entry).__name__}",
                        task_id,
                        i,
                    )
                )
                continue
            for key in sorted(set(entry) - set(REQUIRED_ATTEMPTS)):
                issues.append(
                    ValidationIssue(
                        "extra_key", f"unexpected key {key!r}", task_id, i
                    )
                )
            for attempt in REQUIRED_ATTEMPTS:
                if attempt not in entry:
                    issues.append(
                        ValidationIssue(
                            "missing_attempt",
                            f"{attempt} is missing",
                            task_id,
                            i,
                            attempt,
                        )
                    )
                    continue
                err = grid_error(entry[attempt])
                if err:
                    issues.append(
                        ValidationIssue("bad_grid", err, task_id, i, attempt)
                    )

    return issues


def validate_submission_files(
    submission_path: str | Path, challenges_path: str | Path
) -> list[ValidationIssue]:
    """File-based wrapper around :func:`validate_submission`."""
    with open(submission_path, "r", encoding="utf-8") as fh:
        try:
            submission = json.load(fh)
        except json.JSONDecodeError as exc:
            return [
                ValidationIssue("invalid_json", f"submission is not valid JSON: {exc}")
            ]
    return validate_submission(submission, load_challenges(challenges_path))
