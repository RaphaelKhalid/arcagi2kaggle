"""Local evaluation harness for the ARC Prize 2026 Kaggle competition.

Modules:
    tasks    -- data loading, the Task dataclass, and grid helpers.
    validate -- strict submission-schema validation with structured errors.
    score    -- the official exact-match metric plus per-task breakdowns.
    folds    -- deterministic 5-fold split of the training tasks.
    cli      -- command-line entry point (``python -m arc.cli``).

The package is pure stdlib and treats grids as ``list[list[int]]``.
"""

from arc.tasks import Task, load_challenges, load_solutions, load_tasks
from arc.validate import ValidationIssue, validate_submission
from arc.score import ScoreReport, TaskScore, score_submission
from arc.folds import NUM_FOLDS, SHADOW_FOLD, compute_folds

__all__ = [
    "Task",
    "load_challenges",
    "load_solutions",
    "load_tasks",
    "ValidationIssue",
    "validate_submission",
    "ScoreReport",
    "TaskScore",
    "score_submission",
    "NUM_FOLDS",
    "SHADOW_FOLD",
    "compute_folds",
]
