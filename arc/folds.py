"""Deterministic 5-fold split of the ARC training tasks.

The assignment depends only on the set of task ids -- not on dict
ordering, Python's randomized ``hash()``, or any RNG state -- so it is
byte-for-byte stable across runs, machines, and Python versions:

1. Each task id is keyed by ``sha256(task_id)`` (a fixed cryptographic
   hash, unaffected by ``PYTHONHASHSEED``).
2. Task ids are sorted by ``(digest, task_id)``.
3. Fold ``position % NUM_FOLDS`` is assigned round-robin, which also
   guarantees fold sizes are as equal as possible (200 each for the
   1000-task training set).

SHADOW FOLD POLICY: fold ``SHADOW_FOLD`` (fold 4) is the shadow
hold-out.  It must NOT be used for development, hyper-parameter tuning,
prompt iteration, or model selection.  Evaluate on it only at major
milestones (e.g. before a Kaggle submission) to get an uncontaminated
estimate of generalization; between milestones treat it as unseen.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

NUM_FOLDS = 5
SHADOW_FOLD = 4  # held out between major milestones -- see module docstring


def _digest(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()


def compute_folds(task_ids: Iterable[str]) -> dict[str, int]:
    """Map each task id to a fold in ``0..NUM_FOLDS-1``, deterministically.

    Duplicate ids are collapsed; the result depends only on the set of ids.
    """
    ordered = sorted(set(task_ids), key=lambda tid: (_digest(tid), tid))
    return {tid: pos % NUM_FOLDS for pos, tid in enumerate(ordered)}


def fold_members(assignment: dict[str, int]) -> dict[int, list[str]]:
    """Invert an assignment: fold -> sorted list of task ids."""
    members: dict[int, list[str]] = {f: [] for f in range(NUM_FOLDS)}
    for tid, fold in assignment.items():
        members[fold].append(tid)
    for ids in members.values():
        ids.sort()
    return members


def dev_task_ids(assignment: dict[str, int]) -> list[str]:
    """All task ids outside the shadow fold, sorted (safe for development)."""
    return sorted(t for t, f in assignment.items() if f != SHADOW_FOLD)


def shadow_task_ids(assignment: dict[str, int]) -> list[str]:
    """The shadow-fold task ids, sorted. Touch only at major milestones."""
    return sorted(t for t, f in assignment.items() if f == SHADOW_FOLD)
