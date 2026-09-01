"""Task loading and grid helpers for the ARC Prize 2026 harness.

An ARC grid is a rectangular ``list[list[int]]`` with values in 0-9 and
dimensions between 1x1 and 30x30.  A *challenges* file maps task ids to
``{"train": [{"input": g, "output": g}, ...], "test": [{"input": g}, ...]}``;
a *solutions* file maps task ids to ``[grid, ...]`` -- one ground-truth
output per test input, in the same order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

Grid = list[list[int]]

MIN_DIM = 1
MAX_DIM = 30
MIN_COLOR = 0
MAX_COLOR = 9


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def grid_error(grid: Any) -> Optional[str]:
    """Return a human-readable reason ``grid`` is not a valid ARC grid, or None.

    Checks: outer/inner types, rectangularity, 1x1-30x30 dimensions,
    integer cells (bool excluded), and the 0-9 color range.
    """
    if not isinstance(grid, list):
        return f"grid must be a list, got {type(grid).__name__}"
    if not MIN_DIM <= len(grid) <= MAX_DIM:
        return f"grid must have {MIN_DIM}-{MAX_DIM} rows, got {len(grid)}"
    width: Optional[int] = None
    for r, row in enumerate(grid):
        if not isinstance(row, list):
            return f"row {r} must be a list, got {type(row).__name__}"
        if width is None:
            width = len(row)
            if not MIN_DIM <= width <= MAX_DIM:
                return f"grid must have {MIN_DIM}-{MAX_DIM} columns, got {width}"
        elif len(row) != width:
            return f"ragged grid: row {r} has {len(row)} cells, expected {width}"
        for c, cell in enumerate(row):
            # bool is a subclass of int; reject it explicitly.
            if isinstance(cell, bool) or not isinstance(cell, int):
                return (
                    f"cell ({r},{c}) must be an int, "
                    f"got {type(cell).__name__}"
                )
            if not MIN_COLOR <= cell <= MAX_COLOR:
                return (
                    f"cell ({r},{c}) must be in {MIN_COLOR}-{MAX_COLOR}, "
                    f"got {cell}"
                )
    return None


def is_valid_grid(grid: Any) -> bool:
    """True iff ``grid`` is a well-formed ARC grid."""
    return grid_error(grid) is None


def grid_dims(grid: Grid) -> tuple[int, int]:
    """Return ``(rows, cols)`` of a (non-empty, rectangular) grid."""
    return len(grid), len(grid[0])


def grid_palette(grid: Grid) -> frozenset[int]:
    """Return the set of colors present in the grid."""
    return frozenset(cell for row in grid for cell in row)


def grids_equal(a: Grid, b: Grid) -> bool:
    """Exact equality: identical dimensions and every cell matching."""
    return a == b


# ---------------------------------------------------------------------------
# Task dataclass and loaders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """One ARC task: demonstration pairs plus one or more test inputs.

    ``test_outputs`` is populated only when a solutions file is provided;
    when present it is index-aligned with ``test_inputs``.
    """

    task_id: str
    train: list[tuple[Grid, Grid]] = field(default_factory=list)
    test_inputs: list[Grid] = field(default_factory=list)
    test_outputs: Optional[list[Grid]] = None

    @property
    def num_test(self) -> int:
        return len(self.test_inputs)

    def __repr__(self) -> str:  # keep reprs short: grids are noisy
        return (
            f"Task({self.task_id!r}, train={len(self.train)}, "
            f"test={self.num_test}, solved={self.test_outputs is not None})"
        )


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_challenges(path: str | Path) -> dict[str, Any]:
    """Load a challenges JSON file as a raw dict (task_id -> task dict)."""
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: challenges file must be a JSON object")
    return data


def load_solutions(path: str | Path) -> dict[str, list[Grid]]:
    """Load a solutions JSON file (task_id -> list of output grids)."""
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: solutions file must be a JSON object")
    return data


def load_tasks(
    challenges_path: str | Path,
    solutions_path: Optional[str | Path] = None,
) -> dict[str, Task]:
    """Load challenges (and optionally solutions) into ``Task`` objects.

    Raises ``ValueError`` if the files are structurally inconsistent
    (missing keys, mismatched test counts, or malformed grids).
    """
    challenges = load_challenges(challenges_path)
    solutions = load_solutions(solutions_path) if solutions_path else None

    tasks: dict[str, Task] = {}
    for task_id in sorted(challenges):
        raw = challenges[task_id]
        train: list[tuple[Grid, Grid]] = []
        for i, pair in enumerate(raw.get("train", [])):
            for role in ("input", "output"):
                err = grid_error(pair.get(role))
                if err:
                    raise ValueError(f"{task_id} train[{i}].{role}: {err}")
            train.append((pair["input"], pair["output"]))

        test_inputs: list[Grid] = []
        for i, item in enumerate(raw.get("test", [])):
            err = grid_error(item.get("input"))
            if err:
                raise ValueError(f"{task_id} test[{i}].input: {err}")
            test_inputs.append(item["input"])

        test_outputs: Optional[list[Grid]] = None
        if solutions is not None:
            if task_id not in solutions:
                raise ValueError(f"solutions file missing task {task_id}")
            test_outputs = solutions[task_id]
            if len(test_outputs) != len(test_inputs):
                raise ValueError(
                    f"{task_id}: {len(test_outputs)} solutions for "
                    f"{len(test_inputs)} test inputs"
                )
            for i, grid in enumerate(test_outputs):
                err = grid_error(grid)
                if err:
                    raise ValueError(f"{task_id} solution[{i}]: {err}")

        tasks[task_id] = Task(task_id, train, test_inputs, test_outputs)
    return tasks
