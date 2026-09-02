"""Small, deterministic utilities for first-divergence program repair."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence


Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GridDiff:
    """Bounded, machine-readable difference between two trace states."""

    observed_shape: tuple[int, int] | None
    expected_shape: tuple[int, int]
    changed_cells: tuple[tuple[int, int, int | None, int | None], ...]
    truncated: bool = False

    @property
    def changed_count(self) -> int | None:
        return None if self.observed_shape is None else len(self.changed_cells)

    def render(self) -> str:
        observed = "error" if self.observed_shape is None else (
            f"{self.observed_shape[0]}x{self.observed_shape[1]}"
        )
        expected = f"{self.expected_shape[0]}x{self.expected_shape[1]}"
        cells = ", ".join(
            f"({row},{col}):{old}->{new}"
            for row, col, old, new in self.changed_cells
        ) or "none"
        suffix = "; truncated=true" if self.truncated else ""
        return (
            f"shape observed={observed} expected={expected}; "
            f"changed={self.changed_count}; cells={cells}{suffix}"
        )


def normalize_state(value: Any) -> Grid:
    """Normalize one intermediate grid without accepting ragged states."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    state = tuple(tuple(int(cell) for cell in row) for row in value)
    if not state or not state[0] or any(len(row) != len(state[0]) for row in state):
        raise ValueError("trace states must be non-empty rectangular grids")
    return state


def state_hash(state: Grid) -> str:
    payload = json.dumps(state, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def summarize_grid_diff(
    observed: Any | None,
    expected: Any,
    *,
    max_changes: int = 128,
) -> GridDiff:
    """Summarize a grid mismatch without allowing unbounded prompt growth."""

    if max_changes <= 0:
        raise ValueError("max_changes must be positive")
    expected_grid = normalize_state(expected)
    if observed is None:
        return GridDiff(None, (len(expected_grid), len(expected_grid[0])), ())
    observed_grid = normalize_state(observed)
    height = max(len(observed_grid), len(expected_grid))
    width = max(len(observed_grid[0]), len(expected_grid[0]))
    all_changes: list[tuple[int, int, int | None, int | None]] = []
    for row in range(height):
        for col in range(width):
            old = (observed_grid[row][col]
                   if row < len(observed_grid) and col < len(observed_grid[0])
                   else None)
            new = (expected_grid[row][col]
                   if row < len(expected_grid) and col < len(expected_grid[0])
                   else None)
            if old != new:
                all_changes.append((row, col, old, new))
    return GridDiff(
        (len(observed_grid), len(observed_grid[0])),
        (len(expected_grid), len(expected_grid[0])),
        tuple(all_changes[:max_changes]),
        len(all_changes) > max_changes,
    )


def first_divergence(
    candidate_states: Sequence[Any],
    reference_states: Sequence[Any],
) -> int | None:
    """Return the first differing state index, or ``None`` if traces match."""

    candidate = tuple(normalize_state(state) for state in candidate_states)
    reference = tuple(normalize_state(state) for state in reference_states)
    for index, (left, right) in enumerate(zip(candidate, reference)):
        if left != right:
            return index
    if len(candidate) != len(reference):
        return min(len(candidate), len(reference))
    return None


def repair_targets(
    divergence_index: int | None,
    operation_names: Sequence[str],
) -> tuple[str, ...]:
    """Map a state mismatch to the smallest safe repair scope.

    State ``s_0`` is the input.  A mismatch at ``s_j`` for ``j > 0`` is first
    attributable to operation ``o_j`` (zero-based operation index ``j - 1``).
    A mismatch at the input is not an operation repair and is labeled
    ``"input"`` so callers cannot silently mutate a program to hide it.
    """

    if divergence_index is None:
        return ()
    if divergence_index == 0:
        return ("input",)
    operation_index = divergence_index - 1
    if operation_index >= len(operation_names):
        return ("trace_length",)
    return (str(operation_names[operation_index]),)


if __name__ == "__main__":
    divergence = first_divergence(
        [[[0]], [[1]], [[1]]],
        [[[0]], [[1]], [[2]]],
    )
    assert divergence == 2
    assert repair_targets(divergence, ["paint", "move"]) == ("move",)
    print("trace_repair selftest: PASS", divergence)
