"""Deterministic, bounded prompt construction for verifier-guided repair."""

from __future__ import annotations

from collections.abc import Sequence

try:
    from experiments.trace_repair import summarize_grid_diff
except ModuleNotFoundError:  # direct ``python experiments/repair_prompt.py``
    from trace_repair import summarize_grid_diff


Grid = Sequence[Sequence[int]]


def render_grid(grid: Grid, *, max_cells: int = 900) -> str:
    """Render one rectangular ARC grid, rejecting oversized diagnostics."""

    rows = [tuple(int(cell) for cell in row) for row in grid]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("repair grids must be non-empty and rectangular")
    if len(rows) * len(rows[0]) > max_cells:
        raise ValueError("repair grid exceeds diagnostic cell budget")
    if any(cell < 0 or cell > 9 for row in rows for cell in row):
        raise ValueError("repair grids must use ARC colors 0-9")
    return "\n".join("".join(str(cell) for cell in row) for row in rows)


def build_repair_prompt(
    source: str,
    failed_demo: int,
    observed: Grid | None,
    expected: Grid,
    *,
    max_source_chars: int = 12_000,
) -> str:
    """Build a bounded repair request without exposing any hidden test label."""

    if not source.strip():
        raise ValueError("candidate source must be non-empty")
    if len(source) > max_source_chars:
        raise ValueError("candidate source exceeds repair prompt budget")
    if failed_demo < 0:
        raise ValueError("failed_demo must be non-negative")
    expected_text = render_grid(expected)
    observed_text = "<execution raised an exception>" if observed is None else render_grid(observed)
    diff_text = summarize_grid_diff(observed, expected).render()
    return (
        "Repair the ARC transform below. The verifier found the first failure on "
        f"demonstration index {failed_demo}.\n\n"
        "Observed candidate output:\n"
        f"{observed_text}\n\n"
        "Required demonstration output:\n"
        f"{expected_text}\n\n"
        "Structured execution diff:\n"
        f"{diff_text}\n\n"
        "Keep the function general and preserve behavior on demonstrations that "
        "already pass. Return only a revised Python transform function in a "
        "```python``` block. Do not use files, network access, hidden-test data, "
        "or hardcoded answers.\n\n"
        "Candidate source:\n"
        f"```python\n{source}\n```"
    )


if __name__ == "__main__":
    prompt = build_repair_prompt(
        "def transform(grid):\n    return grid",
        0,
        [[0]],
        [[1]],
    )
    assert "demonstration index 0" in prompt
    print("repair_prompt selftest: PASS")
