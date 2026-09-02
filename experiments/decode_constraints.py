"""Conditional structural constraints for ARC grid decoding.

The constraints are inferred only from demonstration outputs.  Once an
invariant hypothesis is calibrated, it can prune token search; before that,
unknown or weakly supported invariants must leave decoding unconstrained.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

try:
    from experiments.object_deltas import Grid, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/decode_constraints.py``
    from object_deltas import Grid, normalize_grid


_PERMUTE_RE = re.compile(r"^permute([0-9]{10})$")


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _palette(grid: Grid) -> frozenset[int]:
    return frozenset(cell for row in grid for cell in row)


@dataclass(frozen=True)
class GridConstraints:
    """A partial shape/palette constraint for one decoded view."""

    height: int | None = None
    width: int | None = None
    palette: frozenset[int] | None = None

    def validate(self, value: Any) -> bool:
        """Return whether a complete grid satisfies all known constraints."""

        grid = normalize_grid(value)
        if self.height is not None and len(grid) != self.height:
            return False
        if self.width is not None and len(grid[0]) != self.width:
            return False
        return self.palette is None or _palette(grid) <= self.palette

    def max_new_tokens(self) -> int | None:
        """Return a safe upper bound for digit/newline/EOS decoding tokens."""

        if self.height is None or self.width is None:
            return None
        # One token per cell, one newline per row, plus a conservative EOS
        # margin.  The margin avoids coupling this research helper to a
        # tokenizer's exact treatment of the final newline.
        return self.height * (self.width + 1) + 2

    def transformed(
        self,
        operations: Iterable[str],
    ) -> "GridConstraints":
        """Transform shape/palette constraints through a view operation list."""

        height, width = self.height, self.width
        palette = self.palette
        for operation in operations:
            if operation == "transpose" or operation == "rot90":
                if height is not None and width is not None:
                    height, width = width, height
                continue
            match = _PERMUTE_RE.fullmatch(operation)
            if match and palette is not None:
                permutation = tuple(int(value) for value in match.group(1))
                palette = frozenset(permutation[color] for color in palette)
                continue
            if operation in {"copy", "out", "ex", "run"}:
                continue
            raise ValueError(f"unsupported view operation: {operation}")
        return GridConstraints(height, width, palette)


def infer_constraints(outputs: Iterable[Any]) -> GridConstraints:
    """Infer candidate invariants shared by every non-empty demo output."""

    grids = tuple(normalize_grid(output) for output in outputs)
    if not grids:
        raise ValueError("at least one demonstration output is required")
    shapes = {_shape(grid) for grid in grids}
    palettes = {_palette(grid) for grid in grids}
    height, width = next(iter(shapes)) if len(shapes) == 1 else (None, None)
    palette = next(iter(palettes)) if len(palettes) == 1 else None
    return GridConstraints(height, width, palette)


def infer_task_constraints(task: Mapping[str, Any]) -> GridConstraints:
    """Infer constraints from a task's labeled training outputs only."""

    return infer_constraints(pair["output"] for pair in task.get("train", []))


def parse_view_operations(key: str) -> tuple[str, ...]:
    """Parse the small view suffix used by the baseline augmentation keys."""

    operations: list[str] = []
    for operation in key.split(".")[1:]:
        if operation.startswith("permute"):
            if _PERMUTE_RE.fullmatch(operation) is None:
                raise ValueError(f"invalid color permutation: {operation}")
            operations.append(operation)
        elif operation in {"transpose", "rot90", "copy", "out", "ex", "run"}:
            operations.append(operation)
        elif operation:
            raise ValueError(f"unsupported view operation: {operation}")
    return tuple(operations)


if __name__ == "__main__":
    base = infer_constraints([[[0, 1], [1, 0]], [[0, 1], [1, 0]]])
    view = base.transformed(("transpose", "permute1023456789"))
    assert view == GridConstraints(2, 2, frozenset({0, 1}))
    assert base.max_new_tokens() == 8
    print("decode_constraints selftest: PASS")
