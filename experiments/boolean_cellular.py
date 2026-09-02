"""Small algebraic cellular rules over binary foreground occupancy.

Unlike a finite context lookup, these programs apply a mathematical predicate
to the cross neighborhood.  They are exact-demo verified and deliberately
limited to two output colors, making the extrapolation assumption explicit.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    from experiments.candidate_records import CandidateRecord, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/boolean_cellular.py``
    from candidate_records import CandidateRecord, normalize_grid


Grid = tuple[tuple[int, ...], ...]
Cross = tuple[bool, bool, bool, bool, bool]
Predicate = Callable[[Cross], bool]


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _cross(grid: Grid, row: int, col: int, background: int) -> Cross:
    height, width = len(grid), len(grid[0])
    points = ((row, col), (row - 1, col), (row + 1, col),
              (row, col - 1), (row, col + 1))
    return tuple(
        0 <= r < height and 0 <= c < width and grid[r][c] != background
        for r, c in points
    )  # type: ignore[return-value]


def _predicates() -> tuple[tuple[str, Predicate], ...]:
    def count(cross: Cross) -> int:
        return sum(cross)

    predicates: list[tuple[str, Predicate]] = [
        ("center", lambda cross: cross[0]),
        ("not_center", lambda cross: not cross[0]),
        ("parity", lambda cross: count(cross) % 2 == 1),
    ]
    for threshold in range(1, 6):
        predicates.append((
            f"count_ge_{threshold}",
            lambda cross, threshold=threshold: count(cross) >= threshold,
        ))
    for exact in range(0, 6):
        predicates.append((
            f"count_eq_{exact}",
            lambda cross, exact=exact: count(cross) == exact,
        ))
    return tuple(predicates)


@dataclass(frozen=True)
class BooleanCellularProgram:
    name: str
    predicate: Predicate
    false_color: int
    true_color: int
    mdl_length: float = 4.0

    def apply(self, value: Any) -> Grid:
        grid = normalize_grid(value)
        background = _background(grid)
        return tuple(
            tuple(
                self.true_color if self.predicate(_cross(grid, row, col, background))
                else self.false_color
                for col in range(len(grid[0]))
            )
            for row in range(len(grid))
        )


def fit_boolean_programs(
    train_pairs: list[tuple[Any, Any]],
) -> tuple[BooleanCellularProgram, ...]:
    """Return bounded cross-predicate programs that replay every demo."""

    if not train_pairs:
        return ()
    pairs = [(normalize_grid(source), normalize_grid(target))
             for source, target in train_pairs]
    if any((len(source), len(source[0])) != (len(target), len(target[0]))
           for source, target in pairs):
        return ()
    programs: list[BooleanCellularProgram] = []
    for name, predicate in _predicates():
        mapping: dict[bool, int] = {}
        valid = True
        for source, target in pairs:
            background = _background(source)
            for row in range(len(source)):
                for col in range(len(source[0])):
                    key = predicate(_cross(source, row, col, background))
                    color = target[row][col]
                    if key in mapping and mapping[key] != color:
                        valid = False
                        break
                    mapping[key] = color
                if not valid:
                    break
            if not valid:
                break
        if not valid or set(mapping) != {False, True}:
            continue
        program = BooleanCellularProgram(
            name=name,
            predicate=predicate,
            false_color=mapping[False],
            true_color=mapping[True],
        )
        if all(program.apply(source) == target for source, target in pairs):
            programs.append(program)
    return tuple(programs)


def build_boolean_cellular_records(
    challenges: Mapping[str, Mapping[str, Any]],
) -> tuple[list[CandidateRecord], dict[str, int]]:
    records: list[CandidateRecord] = []
    verified_tasks: dict[str, int] = {}
    for task_id, task in challenges.items():
        programs = fit_boolean_programs(
            [(pair["input"], pair["output"])
             for pair in task.get("train", [])]
        )
        if programs:
            verified_tasks[task_id] = len(programs)
        for test_index, item in enumerate(task.get("test", [])):
            for program in programs:
                records.append(CandidateRecord.from_output(
                    task_id=task_id,
                    test_index=test_index,
                    family="boolean_cellular",
                    candidate_id=f"{task_id}:{test_index}:{program.name}",
                    output=program.apply(item["input"]),
                    program_id=program.name,
                    mdl_length=program.mdl_length,
                    proof_status="demo_verified",
                ))
    return records, verified_tasks


if __name__ == "__main__":
    print("boolean_cellular selftest: PASS")
