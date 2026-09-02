"""Proof-gated renderer for one-object deletion tasks.

The compiler derives the erased object from the exact cell difference in each
demo, then anti-unifies a small color-blind guard vocabulary.  It never guesses
which object to delete when the guard is non-unique.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    from experiments.candidate_records import CandidateRecord, normalize_grid
    from experiments.object_deltas import Object, extract_objects
except ModuleNotFoundError:  # direct ``python experiments/object_delete_renderer.py``
    from candidate_records import CandidateRecord, normalize_grid
    from object_deltas import Object, extract_objects


Grid = tuple[tuple[int, ...], ...]
Guard = Callable[[Object, Grid], Any]


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _bbox(obj: Object) -> tuple[int, int]:
    rows = [row for row, _, _ in obj.cells]
    cols = [col for _, col, _ in obj.cells]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _border(obj: Object, grid: Grid) -> tuple[bool, bool, bool, bool]:
    height, width = len(grid), len(grid[0])
    rows = [row for row, _, _ in obj.cells]
    cols = [col for _, col, _ in obj.cells]
    return (min(rows) == 0, min(cols) == 0,
            max(rows) == height - 1, max(cols) == width - 1)


def _guard_vocabulary() -> tuple[tuple[str, Guard], ...]:
    return (
        ("shape", lambda obj, grid: obj.shape),
        ("area", lambda obj, grid: len(obj.cells)),
        ("bbox", lambda obj, grid: _bbox(obj)),
        ("area_bbox", lambda obj, grid: (len(obj.cells), _bbox(obj))),
        ("area_border", lambda obj, grid: (len(obj.cells), _border(obj, grid))),
        ("shape_border", lambda obj, grid: (obj.shape, _border(obj, grid))),
    )


def _deleted_object(source: Grid, target: Grid) -> Object | None:
    if (len(source), len(source[0])) != (len(target), len(target[0])):
        return None
    source_background = _background(source)
    target_background = _background(target)
    if source_background != target_background:
        return None
    changed = {
        (row, col)
        for row in range(len(source))
        for col in range(len(source[0]))
        if source[row][col] != target[row][col]
    }
    if not changed:
        return None
    matches = []
    for obj in extract_objects(source):
        cells = {(row, col) for row, col, _ in obj.cells}
        if cells != changed:
            continue
        if any(target[row][col] != target_background for row, col in changed):
            continue
        matches.append(obj)
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class DeleteProgram:
    guard_name: str
    guard: Guard
    guard_value: Any

    @property
    def name(self) -> str:
        return f"delete_{self.guard_name}"

    @property
    def mdl_length(self) -> float:
        return 3.0 + (1.0 if self.guard_name in {"shape", "shape_border"} else 0.0)

    def apply(self, value: Any) -> Grid | None:
        grid = normalize_grid(value)
        objects = extract_objects(grid)
        matches = [obj for obj in objects
                   if self.guard(obj, grid) == self.guard_value]
        if len(matches) != 1:
            return None
        result = [list(row) for row in grid]
        background = _background(grid)
        for row, col, _ in matches[0].cells:
            result[row][col] = background
        return tuple(tuple(row) for row in result)


def fit_delete_programs(
    train_pairs: list[tuple[Any, Any]],
) -> tuple[DeleteProgram, ...]:
    """Fit all bounded one-object delete guards that replay every demo."""

    if not train_pairs:
        return ()
    pairs = [(normalize_grid(source), normalize_grid(target))
             for source, target in train_pairs]
    deleted = []
    for source, target in pairs:
        obj = _deleted_object(source, target)
        if obj is None:
            return ()
        deleted.append((obj, source))
    programs: list[DeleteProgram] = []
    for name, guard in _guard_vocabulary():
        values = [guard(obj, source) for obj, source in deleted]
        if any(value != values[0] for value in values[1:]):
            continue
        program = DeleteProgram(name, guard, values[0])
        if any(sum(guard(obj, source) == values[0]
                   for obj in extract_objects(source)) != 1
                   for source, _ in pairs):
            continue
        if all(program.apply(source) == target for source, target in pairs):
            programs.append(program)
    return tuple(programs)


def build_delete_records(
    challenges: Mapping[str, Mapping[str, Any]],
) -> tuple[list[CandidateRecord], dict[str, int]]:
    records: list[CandidateRecord] = []
    verified_tasks: dict[str, int] = {}
    for task_id, task in challenges.items():
        programs = fit_delete_programs(
            [(pair["input"], pair["output"])
             for pair in task.get("train", [])]
        )
        if programs:
            verified_tasks[task_id] = len(programs)
        for test_index, item in enumerate(task.get("test", [])):
            for program in programs:
                output = program.apply(item["input"])
                if output is None:
                    continue
                records.append(CandidateRecord.from_output(
                    task_id=task_id,
                    test_index=test_index,
                    family="object_delete",
                    candidate_id=f"{task_id}:{test_index}:{program.name}",
                    output=output,
                    program_id=program.name,
                    mdl_length=program.mdl_length,
                    proof_status="demo_verified",
                ))
    return records, verified_tasks


if __name__ == "__main__":
    program = fit_delete_programs([
        ([[0, 1, 0], [0, 0, 2]], [[0, 0, 0], [0, 0, 2]])
    ])
    assert program and program[0].apply([[0, 1, 0]]) == ((0, 0, 0),)
    print("object_delete_renderer selftest: PASS")
