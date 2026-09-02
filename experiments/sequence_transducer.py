"""Finite row/column sequence transducers for ARC grid proposals.

This family treats a grid as an ordered collection of rows or columns.  It
tests sorting, density ordering, per-line ordering, and alternating reversals
as typed layout rules.  Programs are exact-demo verified and preserve grid
dimensions; the module exists to measure sequence-factorization recall before
adding richer conditional grammars.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    from experiments.candidate_records import CandidateRecord, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/sequence_transducer.py``
    from candidate_records import CandidateRecord, normalize_grid


Grid = tuple[tuple[int, ...], ...]
Transform = Callable[[Grid], Grid]


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _rows(grid: Grid) -> list[tuple[int, ...]]:
    return [tuple(row) for row in grid]


def _cols(grid: Grid) -> list[tuple[int, ...]]:
    return [tuple(grid[row][col] for row in range(len(grid)))
            for col in range(len(grid[0]))]


def _from_cols(cols: list[tuple[int, ...]]) -> Grid:
    return tuple(tuple(cols[col][row] for col in range(len(cols)))
                 for row in range(len(cols[0])))


def _sort_rows(grid: Grid, *, reverse: bool, key: Callable[[tuple[int, ...]], Any]) -> Grid:
    return tuple(sorted(_rows(grid), key=key, reverse=reverse))


def _sort_cols(grid: Grid, *, reverse: bool, key: Callable[[tuple[int, ...]], Any]) -> Grid:
    return _from_cols(sorted(_cols(grid), key=key, reverse=reverse))


def _density_key(grid: Grid, background: int) -> Callable[[tuple[int, ...]], Any]:
    return lambda line: (sum(value != background for value in line), line)


def _row_value_sort(grid: Grid, *, reverse: bool) -> Grid:
    return tuple(tuple(sorted(row, reverse=reverse)) for row in grid)


def _col_value_sort(grid: Grid, *, reverse: bool) -> Grid:
    return _from_cols([tuple(sorted(col, reverse=reverse)) for col in _cols(grid)])


def _reverse_alternating_rows(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) if index % 2 else tuple(row)
                 for index, row in enumerate(grid))


def _reverse_alternating_cols(grid: Grid) -> Grid:
    cols = [tuple(reversed(col)) if index % 2 else tuple(col)
            for index, col in enumerate(_cols(grid))]
    return _from_cols(cols)


@dataclass(frozen=True)
class SequenceProgram:
    name: str
    transform: Transform
    mdl_length: float = 3.0


def _programs_for_grid(grid: Grid) -> tuple[SequenceProgram, ...]:
    programs = [
        SequenceProgram("rows_lex_asc", lambda value: _sort_rows(value, reverse=False, key=lambda line: line)),
        SequenceProgram("rows_lex_desc", lambda value: _sort_rows(value, reverse=True, key=lambda line: line)),
        SequenceProgram("rows_density_asc", lambda value: _sort_rows(value, reverse=False, key=_density_key(value, _background(value)))),
        SequenceProgram("rows_density_desc", lambda value: _sort_rows(value, reverse=True, key=_density_key(value, _background(value)))),
        SequenceProgram("cols_lex_asc", lambda value: _sort_cols(value, reverse=False, key=lambda line: line)),
        SequenceProgram("cols_lex_desc", lambda value: _sort_cols(value, reverse=True, key=lambda line: line)),
        SequenceProgram("cols_density_asc", lambda value: _sort_cols(value, reverse=False, key=_density_key(value, _background(value)))),
        SequenceProgram("cols_density_desc", lambda value: _sort_cols(value, reverse=True, key=_density_key(value, _background(value)))),
        SequenceProgram("row_values_asc", lambda value: _row_value_sort(value, reverse=False)),
        SequenceProgram("row_values_desc", lambda value: _row_value_sort(value, reverse=True)),
        SequenceProgram("col_values_asc", lambda value: _col_value_sort(value, reverse=False)),
        SequenceProgram("col_values_desc", lambda value: _col_value_sort(value, reverse=True)),
        SequenceProgram("alternate_row_reverse", _reverse_alternating_rows),
        SequenceProgram("alternate_col_reverse", _reverse_alternating_cols),
    ]
    return tuple(programs)


def fit_sequence_programs(
    train_pairs: list[tuple[Any, Any]],
) -> tuple[SequenceProgram, ...]:
    """Return deterministic sequence programs that replay all demos exactly."""

    if not train_pairs:
        return ()
    pairs = [(normalize_grid(source), normalize_grid(target))
             for source, target in train_pairs]
    if any((len(source), len(source[0])) != (len(target), len(target[0]))
           for source, target in pairs):
        return ()
    candidates = _programs_for_grid(pairs[0][0])
    return tuple(
        program for program in candidates
        if all(program.transform(source) == target for source, target in pairs)
    )


def build_sequence_records(
    challenges: Mapping[str, Mapping[str, Any]],
) -> tuple[list[CandidateRecord], dict[str, int]]:
    records: list[CandidateRecord] = []
    verified_tasks: dict[str, int] = {}
    for task_id, task in challenges.items():
        programs = fit_sequence_programs(
            [(pair["input"], pair["output"]) for pair in task.get("train", [])]
        )
        if programs:
            verified_tasks[task_id] = len(programs)
        for test_index, item in enumerate(task.get("test", [])):
            for program in programs:
                try:
                    records.append(CandidateRecord.from_output(
                        task_id=task_id,
                        test_index=test_index,
                        family="sequence",
                        candidate_id=f"{task_id}:{test_index}:{program.name}",
                        output=program.transform(normalize_grid(item["input"])),
                        program_id=program.name,
                        mdl_length=program.mdl_length,
                        proof_status="demo_verified",
                    ))
                except (TypeError, ValueError, IndexError):
                    continue
    return records, verified_tasks


if __name__ == "__main__":
    program = fit_sequence_programs([
        ([[0, 2], [0, 1]], [[0, 1], [0, 2]]),
        ([[3, 4], [3, 2]], [[3, 2], [3, 4]]),
    ])
    assert program
    print("sequence_transducer selftest: PASS")
