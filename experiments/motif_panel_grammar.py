"""Typed motif-to-panel renderers with bounded panel-transform induction.

The input is treated as a motif and the output as a panel of equal-sized
motif instances.  A small family of row/column parity rules is inferred from
the panel transform matrix and then replayed on test inputs.  Every program is
required to fit all visible demos exactly; this is a proposal-recall probe,
not a general ARC solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping

try:
    from experiments.candidate_records import CandidateRecord, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/motif_panel_grammar.py``
    from candidate_records import CandidateRecord, normalize_grid


Grid = tuple[tuple[int, ...], ...]


def _flip_h(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def _flip_v(grid: Grid) -> Grid:
    return tuple(reversed(grid))


def _rotate_180(grid: Grid) -> Grid:
    return _flip_v(_flip_h(grid))


def motif_transforms(grid: Grid) -> dict[str, Grid]:
    """Return all dimension-preserving D4 members for a motif."""

    result = {
        "id": grid,
        "flip_h": _flip_h(grid),
        "flip_v": _flip_v(grid),
        "rot180": _rotate_180(grid),
    }
    if len(grid) == len(grid[0]):
        result.update({
            "rot90": tuple(tuple(row) for row in zip(*grid[::-1])),
            "rot270": tuple(tuple(row) for row in zip(*grid))[::-1],
            "transpose": tuple(tuple(row) for row in zip(*grid)),
            "anti_transpose": tuple(
                tuple(reversed(row)) for row in zip(*grid)
            )[::-1],
        })
    # Retain all group names even when a particular motif has a stabilizer.
    # A transform name learned from an asymmetric demo must remain executable
    # on a symmetric demo; rendered equality, not name deletion, is the proper
    # quotient.  Output-class deduplication happens at CandidateRecord level.
    return result


def _panel_matrix(source: Grid, target: Grid) -> tuple[tuple[str, ...], ...] | None:
    height, width = len(source), len(source[0])
    if len(target) % height or len(target[0]) % width:
        return None
    row_factor, col_factor = len(target) // height, len(target[0]) // width
    transforms = motif_transforms(source)
    names: list[tuple[str, ...]] = []
    for panel_row in range(row_factor):
        current: list[str] = []
        for panel_col in range(col_factor):
            block = tuple(
                tuple(target[panel_row * height + row][panel_col * width + col]
                      for col in range(width))
                for row in range(height)
            )
            name = next((name for name, value in transforms.items()
                         if value == block), None)
            if name is None:
                return None
            current.append(name)
        names.append(tuple(current))
    return tuple(names)


def _rule_names(
    matrix: tuple[tuple[str, ...], ...],
) -> tuple[str, ...]:
    """Enumerate bounded patterns over a transform alphabet."""

    rules: list[str] = []
    row_factor = len(matrix)
    col_factor = len(matrix[0]) if matrix else 0
    names = tuple(value for row in matrix for value in row)
    if row_factor == 0 or col_factor == 0:
        return ()
    if all(value == names[0] for value in names):
        rules.append(f"constant:{names[0]}")
    if col_factor == 1 or all(names[row * col_factor] == names[row * col_factor + col - 1]
                              for row in range(row_factor) for col in range(1, col_factor)):
        sequence = tuple(names[row * col_factor] for row in range(row_factor))
        rules.append("row_template:" + ",".join(sequence))
    if row_factor == 1 or all(names[col] == names[col_factor + col] if row_factor > 1 else True
                              for col in range(col_factor)):
        sequence = tuple(names[col] for col in range(col_factor))
        rules.append("col_template:" + ",".join(sequence))
    # Two-state parity rules capture the common alternating-panel construction
    # without enumerating arbitrary matrices.
    if row_factor >= 2 and all(
        names[row * col_factor + col] == names[(row % 2) * col_factor]
        for row in range(row_factor) for col in range(col_factor)
    ):
        rules.append("row_parity:" + ",".join(
            names[(row % 2) * col_factor] for row in range(2)
        ))
    if col_factor >= 2 and all(
        names[row * col_factor + col] == names[col % 2]
        for row in range(row_factor) for col in range(col_factor)
    ):
        rules.append("col_parity:" + ",".join(names[col % 2] for col in range(2)))
    if row_factor >= 2 and col_factor >= 2 and all(
        names[row * col_factor + col] == names[(row + col) % 2]
        for row in range(row_factor) for col in range(col_factor)
    ):
        rules.append("checker:" + ",".join(names[index] for index in range(2)))
    if row_factor <= 4 and col_factor <= 4:
        rules.append("matrix_template:" + "/".join(
            ",".join(value for value in row) for row in matrix
        ))
    return tuple(dict.fromkeys(rules))


def _parse_rule(rule: str) -> tuple[str, tuple[str, ...]]:
    kind, values = rule.split(":", 1)
    if kind == "matrix_template":
        return kind, tuple(tuple(row.split(",")) for row in values.split("/"))  # type: ignore[return-value]
    return kind, tuple(value for value in values.split(",") if value)


def _render(source: Grid, row_factor: int, col_factor: int, rule: str) -> Grid:
    kind, values = _parse_rule(rule)
    transforms = motif_transforms(source)
    transform_names = (
        tuple(value for row in values for value in row)
        if kind == "matrix_template" else values
    )
    if any(value not in transforms for value in transform_names):
        raise ValueError("rule transform is not dimension preserving")
    rows: list[tuple[int, ...]] = []
    for panel_row in range(row_factor):
        blocks: list[Grid] = []
        for panel_col in range(col_factor):
            if kind == "matrix_template":
                matrix = values
                name = matrix[panel_row][panel_col]  # type: ignore[index]
            elif kind == "constant":
                name = values[0]
            elif kind == "row_template":
                name = values[panel_row % len(values)]
            elif kind == "col_template":
                name = values[panel_col % len(values)]
            elif kind == "row_parity":
                name = values[panel_row % 2]
            elif kind == "col_parity":
                name = values[panel_col % 2]
            elif kind == "checker":
                name = values[(panel_row + panel_col) % 2]
            else:
                raise ValueError("unknown panel rule")
            blocks.append(transforms[name])
        for row in range(len(source)):
            rows.append(tuple(value for block in blocks for value in block[row]))
    return tuple(rows)


@dataclass(frozen=True)
class MotifPanelProgram:
    name: str
    row_factor: int
    col_factor: int
    rule: str
    mdl_length: float

    def apply(self, source: Any) -> Grid:
        return _render(normalize_grid(source), self.row_factor, self.col_factor, self.rule)


def fit_motif_panel_programs(
    train_pairs: list[tuple[Any, Any]],
    *,
    max_programs: int = 32,
) -> tuple[MotifPanelProgram, ...]:
    """Infer bounded panel rules that replay every training pair exactly."""

    if max_programs <= 0 or not train_pairs:
        return ()
    pairs = [(normalize_grid(source), normalize_grid(target))
             for source, target in train_pairs]
    first_source, first_target = pairs[0]
    if len(first_target) % len(first_source) or len(first_target[0]) % len(first_source[0]):
        return ()
    row_factor = len(first_target) // len(first_source)
    col_factor = len(first_target[0]) // len(first_source[0])
    first_matrix = _panel_matrix(first_source, first_target)
    if first_matrix is None:
        return ()
    flattened = tuple(value for row in first_matrix for value in row)
    rules = _rule_names(first_matrix)
    programs: list[MotifPanelProgram] = []
    for rule in rules:
        program = MotifPanelProgram(
            name=f"panel_{row_factor}x{col_factor}:{rule}",
            row_factor=row_factor,
            col_factor=col_factor,
            rule=rule,
            mdl_length=3 + len(rule.split(":")) + len(rule),
        )
        try:
            if all(program.apply(source) == target for source, target in pairs):
                programs.append(program)
        except (TypeError, ValueError, IndexError):
            continue
        if len(programs) >= max_programs:
            break
    return tuple(programs)


def build_motif_panel_records(
    challenges: Mapping[str, Mapping[str, Any]],
) -> tuple[list[CandidateRecord], dict[str, int]]:
    records: list[CandidateRecord] = []
    verified_tasks: dict[str, int] = {}
    for task_id, task in challenges.items():
        programs = fit_motif_panel_programs(
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
                        family="motif_panel",
                        candidate_id=f"{task_id}:{test_index}:{program.name}",
                        output=program.apply(item["input"]),
                        program_id=program.name,
                        mdl_length=program.mdl_length,
                        proof_status="demo_verified",
                    ))
                except (TypeError, ValueError, IndexError):
                    continue
    return records, verified_tasks


if __name__ == "__main__":
    program = fit_motif_panel_programs([
        ([[7, 9], [4, 3]], [[7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3],
                              [9, 7, 9, 7, 9, 7], [3, 4, 3, 4, 3, 4],
                              [7, 9, 7, 9, 7, 9], [4, 3, 4, 3, 4, 3]]),
    ])
    assert program and program[0].apply([[3, 2], [7, 8]])
    print("motif_panel_grammar selftest: PASS")
