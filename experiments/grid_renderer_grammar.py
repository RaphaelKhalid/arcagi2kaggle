"""Small complete-grid renderer grammar for representation experiments.

This module intentionally stays finite and CPU-only.  It tests whether a
renderer view (crop/compress/geometric/palette) can recover exact candidates
that the object-edit DSL cannot express.  It is not a claim of ARC coverage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    from experiments.candidate_records import CandidateRecord, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/grid_renderer_grammar.py``
    from candidate_records import CandidateRecord, normalize_grid


Grid = tuple[tuple[int, ...], ...]
Transform = Callable[[Grid], Grid]


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _crop(grid: Grid, rows: tuple[int, int], cols: tuple[int, int]) -> Grid:
    r0, r1 = rows
    c0, c1 = cols
    if r0 < 0 or c0 < 0 or r1 > len(grid) or c1 > len(grid[0]) or r0 >= r1 or c0 >= c1:
        raise ValueError("invalid crop")
    return tuple(tuple(grid[r][c] for c in range(c0, c1)) for r in range(r0, r1))


def crop_content(grid: Grid) -> Grid:
    bg = _background(grid)
    cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row)
             if value != bg]
    if not cells:
        return grid
    return _crop(
        grid,
        (min(r for r, _ in cells), max(r for r, _ in cells) + 1),
        (min(c for _, c in cells), max(c for _, c in cells) + 1),
    )


def trim_empty_border(grid: Grid) -> Grid:
    """Remove only outer rows/columns that are entirely background."""

    bg = _background(grid)
    top, bottom, left, right = 0, len(grid), 0, len(grid[0])
    while top < bottom and all(value == bg for value in grid[top]):
        top += 1
    while bottom > top and all(value == bg for value in grid[bottom - 1]):
        bottom -= 1
    while left < right and all(grid[row][left] == bg for row in range(top, bottom)):
        left += 1
    while right > left and all(grid[row][right - 1] == bg for row in range(top, bottom)):
        right -= 1
    return _crop(grid, (top, bottom), (left, right))


def remove_empty_rows(grid: Grid) -> Grid:
    bg = _background(grid)
    rows = tuple(row for row in grid if any(value != bg for value in row))
    return rows or grid


def remove_empty_cols(grid: Grid) -> Grid:
    bg = _background(grid)
    cols = tuple(col for col in range(len(grid[0]))
                 if any(grid[row][col] != bg for row in range(len(grid))))
    if not cols:
        return grid
    return tuple(tuple(row[col] for col in cols) for row in grid)


def dedupe_adjacent_rows(grid: Grid) -> Grid:
    result = [grid[0]]
    for row in grid[1:]:
        if row != result[-1]:
            result.append(row)
    return tuple(result)


def dedupe_adjacent_cols(grid: Grid) -> Grid:
    keep = [0]
    for col in range(1, len(grid[0])):
        if any(grid[row][col] != grid[row][keep[-1]] for row in range(len(grid))):
            keep.append(col)
    return tuple(tuple(row[col] for col in keep) for row in grid)


def rotate_clockwise(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid[::-1]))


def d8_transforms(grid: Grid) -> tuple[tuple[str, Transform], ...]:
    result: list[tuple[str, Transform]] = []
    current = grid
    for turns in range(4):
        result.append((f"rot{turns * 90}", lambda value, turns=turns: _rotate(value, turns)))
        result.append((f"flip_rot{turns * 90}", lambda value, turns=turns: _flip(_rotate(value, turns))))
        current = rotate_clockwise(current)
    return tuple(result)


def _rotate(grid: Grid, turns: int) -> Grid:
    result = grid
    for _ in range(turns % 4):
        result = rotate_clockwise(result)
    return result


def _flip(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def _fit_palette(pairs: list[tuple[Grid, Grid]]) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for source, target in pairs:
        if (len(source), len(source[0])) != (len(target), len(target[0])):
            return None
        for source_row, target_row in zip(source, target):
            for old, new in zip(source_row, target_row):
                if old in mapping and mapping[old] != new:
                    return None
                mapping[old] = new
    return mapping or None


def _map_palette(grid: Grid, mapping: Mapping[int, int]) -> Grid:
    if not set(value for row in grid for value in row) <= set(mapping):
        raise ValueError("unseen palette value")
    return tuple(tuple(mapping[value] for value in row) for row in grid)


@dataclass(frozen=True)
class RendererProgram:
    name: str
    transform: Transform
    mdl_length: float


def _base_programs() -> tuple[RendererProgram, ...]:
    programs = [
        RendererProgram("identity", lambda grid: grid, 1),
        RendererProgram("crop_content", crop_content, 2),
        RendererProgram("trim_empty_border", trim_empty_border, 2),
        RendererProgram("remove_empty_rows", remove_empty_rows, 2),
        RendererProgram("remove_empty_cols", remove_empty_cols, 2),
        RendererProgram("dedupe_adjacent_rows", dedupe_adjacent_rows, 2),
        RendererProgram("dedupe_adjacent_cols", dedupe_adjacent_cols, 2),
    ]
    programs.extend(
        RendererProgram(name, transform, 1 + (name.count("rot") > 0))
        for name, transform in d8_transforms(((0,),))
    )
    return tuple(programs)


def fit_renderer_programs(
    train_pairs: list[tuple[Any, Any]],
    *,
    max_programs: int = 64,
) -> tuple[RendererProgram, ...]:
    """Fit a bounded renderer set and retain only exact demo replays."""

    if max_programs <= 0:
        return ()
    pairs = [(normalize_grid(source), normalize_grid(target))
             for source, target in train_pairs]
    if not pairs:
        return ()
    candidates = list(_base_programs())
    palette = _fit_palette(pairs)
    if palette is not None:
        candidates.append(RendererProgram(
            "palette_map", lambda grid, palette=palette: _map_palette(grid, palette),
            2 + len(palette),
        ))
    # Fit palette after each dynamic renderer.  This is a composition of two
    # semantic stages, not an open-ended search over arbitrary programs.
    for name, renderer in tuple((program.name, program.transform) for program in candidates):
        rendered_pairs: list[tuple[Grid, Grid]] = []
        try:
            for source, target in pairs:
                rendered_pairs.append((renderer(source), target))
        except (TypeError, ValueError, IndexError):
            continue
        composed_palette = _fit_palette(rendered_pairs)
        if composed_palette is None:
            continue
        candidates.append(RendererProgram(
            f"{name}|palette_map",
            lambda grid, renderer=renderer, mapping=composed_palette: _map_palette(renderer(grid), mapping),
            4 + len(composed_palette),
        ))
    verified: list[RendererProgram] = []
    seen: set[str] = set()
    for program in candidates:
        if program.name in seen:
            continue
        try:
            if all(program.transform(source) == target for source, target in pairs):
                seen.add(program.name)
                verified.append(program)
        except (TypeError, ValueError, IndexError):
            continue
        if len(verified) >= max_programs:
            break
    return tuple(verified)


def build_renderer_records(
    challenges: Mapping[str, Mapping[str, Any]],
) -> tuple[list[CandidateRecord], dict[str, int]]:
    records: list[CandidateRecord] = []
    verified_tasks: dict[str, int] = {}
    for task_id, task in challenges.items():
        programs = fit_renderer_programs(
            [(pair["input"], pair["output"]) for pair in task.get("train", [])]
        )
        if programs:
            verified_tasks[task_id] = len(programs)
        for test_index, test in enumerate(task.get("test", [])):
            for program in programs:
                try:
                    output = program.transform(normalize_grid(test["input"]))
                    records.append(CandidateRecord.from_output(
                        task_id=task_id,
                        test_index=test_index,
                        family="grid_renderer",
                        candidate_id=f"{task_id}:{test_index}:{program.name}",
                        output=output,
                        program_id=program.name,
                        mdl_length=program.mdl_length,
                        proof_status="demo_verified",
                    ))
                except (TypeError, ValueError, IndexError):
                    continue
    return records, verified_tasks


if __name__ == "__main__":
    pairs = [([[0, 1, 0]], [[2]])]
    assert fit_renderer_programs(pairs)
    print("grid_renderer_grammar selftest: PASS")
