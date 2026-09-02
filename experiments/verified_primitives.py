"""Small demo-verified primitive compiler for a CPU-only ARC replay."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    from experiments.candidate_records import CandidateRecord, normalize_grid
    from experiments.replay_harness import replay_score
except ModuleNotFoundError:  # direct ``python experiments/verified_primitives.py``
    from candidate_records import CandidateRecord, normalize_grid
    from replay_harness import replay_score


Grid = tuple[tuple[int, ...], ...]
Transform = Callable[[Grid], Grid]


def rotate_clockwise(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid[::-1]))


def rotate(grid: Grid, turns: int) -> Grid:
    result = grid
    for _ in range(turns % 4):
        result = rotate_clockwise(result)
    return result


def flip_horizontal(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def transpose(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid))


def background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def crop_content(grid: Grid) -> Grid:
    bg = background(grid)
    cells = [(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row)
             if cell != bg]
    if not cells:
        return grid
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    return tuple(tuple(grid[r][c] for c in range(c0, c1 + 1))
                 for r in range(r0, r1 + 1))


def gravity(grid: Grid, direction: str) -> Grid:
    bg = background(grid)
    height, width = len(grid), len(grid[0])
    result = [[bg] * width for _ in range(height)]
    if direction in {"left", "right"}:
        for r, row in enumerate(grid):
            values = [cell for cell in row if cell != bg]
            start = 0 if direction == "left" else width - len(values)
            result[r][start:start + len(values)] = values
    elif direction in {"up", "down"}:
        for c in range(width):
            values = [grid[r][c] for r in range(height) if grid[r][c] != bg]
            start = 0 if direction == "up" else height - len(values)
            for offset, value in enumerate(values):
                result[start + offset][c] = value
    else:
        raise ValueError("unknown gravity direction")
    return tuple(tuple(row) for row in result)


def tile(grid: Grid, row_factor: int, col_factor: int) -> Grid:
    return tuple(
        tuple(grid[r][c] for _ in range(col_factor)
              for c in range(len(grid[0])))
        for _ in range(row_factor)
        for r in range(len(grid))
    )


def upscale(grid: Grid, row_factor: int, col_factor: int) -> Grid:
    return tuple(
        tuple(cell for cell in row for _ in range(col_factor))
        for row in grid for _ in range(row_factor)
    )


def downscale(grid: Grid, row_factor: int, col_factor: int) -> Grid:
    height, width = len(grid), len(grid[0])
    if height % row_factor or width % col_factor:
        raise ValueError("downscale factors must divide the grid")
    result = []
    for r in range(0, height, row_factor):
        row = []
        for c in range(0, width, col_factor):
            values = [grid[rr][cc]
                      for rr in range(r, r + row_factor)
                      for cc in range(c, c + col_factor)]
            row.append(Counter(values).most_common(1)[0][0])
        result.append(tuple(row))
    return tuple(result)


@dataclass(frozen=True)
class VerifiedPrimitive:
    name: str
    family: str
    transform: Transform
    mdl_length: float


def _fixed_primitives() -> list[VerifiedPrimitive]:
    result: list[VerifiedPrimitive] = []
    for turns in range(4):
        result.append(VerifiedPrimitive(
            f"rot{turns * 90}", "geometry",
            lambda grid, turns=turns: rotate(grid, turns), 1 + turns,
        ))
        result.append(VerifiedPrimitive(
            f"flip_rot{turns * 90}", "geometry",
            lambda grid, turns=turns: flip_horizontal(rotate(grid, turns)),
            2 + turns,
        ))
    result.extend([
        VerifiedPrimitive("transpose", "geometry", transpose, 2),
        VerifiedPrimitive("crop_content", "object", crop_content, 3),
    ])
    for direction in ("left", "right", "up", "down"):
        result.append(VerifiedPrimitive(
            f"gravity_{direction}", "dynamics",
            lambda grid, direction=direction: gravity(grid, direction), 3,
        ))
    for row_factor in range(2, 4):
        for col_factor in range(2, 4):
            result.append(VerifiedPrimitive(
                f"tile_{row_factor}x{col_factor}", "resize",
                lambda grid, rf=row_factor, cf=col_factor: tile(grid, rf, cf),
                4,
            ))
            result.append(VerifiedPrimitive(
                f"upscale_{row_factor}x{col_factor}", "resize",
                lambda grid, rf=row_factor, cf=col_factor: upscale(grid, rf, cf),
                4,
            ))
            result.append(VerifiedPrimitive(
                f"downscale_{row_factor}x{col_factor}", "resize",
                lambda grid, rf=row_factor, cf=col_factor: downscale(grid, rf, cf),
                4,
            ))
    return result


def _fit_color_map(pairs: list[tuple[Grid, Grid]]) -> VerifiedPrimitive | None:
    mapping: dict[int, int] = {}
    for source, target in pairs:
        if len(source) != len(target) or len(source[0]) != len(target[0]):
            return None
        for source_row, target_row in zip(source, target):
            for source_cell, target_cell in zip(source_row, target_row):
                old = mapping.get(source_cell)
                if old is not None and old != target_cell:
                    return None
                mapping[source_cell] = target_cell
    if not mapping:
        return None

    def apply(grid: Grid) -> Grid:
        if not set(cell for row in grid for cell in row) <= set(mapping):
            raise ValueError("test contains an unseen color")
        return tuple(tuple(mapping[cell] for cell in row) for row in grid)

    name = "map_" + "_".join(f"{source}{target}" for source, target
                             in sorted(mapping.items()))
    return VerifiedPrimitive(name, "palette", apply, 2 + len(mapping))


def fit_verified_primitives(
    train_pairs: list[tuple[Any, Any]],
) -> list[VerifiedPrimitive]:
    """Fit and demo-verify a finite primitive set."""

    pairs = [(normalize_grid(source), normalize_grid(target))
             for source, target in train_pairs]
    fitted = _fixed_primitives()
    color_map = _fit_color_map(pairs)
    if color_map is not None:
        fitted.append(color_map)
    verified: list[VerifiedPrimitive] = []
    for primitive in fitted:
        try:
            if all(primitive.transform(source) == target for source, target in pairs):
                verified.append(primitive)
        except (TypeError, ValueError, IndexError):
            continue
    return verified


def build_records(
    challenges: Mapping[str, Mapping[str, Any]],
) -> tuple[list[CandidateRecord], dict[str, int], dict[str, int]]:
    records: list[CandidateRecord] = []
    verified_tasks: dict[str, int] = {}
    for task_id, task in challenges.items():
        pairs = [(pair["input"], pair["output"]) for pair in task["train"]]
        primitives = fit_verified_primitives(pairs)
        if primitives:
            verified_tasks[task_id] = len(primitives)
        for test_index, test in enumerate(task["test"]):
            for primitive in primitives:
                try:
                    output = primitive.transform(normalize_grid(test["input"]))
                    records.append(CandidateRecord.from_output(
                        task_id=task_id,
                        test_index=test_index,
                        family=primitive.family,
                        candidate_id=f"{task_id}:{test_index}:{primitive.name}",
                        output=output,
                        program_id=primitive.name,
                        mdl_length=primitive.mdl_length,
                        proof_status="demo_verified",
                    ))
                except (TypeError, ValueError, IndexError):
                    continue
    return records, verified_tasks, {task_id: len(task["test"])
                                    for task_id, task in challenges.items()}


def oracle_candidate_recall(
    records: list[CandidateRecord],
    solutions: Mapping[str, list[Any]],
) -> tuple[int, int]:
    by_position: dict[tuple[str, int], set[str]] = {}
    for record in records:
        by_position.setdefault((record.task_id, record.test_index), set()).add(
            record.output_hash
        )
    covered = total = 0
    for task_id, outputs in solutions.items():
        for index, output in enumerate(outputs):
            total += 1
            truth = CandidateRecord.from_output(
                task_id=task_id, test_index=index, family="truth",
                candidate_id="truth", output=output,
            )
            if truth.output_hash in by_position.get((task_id, index), set()):
                covered += 1
    return covered, total


if __name__ == "__main__":
    pairs = [([[0, 1]], [[0, 2]]), ([[1, 0]], [[2, 0]])]
    fitted = fit_verified_primitives(pairs)
    assert any(primitive.family == "palette" for primitive in fitted)
    print("verified_primitives selftest: PASS", len(fitted))
