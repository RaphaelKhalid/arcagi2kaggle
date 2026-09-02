"""Label-free metamorphic probes for demo-consistent ARC programs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable

try:
    from experiments.cegis_version_space import Executor, Program, freeze_output
    from experiments.object_deltas import normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/metamorphic_probes.py``
    from cegis_version_space import Executor, Program, freeze_output
    from object_deltas import normalize_grid


Grid = tuple[tuple[int, ...], ...]
GridTransform = Callable[[Grid], Grid]


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def rotate_clockwise(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid[::-1]))


def flip_horizontal(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def translate(grid: Grid, row_delta: int, col_delta: int) -> Grid:
    """Translate non-background cells, rejecting out-of-bounds probes."""

    height, width = len(grid), len(grid[0])
    bg = _background(grid)
    result = [[bg] * width for _ in range(height)]
    for row in range(height):
        for col in range(width):
            target = (row + row_delta, col + col_delta)
            if grid[row][col] == bg:
                continue
            if not (0 <= target[0] < height and 0 <= target[1] < width):
                raise ValueError("translation clips a non-background cell")
            result[target[0]][target[1]] = grid[row][col]
    return tuple(tuple(row) for row in result)


def rename_colors(grid: Grid, mapping: dict[int, int]) -> Grid:
    return tuple(tuple(mapping.get(cell, cell) for cell in row) for row in grid)


@dataclass(frozen=True)
class MetamorphicProbe:
    name: str
    input_transform: GridTransform
    output_transform: GridTransform
    justification: str


@dataclass(frozen=True)
class ProbeEvidence:
    probe_name: str
    checked: int
    passed: int
    justification: str

    @property
    def score(self) -> float:
        return self.passed / self.checked if self.checked else 0.0


def d8_probes() -> tuple[MetamorphicProbe, ...]:
    """Return non-identity D8 equivariance probes."""

    probes: list[MetamorphicProbe] = []
    for turns in range(4):
        def transform(grid: Grid, turns: int = turns) -> Grid:
            result = grid
            for _ in range(turns):
                result = rotate_clockwise(result)
            return result
        if turns:
            probes.append(MetamorphicProbe(
                f"rot{turns * 90}", transform, transform,
                "task permits a geometric nuisance-group probe",
            ))
    for turns in range(4):
        def transform(grid: Grid, turns: int = turns) -> Grid:
            result = grid
            for _ in range(turns):
                result = rotate_clockwise(result)
            return flip_horizontal(result)
        probes.append(MetamorphicProbe(
            f"flip_rot{turns * 90}", transform, transform,
            "task permits a geometric nuisance-group probe",
        ))
    return tuple(probes)


def translation_probes() -> tuple[MetamorphicProbe, ...]:
    probes = []
    for row_delta, col_delta, name in (
        (1, 0, "shift_down"), (-1, 0, "shift_up"),
        (0, 1, "shift_right"), (0, -1, "shift_left"),
    ):
        transform = lambda grid, r=row_delta, c=col_delta: translate(grid, r, c)
        probes.append(MetamorphicProbe(
            name, transform, transform,
            "all non-background objects remain on-canvas under translation",
        ))
    return tuple(probes)


def color_swap_probe(first: int, second: int) -> MetamorphicProbe:
    mapping = {first: second, second: first}
    transform = lambda grid: rename_colors(grid, mapping)
    return MetamorphicProbe(
        f"swap_color_{first}_{second}", transform, transform,
        "color-role identity is treated as a soft nuisance hypothesis",
    )


def evaluate_probe(
    program: Program,
    inputs: Iterable[Any],
    execute: Executor,
    probe: MetamorphicProbe,
) -> ProbeEvidence:
    """Measure equivariance without consulting demonstration outputs."""

    checked = passed = 0
    for input_value in inputs:
        grid = normalize_grid(input_value)
        try:
            original = freeze_output(execute(program, grid))
            transformed = freeze_output(execute(program, probe.input_transform(grid)))
            expected = freeze_output(probe.output_transform(normalize_grid(original)))
        except (TypeError, ValueError, IndexError):
            continue
        checked += 1
        passed += int(transformed == expected)
    return ProbeEvidence(probe.name, checked, passed, probe.justification)


def evaluate_probes(
    program: Program,
    inputs: Iterable[Any],
    execute: Executor,
    probes: Iterable[MetamorphicProbe],
) -> tuple[ProbeEvidence, ...]:
    """Evaluate a deterministic probe battery against demo inputs."""

    inputs = tuple(inputs)
    return tuple(evaluate_probe(program, inputs, execute, probe) for probe in probes)


if __name__ == "__main__":
    def execute(program: Program, grid: Grid) -> Grid:
        return grid

    evidence = evaluate_probes(
        Program("identity", "dsl"), [[[0, 1], [0, 0]]], execute, d8_probes()
    )
    assert all(item.score == 1.0 for item in evidence)
    print("metamorphic_probes selftest: PASS")
