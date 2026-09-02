"""Bounded local cellular transducers for complete-grid ARC proposals.

The program class maps a finite input neighborhood (including explicit
boundary sentinels) to one output color.  It is a typed ``grid -> grid``
latent rule, verified on every demonstration and fail-closed on unseen test
contexts.  The module is intentionally small: it measures local-rule recall
before any attempt to enlarge the grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

try:
    from experiments.candidate_records import CandidateRecord, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/cellular_transducer.py``
    from candidate_records import CandidateRecord, normalize_grid


Grid = tuple[tuple[int, ...], ...]
Context = tuple[int, ...]
BOUNDARY = -1
Fallback = Literal["identity", "background", "majority_output"]


def _offsets(radius: int, neighborhood: str) -> tuple[tuple[int, int], ...]:
    if radius < 0 or neighborhood not in {"cross", "square"}:
        raise ValueError("invalid neighborhood")
    if neighborhood == "cross":
        offsets = [(0, 0)]
        offsets.extend((step, 0) for step in range(1, radius + 1))
        offsets.extend((-step, 0) for step in range(1, radius + 1))
        offsets.extend((0, step) for step in range(1, radius + 1))
        offsets.extend((0, -step) for step in range(1, radius + 1))
    else:
        offsets = [
            (row, col)
            for row in range(-radius, radius + 1)
            for col in range(-radius, radius + 1)
        ]
    return tuple(offsets)


def context_at(
    grid: Grid,
    row: int,
    col: int,
    *,
    offsets: tuple[tuple[int, int], ...],
) -> Context:
    height, width = len(grid), len(grid[0])
    return tuple(
        grid[row + dr][col + dc]
        if 0 <= row + dr < height and 0 <= col + dc < width
        else BOUNDARY
        for dr, dc in offsets
    )


def _canonical_palette(grid: Grid) -> tuple[Grid, tuple[int, ...]]:
    """Canonicalize colors by background, frequency, first position, color."""

    counts: dict[int, int] = {}
    first: dict[int, tuple[int, int]] = {}
    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            counts[value] = counts.get(value, 0) + 1
            first.setdefault(value, (row, col))
    ordered = sorted(counts, key=lambda value: (
        -counts[value], first[value], value
    ))
    actual_by_role = tuple(ordered)
    role_by_actual = {value: role for role, value in enumerate(actual_by_role)}
    canonical = tuple(
        tuple(role_by_actual[value] for value in row) for row in grid
    )
    return canonical, actual_by_role


def _background_color(grid: Grid) -> int:
    counts: dict[int, int] = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    return min(counts, key=lambda value: (-counts[value], value))


def _majority(values: tuple[int, ...]) -> int:
    if not values:
        raise ValueError("cannot choose a fallback from an empty rule")
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return min(counts, key=lambda value: (-counts[value], value))


@dataclass(frozen=True)
class CellularProgram:
    neighborhood: str
    radius: int
    rule: tuple[tuple[Context, int], ...]
    color_mode: str = "raw"

    @property
    def name(self) -> str:
        suffix = "_roles" if self.color_mode == "role" else ""
        return f"cellular_{self.neighborhood}_r{self.radius}{suffix}"

    @property
    def mdl_length(self) -> float:
        return 3.0 + 0.25 * len(self.rule)

    def apply(self, value: Any) -> Grid | None:
        grid = normalize_grid(value)
        actual_by_role: tuple[int, ...] | None = None
        if self.color_mode == "role":
            grid, actual_by_role = _canonical_palette(grid)
        elif self.color_mode != "raw":
            raise ValueError("unknown color mode")
        offsets = _offsets(self.radius, self.neighborhood)
        lookup = dict(self.rule)
        result: list[tuple[int, ...]] = []
        for row in range(len(grid)):
            current: list[int] = []
            for col in range(len(grid[0])):
                context = context_at(grid, row, col, offsets=offsets)
                if context not in lookup:
                    return None
                role_or_color = lookup[context]
                if actual_by_role is None:
                    current.append(role_or_color)
                elif not 0 <= role_or_color < len(actual_by_role):
                    return None
                else:
                    current.append(actual_by_role[role_or_color])
            result.append(tuple(current))
        return tuple(result)

    def apply_with_fallback(self, value: Any, fallback: Fallback) -> Grid:
        """Totalize unseen contexts with one explicitly named default.

        The learned table remains unchanged and demo verification is still a
        hard prerequisite.  This method is an ablation for test-time coverage:
        it makes the uncertainty from unseen contexts visible instead of
        silently fabricating a new local rule.
        """

        grid = normalize_grid(value)
        actual_by_role: tuple[int, ...] | None = None
        if self.color_mode == "role":
            grid, actual_by_role = _canonical_palette(grid)
        elif self.color_mode != "raw":
            raise ValueError("unknown color mode")
        offsets = _offsets(self.radius, self.neighborhood)
        lookup = dict(self.rule)
        if fallback == "identity":
            default = None
        elif fallback == "background":
            default = 0 if actual_by_role is not None else _background_color(grid)
        elif fallback == "majority_output":
            default = _majority(tuple(lookup.values()))
        else:
            raise ValueError("unknown fallback")
        result: list[tuple[int, ...]] = []
        for row in range(len(grid)):
            current: list[int] = []
            for col in range(len(grid[0])):
                context = context_at(grid, row, col, offsets=offsets)
                role_or_color = lookup.get(context, default)
                if role_or_color is None:
                    role_or_color = grid[row][col]
                if actual_by_role is None:
                    current.append(role_or_color)
                elif not 0 <= role_or_color < len(actual_by_role):
                    raise ValueError("fallback role outside input palette")
                else:
                    current.append(actual_by_role[role_or_color])
            result.append(tuple(current))
        return tuple(result)


def fit_cellular_program(
    train_pairs: list[tuple[Any, Any]],
    *,
    radius: int = 1,
    neighborhood: str = "cross",
    color_mode: str = "raw",
) -> CellularProgram | None:
    """Fit one fail-closed context rule and verify all demonstrations."""

    if not train_pairs:
        return None
    offsets = _offsets(radius, neighborhood)
    mapping: dict[Context, int] = {}
    for source_value, target_value in train_pairs:
        source, target = normalize_grid(source_value), normalize_grid(target_value)
        if (len(source), len(source[0])) != (len(target), len(target[0])):
            return None
        if color_mode == "role":
            source, actual_by_role = _canonical_palette(source)
            target_roles = {value: role for role, value in enumerate(actual_by_role)}
            if any(value not in target_roles for row in target for value in row):
                return None
            target = tuple(
                tuple(target_roles[value] for value in row) for row in target
            )
        elif color_mode != "raw":
            raise ValueError("unknown color mode")
        for row in range(len(source)):
            for col in range(len(source[0])):
                context = context_at(source, row, col, offsets=offsets)
                output = target[row][col]
                old = mapping.get(context)
                if old is not None and old != output:
                    return None
                mapping[context] = output
    program = CellularProgram(
        neighborhood=neighborhood,
        radius=radius,
        rule=tuple(sorted(mapping.items(), key=repr)),
        color_mode=color_mode,
    )
    try:
        if not all(program.apply(source) == normalize_grid(target)
                   for source, target in train_pairs):
            return None
    except (TypeError, ValueError, IndexError):
        return None
    return program


def fit_cellular_programs(
    train_pairs: list[tuple[Any, Any]],
    *,
    max_radius: int = 1,
) -> tuple[CellularProgram, ...]:
    """Return bounded cross/square local rules in deterministic order."""

    if max_radius < 0:
        raise ValueError("max_radius must be non-negative")
    programs: list[CellularProgram] = []
    seen: set[tuple[str, int, tuple[tuple[Context, int], ...]]] = set()
    for radius in range(max_radius + 1):
        for neighborhood in ("cross", "square"):
            for color_mode in ("raw", "role"):
                program = fit_cellular_program(
                    train_pairs, radius=radius, neighborhood=neighborhood,
                    color_mode=color_mode,
                )
                if program is None:
                    continue
                key = (program.color_mode, program.neighborhood,
                       program.radius, program.rule)
                if key in seen:
                    continue
                seen.add(key)
                programs.append(program)
    return tuple(programs)


def build_cellular_records(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    max_radius: int = 1,
) -> tuple[list[CandidateRecord], dict[str, int]]:
    records: list[CandidateRecord] = []
    verified_tasks: dict[str, int] = {}
    for task_id, task in challenges.items():
        programs = fit_cellular_programs(
            [(pair["input"], pair["output"]) for pair in task.get("train", [])],
            max_radius=max_radius,
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
                    family="cellular",
                    candidate_id=f"{task_id}:{test_index}:{program.name}",
                    output=output,
                    program_id=program.name,
                    mdl_length=program.mdl_length,
                    proof_status="demo_verified",
                ))
    return records, verified_tasks


def build_completed_cellular_records(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    max_radius: int = 1,
    fallbacks: tuple[Fallback, ...] = (
        "identity", "background", "majority_output"
    ),
) -> tuple[list[CandidateRecord], dict[str, int]]:
    """Build bounded totalized records from demo-verified local programs."""

    records: list[CandidateRecord] = []
    verified_tasks: dict[str, int] = {}
    for task_id, task in challenges.items():
        programs = fit_cellular_programs(
            [(pair["input"], pair["output"])
             for pair in task.get("train", [])],
            max_radius=max_radius,
        )
        if programs:
            verified_tasks[task_id] = len(programs)
        for test_index, item in enumerate(task.get("test", [])):
            for program in programs:
                for fallback in fallbacks:
                    try:
                        output = program.apply_with_fallback(
                            item["input"], fallback
                        )
                    except (TypeError, ValueError, IndexError):
                        continue
                    records.append(CandidateRecord.from_output(
                        task_id=task_id,
                        test_index=test_index,
                        family="cellular_completion",
                        candidate_id=(
                            f"{task_id}:{test_index}:{program.name}:"
                            f"{fallback}"
                        ),
                        output=output,
                        program_id=f"{program.name}:{fallback}",
                        mdl_length=program.mdl_length + 0.5,
                        proof_status="demo_verified",
                    ))
    return records, verified_tasks


if __name__ == "__main__":
    program = fit_cellular_program([
        ([[0, 1, 0]], [[0, 2, 0]]),
        ([[1, 0, 1]], [[2, 0, 2]]),
    ])
    assert program is not None and program.apply([[0, 1, 0]]) == ((0, 2, 0),)
    print("cellular_transducer selftest: PASS")
