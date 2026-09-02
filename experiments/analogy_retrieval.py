"""Label-safe analogical retrieval from labeled training demonstrations.

An entry is reusable only under a witnessed D8 geometric action and a
bijective color map.  It is an unverified proposal for the target task: the
source demonstration's exactness does not prove that the same relation applies
to a new task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

try:
    from experiments.candidate_records import CandidateRecord, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/analogy_retrieval.py``
    from candidate_records import CandidateRecord, normalize_grid


Grid = tuple[tuple[int, ...], ...]


def rotate_clockwise(grid: Grid) -> Grid:
    return tuple(tuple(row) for row in zip(*grid[::-1]))


def d8_transforms(value: Any) -> tuple[tuple[str, Grid], ...]:
    current = normalize_grid(value)
    result: list[tuple[str, Grid]] = []
    for index in range(4):
        result.append((f"rot{index * 90}", current))
        result.append((
            f"flip_rot{index * 90}",
            tuple(tuple(reversed(row)) for row in current),
        ))
        current = rotate_clockwise(current)
    return tuple(result)


def _fit_bijective_color_map(source: Grid, query: Grid) -> dict[int, int] | None:
    if (len(source), len(source[0])) != (len(query), len(query[0])):
        return None
    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for source_row, query_row in zip(source, query):
        for source_cell, query_cell in zip(source_row, query_row):
            old = mapping.get(source_cell)
            if old is not None and old != query_cell:
                return None
            old_reverse = reverse.get(query_cell)
            if old_reverse is not None and old_reverse != source_cell:
                return None
            mapping[source_cell] = query_cell
            reverse[query_cell] = source_cell
    return mapping if len(mapping) == len(reverse) else None


def _map_grid(grid: Grid, mapping: Mapping[int, int]) -> Grid | None:
    if any(cell not in mapping for row in grid for cell in row):
        return None
    return tuple(tuple(mapping[cell] for cell in row) for row in grid)


@dataclass(frozen=True)
class TrainingAnalogy:
    source_task_id: str
    demo_index: int
    source: Grid
    target: Grid


def build_training_analogy_library(
    challenges: Mapping[str, Mapping[str, Any]],
) -> tuple[TrainingAnalogy, ...]:
    entries: list[TrainingAnalogy] = []
    for task_id, task in challenges.items():
        for demo_index, pair in enumerate(task.get("train", [])):
            try:
                entries.append(TrainingAnalogy(
                    task_id, demo_index,
                    normalize_grid(pair["input"]),
                    normalize_grid(pair["output"]),
                ))
            except (TypeError, ValueError, IndexError):
                continue
    return tuple(entries)


def retrieve_analogy_outputs(
    query: Any,
    library: tuple[TrainingAnalogy, ...],
    *,
    max_outputs: int = 32,
) -> tuple[tuple[str, Grid], ...]:
    """Return distinct equivariant outputs for one unlabeled query grid."""

    if max_outputs < 0:
        raise ValueError("max_outputs must be non-negative")
    query_grid = normalize_grid(query)
    result: list[tuple[str, Grid]] = []
    seen: set[Grid] = set()
    for entry in library:
        for transform_name, source_view in d8_transforms(entry.source):
            color_map = _fit_bijective_color_map(source_view, query_grid)
            if color_map is None:
                continue
            target_view = dict(d8_transforms(entry.target)).get(transform_name)
            if target_view is None:
                continue
            output = _map_grid(target_view, color_map)
            if output is None or output in seen:
                continue
            seen.add(output)
            result.append((
                f"{entry.source_task_id}:{entry.demo_index}:{transform_name}",
                output,
            ))
            if len(result) >= max_outputs:
                return tuple(result)
    return tuple(result)


def build_analogy_records(
    challenges: Mapping[str, Mapping[str, Any]],
    library: tuple[TrainingAnalogy, ...],
    *,
    max_outputs: int = 32,
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for task_id, task in challenges.items():
        for test_index, item in enumerate(task.get("test", [])):
            for analogy_id, output in retrieve_analogy_outputs(
                item["input"], library, max_outputs=max_outputs
            ):
                records.append(CandidateRecord.from_output(
                    task_id=task_id,
                    test_index=test_index,
                    family="analogy",
                    candidate_id=f"{task_id}:{test_index}:{analogy_id}",
                    output=output,
                    program_id=analogy_id,
                    mdl_length=8.0,
                    proof_status="unverified",
                ))
    return records


if __name__ == "__main__":
    library = (TrainingAnalogy(
        "source", 0, ((0, 1), (0, 0)), ((0, 0), (1, 0))
    ),)
    assert retrieve_analogy_outputs([[0, 7], [0, 0]], library)
    print("analogy_retrieval selftest: PASS")
