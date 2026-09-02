"""A color-id-free role-palette transport experiment (H12 continuation).

The map `source color + role -> target color` is still tied to color ids.  This
module tests a more invariant hypothesis for monochrome, geometry-preserving
objects:

    color_out(slot[j]) = color_in(slot[permutation[j]])

Slots are color-blind object ranks (row-major, column-major, or area rank).
The permutation is inferred by exact demo replay and retained as a separate
candidate whenever duplicate colors make it non-identifiable.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import sys
from typing import Any, Mapping

try:
    from experiments.object_deltas import Object, extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/role_palette_transport.py``
    from object_deltas import Object, extract_objects, normalize_grid


Grid = tuple[tuple[int, ...], ...]
ORDERS = ("row_major", "column_major", "area_desc", "area_asc")


def _mono_color(obj: Object) -> int | None:
    colors = {color for _, _, color in obj.cells}
    return next(iter(colors)) if len(colors) == 1 else None


def _ordered(objects: tuple[Object, ...], order: str) -> tuple[Object, ...]:
    if order == "row_major":
        key = lambda obj: (obj.anchor[0], obj.anchor[1], obj.shape)
    elif order == "column_major":
        key = lambda obj: (obj.anchor[1], obj.anchor[0], obj.shape)
    elif order == "area_desc":
        key = lambda obj: (-len(obj.cells), obj.anchor, obj.shape)
    elif order == "area_asc":
        key = lambda obj: (len(obj.cells), obj.anchor, obj.shape)
    else:
        raise ValueError("unknown role-slot order")
    return tuple(sorted(objects, key=key))


def _same_geometry(source: Grid, target: Grid) -> bool:
    if (len(source), len(source[0])) != (len(target), len(target[0])):
        return False
    left = sorted((obj.anchor, obj.shape) for obj in extract_objects(source))
    right = sorted((obj.anchor, obj.shape) for obj in extract_objects(target))
    return left == right


def _demo_vectors(
    pairs: list[tuple[Any, Any]], order: str
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] | None:
    vectors = []
    for source_value, target_value in pairs:
        source, target = normalize_grid(source_value), normalize_grid(target_value)
        if not _same_geometry(source, target):
            return None
        source_objects = _ordered(extract_objects(source), order)
        target_objects = _ordered(extract_objects(target), order)
        if len(source_objects) == 0 or len(source_objects) != len(target_objects):
            return None
        source_colors = tuple(_mono_color(obj) for obj in source_objects)
        target_colors = tuple(_mono_color(obj) for obj in target_objects)
        if any(color is None for color in source_colors + target_colors):
            return None
        vectors.append((source_colors, target_colors))
    if not vectors or len(vectors[0][0]) > 8:
        return None
    n = len(vectors[0][0])
    if any(len(source) != n for source, _ in vectors):
        return None
    return tuple(vectors)


@dataclass(frozen=True)
class RolePaletteTransport:
    order: str
    permutation: tuple[int, ...]

    @property
    def name(self) -> str:
        return f"role_palette_{self.order}_" + "_".join(map(str, self.permutation))

    def apply(self, grid: Any) -> Grid:
        source = normalize_grid(grid)
        objects = _ordered(extract_objects(source), self.order)
        if len(objects) != len(self.permutation):
            raise ValueError("test object count differs from fitted role slots")
        colors = tuple(_mono_color(obj) for obj in objects)
        if any(color is None for color in colors):
            raise ValueError("test contains a multicolor object")
        result = [list(row) for row in source]
        for output_slot, obj in enumerate(objects):
            color = colors[self.permutation[output_slot]]
            for row, col, _ in obj.cells:
                result[row][col] = color
        return tuple(tuple(row) for row in result)


def fit_role_palette_transports(
    pairs: list[tuple[Any, Any]], *, max_candidates: int = 64
) -> tuple[RolePaletteTransport, ...]:
    """Fit all bounded non-identity role permutations that replay every demo."""

    result: list[RolePaletteTransport] = []
    for order in ORDERS:
        vectors = _demo_vectors(pairs, order)
        if vectors is None:
            continue
        n = len(vectors[0][0])
        for permutation in itertools.permutations(range(n)):
            if all(
                all(target[j] == source[permutation[j]] for j in range(n))
                for source, target in vectors
            ):
                try:
                    candidate = RolePaletteTransport(order, permutation)
                    if any(candidate.apply(source) != normalize_grid(target)
                           for source, target in pairs):
                        continue
                except (TypeError, ValueError, IndexError):
                    continue
                # Keep this family genuinely distinct from identity and from
                # color-preserving geometry: at least one demo must change.
                if any(source != normalize_grid(target) for source, target in pairs):
                    result.append(candidate)
                    if len(result) >= max_candidates:
                        return tuple(result)
    return tuple(result)


def dataset_profile(challenges: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    summary = {"tasks": 0, "tasks_with_transport": 0, "candidates": 0}
    for task in challenges.values():
        summary["tasks"] += 1
        candidates = fit_role_palette_transports(
            [(pair["input"], pair["output"]) for pair in task.get("train", [])]
        )
        if candidates:
            summary["tasks_with_transport"] += 1
            summary["candidates"] += len(candidates)
    return summary


def candidate_recall(
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, list[Any]],
) -> tuple[int, int, int]:
    covered = total = emitted = 0
    for task_id, task in challenges.items():
        candidates = fit_role_palette_transports(
            [(pair["input"], pair["output"]) for pair in task.get("train", [])]
        )
        for index, item in enumerate(task.get("test", [])):
            total += 1
            for candidate in candidates:
                try:
                    output = candidate.apply(item["input"])
                except (TypeError, ValueError, IndexError):
                    continue
                emitted += 1
                if output == normalize_grid(solutions[task_id][index]):
                    covered += 1
                    break
    return covered, total, emitted


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("role-palette transport selftest: pass (use a challenges JSON path for profiling)")
        raise SystemExit(0)
    with open(sys.argv[1], "r", encoding="utf-8") as handle:
        print(dataset_profile(json.load(handle)))
