"""Object-level delta profiling for prioritizing relational DSL operators."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping


Grid = tuple[tuple[int, ...], ...]
Connectivity = Literal[4, 8]


def normalize_grid(value: Any) -> Grid:
    if hasattr(value, "tolist"):
        value = value.tolist()
    grid = tuple(tuple(int(cell) for cell in row) for row in value)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be rectangular and non-empty")
    return grid


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


@dataclass(frozen=True)
class Object:
    cells: tuple[tuple[int, int, int], ...]
    anchor: tuple[int, int]
    shape: tuple[tuple[int, int], ...]
    colored_shape: tuple[tuple[int, int, int], ...]


def extract_objects(
    value: Any, *, connectivity: Connectivity = 4
) -> tuple[Object, ...]:
    """Extract non-background components under an explicit adjacency model."""

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    grid = normalize_grid(value)
    height, width = len(grid), len(grid[0])
    bg = _background(grid)
    seen: set[tuple[int, int]] = set()
    objects: list[Object] = []
    for row in range(height):
        for col in range(width):
            if grid[row][col] == bg or (row, col) in seen:
                continue
            stack = [(row, col)]
            seen.add((row, col))
            cells: list[tuple[int, int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c, grid[r][c]))
                neighbors = ((r - 1, c), (r + 1, c),
                             (r, c - 1), (r, c + 1))
                if connectivity == 8:
                    neighbors = tuple(
                        (r + dr, c + dc)
                        for dr in (-1, 0, 1)
                        for dc in (-1, 0, 1)
                        if (dr, dc) != (0, 0)
                    )
                for nr, nc in neighbors:
                    if (0 <= nr < height and 0 <= nc < width
                            and grid[nr][nc] != bg
                            and (nr, nc) not in seen):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            anchor = (min(r for r, _, _ in cells), min(c for _, c, _ in cells))
            shape = tuple(sorted((r - anchor[0], c - anchor[1])
                                 for r, c, _ in cells))
            colored_shape = tuple(sorted((r - anchor[0], c - anchor[1], color)
                                         for r, c, color in cells))
            objects.append(Object(tuple(sorted(cells)), anchor, shape, colored_shape))
    return tuple(objects)


def _same_shape_pairs(source: tuple[Object, ...], target: tuple[Object, ...]) -> list[tuple[Object, Object]]:
    remaining = list(target)
    pairs: list[tuple[Object, Object]] = []
    for obj in source:
        matches = [candidate for candidate in remaining if candidate.shape == obj.shape]
        if matches:
            candidate = min(matches, key=lambda item: (item.anchor, item.colored_shape))
            remaining.remove(candidate)
            pairs.append((obj, candidate))
    return pairs


def classify_delta(source: Any, target: Any) -> frozenset[str]:
    """Return conservative object-level change labels for one demo pair."""

    source_grid, target_grid = normalize_grid(source), normalize_grid(target)
    if source_grid == target_grid:
        return frozenset({"identity"})
    labels: set[str] = set()
    if (len(source_grid), len(source_grid[0])) != (len(target_grid), len(target_grid[0])):
        labels.add("grid_resize")
    source_objects = extract_objects(source_grid)
    target_objects = extract_objects(target_grid)
    pairs = _same_shape_pairs(source_objects, target_objects)
    for left, right in pairs:
        if left.anchor != right.anchor and left.colored_shape == right.colored_shape:
            labels.add("object_move")
        elif left.anchor == right.anchor and left.colored_shape != right.colored_shape:
            labels.add("object_recolor")
        elif left.colored_shape != right.colored_shape:
            labels.add("object_transform")
    if len(target_objects) > len(pairs):
        labels.add("object_add")
    if len(source_objects) > len(pairs):
        labels.add("object_delete")
    if not labels:
        labels.add("cellwise_or_topology")
    return frozenset(labels)


def task_delta_profile(task: Mapping[str, Any]) -> Counter[str]:
    labels: Counter[str] = Counter()
    for pair in task.get("train", []):
        labels.update(classify_delta(pair["input"], pair["output"]))
    return labels


def dataset_delta_profile(
    challenges: Mapping[str, Mapping[str, Any]],
) -> dict[str, Counter[str]]:
    return {task_id: task_delta_profile(task) for task_id, task in challenges.items()}


def task_consistency_profile(
    challenges: Mapping[str, Mapping[str, Any]],
) -> Counter[str]:
    """Count tasks whose demos all contain each delta label."""

    result: Counter[str] = Counter()
    for task in challenges.values():
        demo_labels = [set(classify_delta(pair["input"], pair["output"]))
                       for pair in task.get("train", [])]
        if not demo_labels:
            continue
        for label in set.intersection(*demo_labels):
            result[label] += 1
    return result


if __name__ == "__main__":
    source = ((0, 1), (0, 0))
    target = ((0, 2), (0, 0))
    assert "object_recolor" in classify_delta(source, target)
    print("object_deltas selftest: PASS", classify_delta(source, target))
