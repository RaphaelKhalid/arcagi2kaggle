"""Leakage-safe structural grouping for distribution-shift diagnostics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def _bucket(value: int, cutoffs: tuple[int, ...]) -> str:
    for cutoff in cutoffs:
        if value <= cutoff:
            return f"le_{cutoff}"
    return f"gt_{cutoffs[-1]}"


def _grid_features(grid: list[list[int]]) -> dict[str, int | bool]:
    height, width = len(grid), len(grid[0])
    palette = {cell for row in grid for cell in row}
    background = Counter(cell for row in grid for cell in row).most_common(1)[0][0]
    seen: set[tuple[int, int]] = set()
    components = 0
    for row in range(height):
        for col in range(width):
            if grid[row][col] == background or (row, col) in seen:
                continue
            components += 1
            stack = [(row, col)]
            seen.add((row, col))
            while stack:
                r, c = stack.pop()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= nr < height and 0 <= nc < width
                            and grid[nr][nc] != background
                            and (nr, nc) not in seen):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
    return {
        "area": height * width,
        "palette": len(palette),
        "components": components,
        "square": height == width,
    }


def task_features(task: Mapping[str, Any]) -> dict[str, str]:
    """Compute features using only challenge-visible grids."""

    train_inputs = [pair["input"] for pair in task.get("train", [])]
    test_inputs = [item["input"] for item in task.get("test", [])]
    all_test = [_grid_features(grid) for grid in test_inputs]
    all_train = [_grid_features(grid) for grid in train_inputs]
    test = all_test or [{"area": 0, "palette": 0, "components": 0, "square": False}]
    train = all_train or [{"area": 0, "palette": 0, "components": 0, "square": False}]
    return {
        "demos": _bucket(len(train_inputs), (2, 3, 5)),
        "tests": str(len(test_inputs)),
        "test_area": _bucket(max(int(item["area"]) for item in test), (25, 100, 400)),
        "train_area": _bucket(max(int(item["area"]) for item in train), (25, 100, 400)),
        "test_palette": _bucket(max(int(item["palette"]) for item in test), (2, 4, 6)),
        "test_components": _bucket(max(int(item["components"]) for item in test), (1, 3, 6)),
        "test_square": str(all(item["square"] for item in test)),
    }


def task_position_features(
    task: Mapping[str, Any], test_index: int
) -> dict[str, str]:
    """Add visible structural features for one concrete test input.

    The task-level profile remains included as context, while position-level
    buckets prevent heterogeneous test panels from being assigned one copied
    rate. Only the input grid at ``test_index`` is read; hidden outputs are
    never consulted.
    """

    tests = task.get("test", [])
    if not isinstance(test_index, int) or not 0 <= test_index < len(tests):
        raise ValueError("test_index must identify a visible test input")
    position = _grid_features(tests[test_index]["input"])
    features = task_features(task)
    features.update({
        "position_area": _bucket(int(position["area"]), (25, 100, 400)),
        "position_palette": _bucket(int(position["palette"]), (2, 4, 6)),
        "position_components": _bucket(int(position["components"]), (1, 3, 6)),
        "position_square": str(position["square"]),
    })
    return features


def feature_distributions(
    challenges: Mapping[str, Mapping[str, Any]],
) -> dict[str, Counter[str]]:
    distributions: dict[str, Counter[str]] = {}
    for task in challenges.values():
        for name, value in task_features(task).items():
            distributions.setdefault(name, Counter())[value] += 1
    return distributions


def total_variation(left: Counter[str], right: Counter[str]) -> float:
    """Total-variation distance between two finite empirical distributions."""

    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return 0.0 if not left_total and not right_total else 1.0
    keys = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in keys
    )


def distribution_shift(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    left_profiles = feature_distributions(left)
    right_profiles = feature_distributions(right)
    return {
        name: total_variation(left_profiles.get(name, Counter()),
                              right_profiles.get(name, Counter()))
        for name in sorted(set(left_profiles) | set(right_profiles))
    }


if __name__ == "__main__":
    demo = {"a": {"train": [{"input": [[0, 1]], "output": [[1]]}],
                   "test": [{"input": [[0, 1]]}]}}
    assert task_features(demo["a"])["tests"] == "1"
    assert total_variation(Counter({"x": 1}), Counter({"y": 1})) == 1.0
    print("structural_groups selftest: PASS", task_features(demo["a"]))
