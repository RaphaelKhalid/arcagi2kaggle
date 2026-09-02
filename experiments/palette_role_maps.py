"""Proof-gated, task-local recoloring conditioned on scene roles.

This is the smallest executable slice of H12 (latent color-role algebra).
It deliberately does not infer semantics from ARC color ids.  A role is a
color-blind structural signature of a 4-connected object.  A candidate is
accepted only when every demonstration has identical geometry and a unique
same-shape/same-anchor object match, and the resulting role-conditioned map
replays every demonstration exactly.

The experiment is diagnostic: a demo-verified role map is a candidate
generator, not a claim that the hidden test obeys an arbitrary extrapolation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import sys
from typing import Any, Mapping

try:
    from experiments.object_deltas import Object, extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/palette_role_maps.py``
    from object_deltas import Object, extract_objects, normalize_grid


Grid = tuple[tuple[int, ...], ...]
Role = tuple[Any, ...]


def _background(grid: Grid) -> int:
    return Counter(cell for row in grid for cell in row).most_common(1)[0][0]


def _bucket(value: int) -> int:
    return min(3, value)


def role_key(grid: Any, obj: Object, level: int = 0) -> Role:
    """Return a deterministic color-blind structural role signature.

    Level 0 is most specific.  Higher levels quotient away shape details,
    allowing a role such as "small object on the top border" to transfer
    across demonstrations with different exact silhouettes.
    """

    value = normalize_grid(grid)
    height, width = len(value), len(value[0])
    rows = [row for row, _, _ in obj.cells]
    cols = [col for _, col, _ in obj.cells]
    bbox = (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
    objects = extract_objects(value)
    left = right = above = below = same_row = same_col = 0
    for other in objects:
        if other == obj:
            continue
        dr = other.anchor[0] - obj.anchor[0]
        dc = other.anchor[1] - obj.anchor[1]
        left += dc < 0
        right += dc > 0
        above += dr < 0
        below += dr > 0
        same_row += dr == 0
        same_col += dc == 0
    border = (
        obj.anchor[0] == 0,
        obj.anchor[1] == 0,
        max(rows) == height - 1,
        max(cols) == width - 1,
    )
    context = (
        _bucket(len(obj.cells)),
        _bucket(bbox[0]),
        _bucket(bbox[1]),
        border,
        _bucket(left), _bucket(right), _bucket(above), _bucket(below),
        _bucket(same_row), _bucket(same_col),
        _bucket(len(objects) - 1),
    )
    if level == 0:
        return ("shape", obj.shape, "context", context)
    if level == 1:
        return ("context", context)
    if level == 2:
        # A still coarser role: size and boundary status, but not exact
        # directional multiplicities.  It is useful for cross-demo transfer
        # but is intentionally exposed as a separate hypothesis family.
        return ("coarse", context[0], context[3], context[-1])
    raise ValueError("role level must be 0, 1, or 2")


def _unique_pairs(source: Grid, target: Grid) -> tuple[tuple[Object, Object], ...] | None:
    """Match objects only when shape and anchor identify one target object."""

    if (len(source), len(source[0])) != (len(target), len(target[0])):
        return None
    source_objects = extract_objects(source)
    target_objects = extract_objects(target)
    result: list[tuple[Object, Object]] = []
    used: set[int] = set()
    for left in source_objects:
        matches = [
            (index, right) for index, right in enumerate(target_objects)
            if index not in used
            and right.shape == left.shape
            and right.anchor == left.anchor
        ]
        if len(matches) != 1:
            return None
        index, right = matches[0]
        used.add(index)
        result.append((left, right))
    if len(result) != len(target_objects):
        return None
    return tuple(result)


@dataclass(frozen=True)
class RoleMap:
    level: int
    mapping: tuple[tuple[int, Role, int], ...]
    source_background: int
    target_background: int

    @property
    def name(self) -> str:
        return f"role_recolor_l{self.level}"

    @property
    def conditioned(self) -> bool:
        """True iff the same source color has role-dependent outputs."""

        by_color: dict[int, set[int]] = {}
        for source_color, _, target_color in self.mapping:
            by_color.setdefault(source_color, set()).add(target_color)
        return any(len(targets) > 1 for targets in by_color.values())

    def apply(self, grid: Any) -> Grid:
        source = normalize_grid(grid)
        if _background(source) != self.source_background:
            # Background is a structural convention, not an arbitrary color
            # id.  Requiring it avoids silently applying a map to a different
            # scene interpretation.
            raise ValueError("test background differs from fitted role map")
        result = [list(row) for row in source]
        for obj in extract_objects(source):
            role = role_key(source, obj, self.level)
            for row, col, color in obj.cells:
                candidates = [
                    target for source_color, mapped_role, target in self.mapping
                    if source_color == color and mapped_role == role
                ]
                if len(candidates) != 1:
                    raise ValueError("test has an unseen or ambiguous role")
                result[row][col] = candidates[0]
        return tuple(tuple(row) for row in result)

    def apply_partial(self, grid: Any) -> Grid:
        """Apply known role rules and preserve unproven roles by identity.

        This is a proposal-mode operation.  It is intentionally separate from
        ``apply``: exact proof requires complete role coverage, while a
        candidate generator may leave an unseen object untouched and let a
        different family propose its change.
        """

        source = normalize_grid(grid)
        if _background(source) != self.source_background:
            raise ValueError("test background differs from fitted role map")
        result = [list(row) for row in source]
        for obj in extract_objects(source):
            role = role_key(source, obj, self.level)
            for row, col, color in obj.cells:
                candidates = [
                    target for source_color, mapped_role, target in self.mapping
                    if source_color == color and mapped_role == role
                ]
                if len(candidates) == 1:
                    result[row][col] = candidates[0]
        return tuple(tuple(row) for row in result)


def fit_role_map(
    pairs: list[tuple[Any, Any]],
    *,
    level: int = 0,
    require_conditioned: bool = False,
) -> RoleMap | None:
    """Fit an exact role-conditioned map, or abstain."""

    if not pairs:
        return None
    mapping: dict[tuple[int, Role], int] = {}
    source_background: int | None = None
    target_background: int | None = None
    for source_value, target_value in pairs:
        source, target = normalize_grid(source_value), normalize_grid(target_value)
        if source_background is None:
            source_background, target_background = _background(source), _background(target)
        if _background(source) != source_background or _background(target) != target_background:
            return None
        matched = _unique_pairs(source, target)
        if matched is None:
            return None
        for left, right in matched:
            role = role_key(source, left, level)
            for (row, col, source_color), (_, _, target_color) in zip(
                left.cells, right.cells
            ):
                key = (source_color, role)
                old = mapping.get(key)
                if old is not None and old != target_color:
                    return None
                mapping[key] = target_color
        # The source background is not an object.  It may change only through
        # a separately proved global rule; this slice keeps role maps object
        # conditioned and therefore requires its identity here.
        if source_background != target_background:
            return None
    candidate = RoleMap(
        level, tuple((color, role, target) for (color, role), target in sorted(
            mapping.items(), key=lambda item: (item[0][0], repr(item[0][1]))
        )), source_background, target_background,
    )
    if require_conditioned and not candidate.conditioned:
        return None
    try:
        if not all(candidate.apply(source) == normalize_grid(target)
                   for source, target in pairs):
            return None
    except (TypeError, ValueError, IndexError):
        return None
    return candidate


def fit_role_maps(pairs: list[tuple[Any, Any]]) -> tuple[RoleMap, ...]:
    """Return role-map abstractions that pass all demos."""

    return tuple(
        candidate for level in range(3)
        for candidate in (fit_role_map(pairs, level=level, require_conditioned=True),)
        if candidate is not None
    )


def dataset_profile(challenges: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    summary = {
        "tasks": 0, "level0": 0, "level1": 0, "level2": 0,
        "any": 0, "conditioned": 0,
    }
    for task in challenges.values():
        summary["tasks"] += 1
        maps = fit_role_maps([(p["input"], p["output"]) for p in task.get("train", [])])
        for candidate in maps:
            summary[f"level{candidate.level}"] += 1
        if maps:
            summary["any"] += 1
            summary["conditioned"] += int(any(candidate.conditioned for candidate in maps))
    return summary


def candidate_recall(
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, list[Any]],
) -> tuple[int, int, int]:
    """Return (covered outputs, total outputs, partial candidates emitted)."""

    covered = total = executable = 0
    for task_id, task in challenges.items():
        maps = fit_role_maps([(p["input"], p["output"]) for p in task.get("train", [])])
        for index, item in enumerate(task.get("test", [])):
            total += 1
            for candidate in maps:
                try:
                    output = candidate.apply_partial(item["input"])
                except (TypeError, ValueError, IndexError):
                    continue
                executable += 1
                if output == normalize_grid(solutions[task_id][index]):
                    covered += 1
                    break
    return covered, total, executable


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("role-map selftest: pass (use a challenges JSON path for profiling)")
        raise SystemExit(0)
    root = sys.argv[1]
    with open(root, "r", encoding="utf-8") as handle:
        challenges = json.load(handle)
    print(dataset_profile(challenges))
