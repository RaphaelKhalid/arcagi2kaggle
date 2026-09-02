"""Top-k global object correspondence with explicit unmatched objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from experiments.object_deltas import (
        Connectivity, Object, extract_objects, normalize_grid,
    )
except ModuleNotFoundError:  # direct ``python experiments/object_correspondence.py``
    from object_deltas import Connectivity, Object, extract_objects, normalize_grid


@dataclass(frozen=True)
class Correspondence:
    pairs: tuple[tuple[int, int], ...]
    unmatched_source: tuple[int, ...]
    unmatched_target: tuple[int, ...]
    cost: int


def _object_cost(source: Object, target: Object) -> int:
    """Cheap structural cost; lower is better, but never a hard decision."""

    cost = 0
    if source.shape != target.shape:
        cost += 4
    if source.colored_shape != target.colored_shape:
        cost += 2
    cost += min(4, abs(source.anchor[0] - target.anchor[0])
                + abs(source.anchor[1] - target.anchor[1]))
    return cost


def top_k_correspondences(
    source_grid: Any,
    target_grid: Any,
    *,
    k: int = 4,
    unmatched_penalty: int = 5,
    max_objects: int = 10,
    connectivity: Connectivity = 4,
) -> tuple[Correspondence, ...]:
    """Return up to ``k`` minimum-cost partial assignments.

    Dynamic programming over target bitmasks is exact for ``max_objects``.
    Every source object may be left unmatched; target objects not selected by a
    pair incur the same penalty.  The returned alternatives are intentionally
    not deduplicated by semantic label: ambiguity in the correspondence is
    evidence that later task context should remain in the version space.
    """

    if k <= 0 or unmatched_penalty < 0:
        raise ValueError("k must be positive and unmatched_penalty non-negative")
    source = extract_objects(normalize_grid(source_grid), connectivity=connectivity)
    target = extract_objects(normalize_grid(target_grid), connectivity=connectivity)
    if len(source) > max_objects or len(target) > max_objects:
        raise ValueError("object count exceeds exact correspondence limit")

    # State: (target_mask, partial pairs, unmatched source, partial cost).
    states: dict[int, list[tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]]] = {
        0: [(0, (), ())]
    }
    for source_index, source_object in enumerate(source):
        next_states: dict[int, list[tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]]] = {}
        for mask, candidates in states.items():
            for partial_cost, pairs, unmatched_source in candidates:
                next_states.setdefault(mask, []).append((
                    partial_cost + unmatched_penalty,
                    pairs,
                    unmatched_source + (source_index,),
                ))
                for target_index, target_object in enumerate(target):
                    if mask & (1 << target_index):
                        continue
                    new_mask = mask | (1 << target_index)
                    next_states.setdefault(new_mask, []).append((
                        partial_cost + _object_cost(source_object, target_object),
                        pairs + ((source_index, target_index),),
                        unmatched_source,
                    ))
        states = {
            mask: sorted(candidates, key=lambda item: (item[0], item[1], item[2]))[:k]
            for mask, candidates in next_states.items()
        }

    result: list[Correspondence] = []
    target_all = set(range(len(target)))
    for mask, candidates in states.items():
        for partial_cost, pairs, unmatched_source in candidates:
            used = {target_index for _, target_index in pairs}
            missing = tuple(sorted(target_all - used))
            result.append(Correspondence(
                pairs=pairs,
                unmatched_source=unmatched_source,
                unmatched_target=missing,
                cost=partial_cost + unmatched_penalty * len(missing),
            ))
    result.sort(key=lambda item: (item.cost, item.pairs,
                                  item.unmatched_source, item.unmatched_target))
    unique: list[Correspondence] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for correspondence in result:
        key = correspondence.pairs
        if key in seen:
            continue
        seen.add(key)
        unique.append(correspondence)
        if len(unique) == k:
            break
    return tuple(unique)


def correspondence_profile(
    source_grid: Any,
    target_grid: Any,
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, int | bool]:
    """Summarize assignment ambiguity without selecting a semantic rule."""

    alternatives = top_k_correspondences(
        source_grid, target_grid, k=k, max_objects=max_objects
    )
    if not alternatives:
        return {"skipped": False, "n_alternatives": 0, "best_cost": 0,
                "tied_best": False}
    best_cost = alternatives[0].cost
    return {
        "skipped": False,
        "n_alternatives": len(alternatives),
        "best_cost": best_cost,
        "tied_best": sum(item.cost == best_cost for item in alternatives) > 1,
    }


if __name__ == "__main__":
    alternatives = top_k_correspondences(
        [[0, 1, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0]],
    )
    assert alternatives and alternatives[0].pairs == ((0, 0),)
    print("object_correspondence selftest: PASS", alternatives[0])
