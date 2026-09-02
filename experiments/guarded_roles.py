"""Unique-role proof gates for graph-LGG action schemas."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from experiments.graph_lgg import (
        ActionSchema,
        lgg_observations,
        observations_for_correspondence,
        role_predicates,
    )
    from experiments.object_correspondence import top_k_correspondences
    from experiments.object_deltas import Connectivity, extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/guarded_roles.py``
    from graph_lgg import (
        ActionSchema,
        lgg_observations,
        observations_for_correspondence,
        role_predicates,
    )
    from object_correspondence import top_k_correspondences
    from object_deltas import Connectivity, extract_objects, normalize_grid


@dataclass(frozen=True)
class RoleSelection:
    source_indices: tuple[int, ...]
    target_indices: tuple[int, ...]


def select_roles(
    value: Any,
    guard: frozenset[tuple[Any, ...]],
    *,
    connectivity: Connectivity = 4,
) -> tuple[int, ...]:
    """Select all objects whose id-free predicates contain ``guard``."""

    objects = extract_objects(normalize_grid(value), connectivity=connectivity)
    return tuple(
        index for index in range(len(objects))
        if guard.issubset(role_predicates(value, index, connectivity=connectivity))
    )


def select_schema_roles(
    source_grid: Any,
    target_grid: Any,
    schema: ActionSchema,
    *,
    connectivity: Connectivity = 4,
) -> RoleSelection:
    return RoleSelection(
        source_indices=select_roles(
            source_grid, schema.source_guard, connectivity=connectivity
        ),
        target_indices=select_roles(
            target_grid, schema.target_guard, connectivity=connectivity
        ),
    )


def schema_has_unique_roles(
    source_grid: Any,
    target_grid: Any,
    schema: ActionSchema,
    *,
    connectivity: Connectivity = 4,
) -> bool:
    """Proof gate for a single action; empty guards are deliberately unsafe."""

    if not schema.source_guard or not schema.target_guard:
        return False
    selection = select_schema_roles(
        source_grid, target_grid, schema, connectivity=connectivity
    )
    return len(selection.source_indices) == 1 and len(selection.target_indices) == 1


def top1_lgg_for_task(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    connectivity: Connectivity = 4,
) -> tuple[tuple[ActionSchema, ...], tuple[RoleSelection, ...]] | None:
    """Build a top-1 LGG and its per-demo selections, if bounded."""

    traces = []
    pairs = task.get("train", [])
    if not pairs:
        return None
    try:
        for pair in pairs:
            candidates = top_k_correspondences(
                pair["input"], pair["output"], k=k,
                max_objects=max_objects, connectivity=connectivity
            )
            if not candidates:
                return None
            traces.append(observations_for_correspondence(
                pair["input"], pair["output"], candidates[0],
                connectivity=connectivity,
            ))
    except ValueError:
        return None
    schemas = lgg_observations(traces)
    if schemas is None:
        return None
    selections: list[RoleSelection] = []
    for pair in pairs:
        # The schema is aligned with the canonical sorted observation order;
        # this audit is intentionally restricted to one-action traces.
        if len(schemas) != 1:
            return schemas, ()
        selections.append(select_schema_roles(
            pair["input"], pair["output"], schemas[0],
            connectivity=connectivity,
        ))
    return schemas, tuple(selections)


def dataset_guard_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
    connectivity: Connectivity = 4,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "bounded_lgg_tasks": 0,
        "single_action_lgg_tasks": 0,
        "unique_role_tasks": 0,
        "ambiguous_or_empty_role_tasks": 0,
        "cap_or_no_lgg_tasks": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        result = top1_lgg_for_task(
            task, k=k, max_objects=max_objects, connectivity=connectivity
        )
        if result is None:
            summary["cap_or_no_lgg_tasks"] += 1
            continue
        schemas, selections = result
        summary["bounded_lgg_tasks"] += 1
        if len(schemas) != 1 or not selections:
            continue
        summary["single_action_lgg_tasks"] += 1
        unique = all(
            len(selection.source_indices) == 1
            and len(selection.target_indices) == 1
            and schemas[0].source_guard
            and schemas[0].target_guard
            for selection in selections
        )
        if unique:
            summary["unique_role_tasks"] += 1
        else:
            summary["ambiguous_or_empty_role_tasks"] += 1
    return dict(summary)


if __name__ == "__main__":
    source = [[0, 1], [0, 0]]
    target = [[0, 2], [0, 0]]
    result = top1_lgg_for_task({
        "train": [{"input": source, "output": target}]
    })
    assert result is not None
    print("guarded_roles selftest: PASS", result[1][0])
