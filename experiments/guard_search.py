"""Bounded synthesis of one-guard contextual programs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable

from experiments.guarded_programs import (
    GuardedBranch,
    GuardedProgram,
    GuardedProof,
    verify_guarded_program,
)
from experiments.object_deltas import extract_objects, normalize_grid


@dataclass(frozen=True)
class NamedAction:
    name: str
    action: Any
    mdl_length: float = 1.0


@dataclass(frozen=True)
class NamedPredicate:
    name: str
    predicate: Any
    mdl_length: float = 1.0


def feature_predicates(inputs: Iterable[Any]) -> tuple[NamedPredicate, ...]:
    """Generate a deterministic, finite guard vocabulary from visible inputs."""

    grids = tuple(normalize_grid(value) for value in inputs)
    if not grids:
        return ()
    features: list[NamedPredicate] = []
    heights = sorted({len(grid) for grid in grids})
    widths = sorted({len(grid[0]) for grid in grids})
    object_counts = sorted({len(extract_objects(grid)) for grid in grids})
    colors = sorted({cell for grid in grids for row in grid for cell in row})
    for height in heights:
        features.append(NamedPredicate(
            f"height={height}", lambda grid, value=height: len(grid) == value,
        ))
    for width in widths:
        features.append(NamedPredicate(
            f"width={width}", lambda grid, value=width: len(grid[0]) == value,
        ))
    for count in object_counts:
        features.append(NamedPredicate(
            f"objects={count}",
            lambda grid, value=count: len(extract_objects(grid)) == value,
        ))
    for color in colors:
        features.append(NamedPredicate(
            f"contains={color}",
            lambda grid, value=color: any(
                cell == value for row in grid for cell in row
            ),
        ))
    return tuple(features)


def search_one_guard(
    train_pairs: Iterable[tuple[Any, Any]],
    predicates: Iterable[NamedPredicate],
    actions: Iterable[NamedAction],
    *,
    max_candidates: int = 512,
) -> tuple[tuple[GuardedProgram, GuardedProof], ...]:
    """Enumerate and exact-verify a bounded one-guard/fallback product."""

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    train_pairs = tuple(train_pairs)
    predicates = tuple(sorted(predicates, key=lambda item: item.name))
    actions = tuple(sorted(actions, key=lambda item: item.name))
    results: list[tuple[GuardedProgram, GuardedProof]] = []
    attempted = 0
    for predicate, action, fallback in product(predicates, actions, actions):
        attempted += 1
        if attempted > max_candidates:
            break
        branch = GuardedBranch(
            predicate.name, predicate.predicate, action.name, action.action,
            predicate.mdl_length + action.mdl_length,
        )
        program = GuardedProgram(
            (branch,), fallback.name, fallback.action,
            predicate.mdl_length + action.mdl_length + fallback.mdl_length,
        )
        proof = verify_guarded_program(program, train_pairs)
        if proof is not None:
            results.append((program, proof))
    return tuple(sorted(results, key=lambda item: (
        item[0].mdl_length,
        item[0].branches[0].guard_name,
        item[0].branches[0].action_name,
        item[0].fallback_name,
    )))
