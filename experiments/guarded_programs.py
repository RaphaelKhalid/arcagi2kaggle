"""Proof-carrying guarded composition for contextual ARC rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from experiments.object_deltas import normalize_grid


Grid = tuple[tuple[int, ...], ...]
Predicate = Callable[[Grid], bool]
Transform = Callable[[Grid], Grid]


@dataclass(frozen=True)
class GuardedBranch:
    guard_name: str
    predicate: Predicate
    action_name: str
    action: Transform
    mdl_length: float = 1.0


@dataclass(frozen=True)
class GuardedProgram:
    branches: tuple[GuardedBranch, ...]
    fallback_name: str
    fallback: Transform
    mdl_length: float = 1.0


@dataclass(frozen=True)
class GuardedProof:
    """Exact demo proof plus branch truth table."""

    truth_table: tuple[tuple[str, ...], ...]
    outputs: tuple[Grid, ...]

    @property
    def branch_is_exclusive(self) -> bool:
        return all(len(matches) <= 1 for matches in self.truth_table)


def execute_guarded_program(program: GuardedProgram, value: Any) -> Grid | None:
    """Execute only when zero or one branch matches the scene."""

    grid = normalize_grid(value)
    matches = [branch for branch in program.branches if branch.predicate(grid)]
    if len(matches) > 1:
        return None
    action = matches[0].action if matches else program.fallback
    try:
        return normalize_grid(action(grid))
    except (TypeError, ValueError, IndexError):
        return None


def verify_guarded_program(
    program: GuardedProgram,
    train_pairs: Iterable[tuple[Any, Any]],
) -> GuardedProof | None:
    """Require exact outputs and an unambiguous branch on every demo."""

    truth_table: list[tuple[str, ...]] = []
    outputs: list[Grid] = []
    for source, target in train_pairs:
        grid = normalize_grid(source)
        matches = tuple(
            f"{branch.guard_name}->{branch.action_name}"
            for branch in program.branches if branch.predicate(grid)
        )
        if len(matches) > 1:
            return None
        predicted = execute_guarded_program(program, grid)
        if predicted is None or predicted != normalize_grid(target):
            return None
        truth_table.append(matches or (f"fallback->{program.fallback_name}",))
        outputs.append(predicted)
    return GuardedProof(tuple(truth_table), tuple(outputs))


def compose_guarded_program(
    branches: Iterable[GuardedBranch],
    fallback_name: str,
    fallback: Transform,
    train_pairs: Iterable[tuple[Any, Any]],
) -> tuple[GuardedProgram, GuardedProof] | None:
    """Build and immediately prove a bounded guarded composition."""

    branches = tuple(branches)
    if not branches:
        raise ValueError("at least one guarded branch is required")
    if any(branch.mdl_length < 0 for branch in branches):
        raise ValueError("branch MDL lengths must be non-negative")
    program = GuardedProgram(branches, fallback_name, fallback)
    proof = verify_guarded_program(program, train_pairs)
    return None if proof is None else (program, proof)
