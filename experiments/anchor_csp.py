"""Bounded constraint solver for relational target-anchor systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

try:
    from experiments.relation_equations import Relation, anchor_relation
except ModuleNotFoundError:  # direct ``python experiments/anchor_csp.py``
    from relation_equations import Relation, anchor_relation


Anchor = tuple[int, int]
Shape = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RelationConstraint:
    left: str
    right: str
    relation: Relation


@dataclass(frozen=True)
class AnchorSystem:
    grid_shape: tuple[int, int]
    shapes: Mapping[str, Shape]
    relations: tuple[RelationConstraint, ...] = ()
    blocked: frozenset[tuple[int, int]] = frozenset()
    fixed: Mapping[str, Anchor] = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AnchorSolveResult:
    solutions: tuple[Mapping[str, Anchor], ...]
    exhausted: bool


def _cells(anchor: Anchor, shape: Shape) -> frozenset[tuple[int, int]]:
    return frozenset((anchor[0] + row, anchor[1] + col) for row, col in shape)


def _domain(system: AnchorSystem, role: str) -> tuple[Anchor, ...]:
    height, width = system.grid_shape
    shape = system.shapes[role]
    fixed = dict(system.fixed or {})
    candidates = [fixed[role]] if role in fixed else [
        (row, col) for row in range(height) for col in range(width)
    ]
    return tuple(
        anchor for anchor in candidates
        if all(0 <= anchor[0] + row < height
               and 0 <= anchor[1] + col < width
               for row, col in shape)
        and not _cells(anchor, shape).intersection(system.blocked)
    )


def _consistent(
    system: AnchorSystem,
    assignment: Mapping[str, Anchor],
    role: str,
    anchor: Anchor,
) -> bool:
    candidate_cells = _cells(anchor, system.shapes[role])
    for other, other_anchor in assignment.items():
        if candidate_cells.intersection(_cells(other_anchor, system.shapes[other])):
            return False
    tentative = dict(assignment)
    tentative[role] = anchor
    for constraint in system.relations:
        if constraint.left in tentative and constraint.right in tentative:
            if anchor_relation(
                tentative[constraint.left], tentative[constraint.right]
            ) != constraint.relation:
                return False
    return True


def solve_anchor_system(
    system: AnchorSystem,
    *,
    max_solutions: int = 2,
    node_budget: int = 100_000,
) -> AnchorSolveResult:
    """Enumerate at most ``max_solutions`` satisfying assignments."""

    if max_solutions <= 0 or node_budget <= 0:
        raise ValueError("solution and node budgets must be positive")
    domains = {role: _domain(system, role) for role in system.shapes}
    if any(not domain for domain in domains.values()):
        return AnchorSolveResult((), False)
    solutions: list[Mapping[str, Anchor]] = []
    nodes = 0
    exhausted = False

    def search(assignment: dict[str, Anchor]) -> None:
        nonlocal nodes, exhausted
        if len(solutions) >= max_solutions or exhausted:
            return
        nodes += 1
        if nodes > node_budget:
            exhausted = True
            return
        if len(assignment) == len(domains):
            solutions.append(dict(assignment))
            return
        unassigned = [role for role in domains if role not in assignment]
        # MRV with deterministic role tie-break.
        feasible: list[tuple[int, str, tuple[Anchor, ...]]] = []
        for role in unassigned:
            candidates = tuple(
                anchor for anchor in domains[role]
                if _consistent(system, assignment, role, anchor)
            )
            if not candidates:
                return
            feasible.append((len(candidates), role, candidates))
        _, role, candidates = min(feasible, key=lambda item: (item[0], item[1]))
        for anchor in candidates:
            search({**assignment, role: anchor})

    search({})
    return AnchorSolveResult(tuple(solutions), exhausted)


if __name__ == "__main__":
    system = AnchorSystem(
        grid_shape=(1, 5),
        shapes={"A": ((0, 0),), "B": ((0, 0),)},
        relations=(RelationConstraint(
            "A", "B", ((0, -1), "horizontal", 1)
        ),),
        fixed={"B": (0, 4)},
    )
    result = solve_anchor_system(system)
    assert result.solutions == ({"A": (0, 3), "B": (0, 4)},)
    print("anchor_csp selftest: PASS")
