"""Cost-ordered typed search for a small offline ARC DSL.

The engine is intentionally semantic-free: a primitive's executor is owned by
the caller.  Its job is to remove compositions that cannot type-check and to
enumerate the remaining symbolic paths in increasing description cost.  This
is the combinatorial control suggested by a deterministic C++ DSL, expressed
in a compact CPU-testable form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Primitive:
    name: str
    input_type: str
    output_type: str
    cost: int = 1


@dataclass(frozen=True)
class TypedProgram:
    steps: tuple[str, ...]
    output_type: str
    cost: int


def enumerate_typed_paths(
    primitives: Iterable[Primitive],
    *,
    start_type: str,
    goal_type: str,
    max_cost: int,
    max_steps: int = 8,
) -> tuple[TypedProgram, ...]:
    """Enumerate unique well-typed unary programs by increasing cost."""

    if max_cost < 0 or max_steps < 0:
        raise ValueError("search bounds must be non-negative")
    primitives = tuple(primitives)
    if any(primitive.cost <= 0 for primitive in primitives):
        raise ValueError("primitive costs must be positive")
    frontier: list[TypedProgram] = [TypedProgram((), start_type, 0)]
    found: dict[tuple[str, ...], TypedProgram] = {}
    for _ in range(max_steps + 1):
        next_frontier: list[TypedProgram] = []
        for program in frontier:
            if program.output_type == goal_type and program.steps:
                found.setdefault(program.steps, program)
            if len(program.steps) == max_steps:
                continue
            for primitive in primitives:
                if primitive.input_type != program.output_type:
                    continue
                cost = program.cost + primitive.cost
                if cost > max_cost:
                    continue
                next_frontier.append(TypedProgram(
                    program.steps + (primitive.name,),
                    primitive.output_type,
                    cost,
                ))
        frontier = next_frontier
    return tuple(sorted(found.values(), key=lambda program: (
        program.cost, len(program.steps), program.steps
    )))


if __name__ == "__main__":
    paths = enumerate_typed_paths(
        [Primitive("flip", "grid", "grid"), Primitive("objects", "grid", "objects")],
        start_type="grid", goal_type="grid", max_cost=2,
    )
    assert paths[0].steps == ("flip",)
    print("typed_search selftest: PASS", paths)
