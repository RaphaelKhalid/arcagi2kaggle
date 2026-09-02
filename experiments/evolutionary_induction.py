"""Proof-guided evolutionary search over a small symbolic ARC program space.

The operator is executor-agnostic.  It is meant to sit between a language
model that proposes an initial genome and the deterministic replay/CEGIS
layers: mutations expand coverage, while exact demo fitness and MDL prevent
the population from drifting toward merely plausible programs. Parents
compete with their mutants on a Pareto frontier; dominated parents are allowed
to disappear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from experiments.cegis_version_space import Demonstration, freeze_output


Executor = Callable[["Genome", Any], Any]


@dataclass(frozen=True)
class Genome:
    """Canonical symbolic program representation."""

    operations: tuple[str, ...]

    def __post_init__(self) -> None:
        if any(not operation for operation in self.operations):
            raise ValueError("operation names must be non-empty")


@dataclass(frozen=True)
class Fitness:
    genome: Genome
    correct_demos: int
    total_demos: int
    mdl_length: float
    cell_correct: int = 0
    cell_total: int = 0
    probe_signature: tuple[Any, ...] = ()

    @property
    def exact(self) -> bool:
        return self.total_demos > 0 and self.correct_demos == self.total_demos

    @property
    def cell_accuracy(self) -> float:
        """Secondary partial-grid signal for steering, never proof."""

        return self.cell_correct / self.cell_total if self.cell_total else 0.0


def partial_cell_match(predicted: Any, expected: Any) -> tuple[int, int]:
    """Count cell matches with shape mismatch penalized by the larger grid.

    This is deliberately a search heuristic. Exact demo equality remains the
    only proof gate, and non-grid outputs receive a one-bit exact score.
    """

    predicted = predicted.tolist() if hasattr(predicted, "tolist") else predicted
    expected = expected.tolist() if hasattr(expected, "tolist") else expected
    if not isinstance(predicted, (list, tuple)) or not isinstance(expected, (list, tuple)):
        return (int(freeze_output(predicted) == freeze_output(expected)), 1)
    if not predicted or not expected:
        return (int(freeze_output(predicted) == freeze_output(expected)), 1)
    predicted_rows = tuple(predicted)
    expected_rows = tuple(expected)
    if not all(isinstance(row, (list, tuple)) for row in predicted_rows + expected_rows):
        return (int(freeze_output(predicted) == freeze_output(expected)), 1)
    height = max(len(predicted_rows), len(expected_rows))
    width = max(
        max((len(row) for row in predicted_rows), default=0),
        max((len(row) for row in expected_rows), default=0),
    )
    correct = 0
    predicted_cells = sum(len(row) for row in predicted_rows)
    expected_cells = sum(len(row) for row in expected_rows)
    for row in range(height):
        for col in range(width):
            left = predicted_rows[row][col] if row < len(predicted_rows) and col < len(predicted_rows[row]) else None
            right = expected_rows[row][col] if row < len(expected_rows) and col < len(expected_rows[row]) else None
            if left is not None and right is not None:
                correct += int(left == right)
    return correct, max(predicted_cells, expected_cells, 1)


def mutate_genome(
    genome: Genome,
    operation_library: Iterable[str],
    *,
    max_steps: int = 8,
) -> tuple[Genome, ...]:
    """Return deterministic one-edit insert/delete/replace mutations."""

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if len(genome.operations) > max_steps:
        raise ValueError("genome already exceeds max_steps")
    library = tuple(sorted(set(operation_library)))
    if any(not operation for operation in library):
        raise ValueError("operation names must be non-empty")
    candidates: set[Genome] = set()
    for index, old in enumerate(genome.operations):
        if len(genome.operations) > 1:
            candidates.add(Genome(genome.operations[:index] + genome.operations[index + 1:]))
        for operation in library:
            if operation != old:
                candidates.add(Genome(
                    genome.operations[:index] + (operation,)
                    + genome.operations[index + 1:]
                ))
    if len(genome.operations) < max_steps:
        for index in range(len(genome.operations) + 1):
            for operation in library:
                candidates.add(Genome(
                    genome.operations[:index] + (operation,)
                    + genome.operations[index:]
                ))
    return tuple(sorted(candidates, key=lambda item: (len(item.operations), item.operations)))


def mutation_shell(
    genome: Genome,
    operation_library: Iterable[str],
    *,
    radius: int = 1,
    max_steps: int = 8,
    max_candidates: int = 4_096,
) -> tuple[Genome, ...]:
    """Enumerate a bounded breadth-first shell of one-edit mutations.

    ``radius`` allows an explicit escape from a local basin. The breadth-first
    order is canonical and the cap fails closed by retaining the earliest
    canonical candidates; callers must treat a capped shell as incomplete.
    """

    if radius < 0:
        raise ValueError("radius must be non-negative")
    if max_candidates < 0:
        raise ValueError("max_candidates must be non-negative")
    if radius == 0 or max_candidates == 0:
        return ()
    frontier = (genome,)
    seen = {genome}
    for _ in range(radius):
        next_candidates: set[Genome] = set()
        for parent in frontier:
            next_candidates.update(
                mutate_genome(parent, operation_library, max_steps=max_steps)
            )
        next_frontier = sorted(
            next_candidates - seen,
            key=lambda item: (len(item.operations), item.operations),
        )
        remaining = max_candidates - len(seen) + 1
        if len(next_frontier) > remaining:
            next_frontier = next_frontier[:max(0, remaining)]
        if not next_frontier:
            break
        seen.update(next_frontier)
        frontier = tuple(next_frontier)
        if len(seen) - 1 >= max_candidates:
            break
    return tuple(sorted(
        seen - {genome},
        key=lambda item: (len(item.operations), item.operations),
    ))


def guided_mutations(
    genome: Genome,
    divergence_index: int | None,
    operation_library: Iterable[str],
    *,
    max_steps: int = 8,
) -> tuple[Genome, ...]:
    """Restrict edits to the operation causing the first trace divergence."""

    all_mutations = mutate_genome(genome, operation_library, max_steps=max_steps)
    if divergence_index is None or divergence_index <= 0:
        return all_mutations
    target = divergence_index - 1
    if target >= len(genome.operations):
        return all_mutations

    library = tuple(sorted(set(operation_library)))
    local: set[Genome] = set()
    operations = genome.operations
    for operation in library:
        if operation != operations[target]:
            local.add(Genome(operations[:target] + (operation,) + operations[target + 1:]))
    if len(operations) > 1:
        local.add(Genome(operations[:target] + operations[target + 1:]))
    if len(operations) < max_steps:
        for operation in library:
            local.add(Genome(operations[:target] + (operation,) + operations[target:]))
            local.add(Genome(operations[:target + 1] + (operation,) + operations[target + 1:]))
    return tuple(sorted(local, key=lambda item: (len(item.operations), item.operations)))


def crossover_genomes(
    left: Genome,
    right: Genome,
    *,
    max_steps: int = 8,
) -> tuple[Genome, ...]:
    """Return bounded one-point recombinations of two parent genomes.

    Every cut pair is deterministic and both parent orderings are considered.
    Crossover is a proposal operator only; every child still requires exact
    demonstration replay before it can become a verified candidate.
    """

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if len(left.operations) > max_steps or len(right.operations) > max_steps:
        raise ValueError("parent genome exceeds max_steps")
    children: set[Genome] = set()
    for left_cut in range(len(left.operations) + 1):
        for right_cut in range(len(right.operations) + 1):
            for first, second in ((left, right), (right, left)):
                operations = (
                    first.operations[:left_cut] + second.operations[right_cut:]
                )
                if len(operations) <= max_steps:
                    children.add(Genome(operations))
    return tuple(sorted(children, key=lambda item: (len(item.operations), item.operations)))


def evaluate_genome(
    genome: Genome,
    demonstrations: Iterable[Demonstration],
    execute: Executor,
    *,
    mdl_costs: Mapping[str, float] | None = None,
    probe_inputs: Iterable[Any] = (),
) -> Fitness:
    """Score exact demonstration replay and symbolic description length."""

    demonstrations = tuple(demonstrations)
    costs = mdl_costs or {}
    if any(float(value) < 0.0 for value in costs.values()):
        raise ValueError("MDL costs must be non-negative")
    correct = 0
    cell_correct = cell_total = 0
    for demo in demonstrations:
        observed = execute(genome, demo.input)
        correct += int(freeze_output(observed) == freeze_output(demo.output))
        matched, total = partial_cell_match(observed, demo.output)
        cell_correct += matched
        cell_total += total
    mdl = sum(float(costs.get(operation, 1.0)) for operation in genome.operations)
    probe_signature = tuple(
        freeze_output(execute(genome, value)) for value in probe_inputs
    )
    return Fitness(
        genome, correct, len(demonstrations), mdl, cell_correct, cell_total,
        probe_signature,
    )


def pareto_frontier(fitnesses: Iterable[Fitness]) -> tuple[Fitness, ...]:
    """Keep non-dominated programs: maximize demo correctness, minimize MDL."""

    values = tuple(fitnesses)
    frontier = []
    for candidate in values:
        dominated = any(
            other.correct_demos >= candidate.correct_demos
            and other.cell_accuracy >= candidate.cell_accuracy
            and other.mdl_length <= candidate.mdl_length
            and (
                not other.probe_signature
                or not candidate.probe_signature
                or other.probe_signature == candidate.probe_signature
                or other.correct_demos > candidate.correct_demos
                or other.cell_accuracy > candidate.cell_accuracy
            )
            and (
                other.correct_demos > candidate.correct_demos
                or other.cell_accuracy > candidate.cell_accuracy
                or other.mdl_length < candidate.mdl_length
            )
            for other in values
            if other.genome != candidate.genome
        )
        if not dominated:
            frontier.append(candidate)
    return tuple(sorted(
        frontier,
        key=lambda item: (-item.correct_demos, item.mdl_length, item.genome.operations),
    ))


def bounded_frontier(
    fitnesses: Iterable[Fitness],
    *,
    max_items: int,
) -> tuple[Fitness, ...]:
    """Cap a frontier while preserving as much probe-output coverage as possible."""

    if max_items < 0:
        raise ValueError("max_items must be non-negative")
    frontier = list(pareto_frontier(fitnesses))
    if len(frontier) <= max_items:
        return tuple(frontier)
    if max_items == 0:
        return ()

    def coverage(item: Fitness) -> set[tuple[int, str]]:
        return {
            (index, repr(value))
            for index, value in enumerate(item.probe_signature)
        }

    selected: list[Fitness] = []
    remaining = set(frontier)
    covered: set[tuple[int, str]] = set()
    while remaining and len(selected) < max_items:
        chosen = max(
            remaining,
            key=lambda item: (
                len(coverage(item) - covered),
                item.correct_demos,
                item.cell_accuracy,
                -item.mdl_length,
                tuple(repr(value) for value in item.probe_signature),
                tuple(reversed(item.genome.operations)),
            ),
        )
        selected.append(chosen)
        covered.update(coverage(chosen))
        remaining.remove(chosen)
    return tuple(sorted(
        selected,
        key=lambda item: (-item.correct_demos, item.mdl_length, item.genome.operations),
    ))


def evolve_generation(
    population: Iterable[Genome],
    demonstrations: Iterable[Demonstration],
    execute: Executor,
    operation_library: Iterable[str],
    *,
    divergence_indices: Mapping[Genome, int | None] | None = None,
    max_steps: int = 8,
    mdl_costs: Mapping[str, float] | None = None,
    include_crossover: bool = False,
    max_parent_pairs: int = 64,
    mutation_radius: int = 1,
    max_mutation_candidates: int = 4_096,
    probe_inputs: Iterable[Any] = (),
    max_frontier_items: int | None = None,
) -> tuple[Fitness, ...]:
    """Expand one population and return its correctness/MDL Pareto frontier."""

    parents = tuple(sorted(set(population), key=lambda item: item.operations))
    if not parents:
        return ()
    if max_parent_pairs < 0:
        raise ValueError("max_parent_pairs must be non-negative")
    if mutation_radius < 0 or max_mutation_candidates < 0:
        raise ValueError("mutation radius and candidate cap must be non-negative")
    if max_frontier_items is not None and max_frontier_items < 0:
        raise ValueError("max_frontier_items must be non-negative or None")
    children: set[Genome] = set(parents)
    for parent in parents:
        if divergence_indices is None:
            mutations = mutation_shell(
                parent,
                operation_library,
                radius=mutation_radius,
                max_steps=max_steps,
                max_candidates=max_mutation_candidates,
            )
        else:
            if mutation_radius != 1:
                raise ValueError("guided divergence mutations require radius=1")
            mutations = guided_mutations(
                parent,
                divergence_indices.get(parent),
                operation_library,
                max_steps=max_steps,
            )
        children.update(mutations)
    if include_crossover:
        pair_count = 0
        for left_index, left in enumerate(parents):
            for right in parents[left_index + 1:]:
                if pair_count >= max_parent_pairs:
                    break
                children.update(crossover_genomes(left, right, max_steps=max_steps))
                pair_count += 1
            if pair_count >= max_parent_pairs:
                break
    frontier = pareto_frontier(
        evaluate_genome(
            genome, demonstrations, execute, mdl_costs=mdl_costs,
            probe_inputs=probe_inputs,
        )
        for genome in children
    )
    if max_frontier_items is None:
        return frontier
    return bounded_frontier(frontier, max_items=max_frontier_items)


if __name__ == "__main__":
    genome = Genome(("copy",))
    result = evolve_generation(
        [genome], [Demonstration(0, 1)],
        lambda item, value: value + (item.operations == ("inc",)),
        ["copy", "inc"],
    )
    assert any(item.exact for item in result)
    print("evolutionary_induction selftest: PASS")
