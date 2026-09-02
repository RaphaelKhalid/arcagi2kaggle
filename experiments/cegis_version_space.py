"""Counterexample-guided version-space control for an offline ARC library.

CEGIS is the proof layer between a transform library and a prediction: keep
only programs that reproduce every demonstration, quotient syntactic programs
by their predicted grid, and make the pass@2 action from the resulting output
posterior.  The module is deliberately executor-agnostic so a Kaggle notebook
can plug in verified primitives, a DSL compiler, or a cached model program.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, fsum, log
from typing import Any, Callable, Iterable, Mapping


OutputKey = Any
Executor = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class Program:
    """One executable hypothesis with an offline prior and a family label."""

    program_id: str
    family: str
    mdl_length: float = 0.0
    prior_weight: float = 1.0


@dataclass(frozen=True)
class Demonstration:
    """A single input/output constraint."""

    input: Any
    output: Any


@dataclass(frozen=True)
class CEGISResult:
    """The surviving proof-carrying hypotheses and official test action."""

    survivors: tuple[str, ...]
    selected_outputs: tuple[tuple[OutputKey, ...], ...]


def freeze_output(value: Any) -> OutputKey:
    """Canonicalize nested ARC grids (or scalar test doubles) for equality."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(freeze_output(item) for item in value)
    return value


def _prediction(program: Program, input_value: Any, execute: Executor) -> OutputKey:
    result = execute(program, input_value)
    if result is None:
        raise ValueError(f"program {program.program_id!r} returned no output")
    return freeze_output(result)


def filter_version_space(
    programs: Iterable[Program],
    demonstrations: Iterable[Demonstration],
    execute: Executor,
) -> tuple[Program, ...]:
    """Return exactly the programs consistent with every demonstration."""

    programs = tuple(programs)
    demonstrations = tuple(demonstrations)
    survivors: list[Program] = []
    for program in programs:
        if all(
            _prediction(program, demo.input, execute) == freeze_output(demo.output)
            for demo in demonstrations
        ):
            survivors.append(program)
    return tuple(survivors)


def most_discriminating_demo(
    programs: Iterable[Program],
    demonstrations: Iterable[Demonstration],
    execute: Executor,
) -> int | None:
    """Choose the next demo that removes the most hypotheses.

    The score is ``n - largest prediction bucket``.  It is a conservative
    CEGIS heuristic: a demo with a large partition is useful, while ties are
    resolved by original order.  Returning ``None`` means there is no demo.
    """

    programs = tuple(programs)
    demonstrations = tuple(demonstrations)
    if not demonstrations or not programs:
        return None
    best_index: int | None = None
    best_score = -1
    for index, demo in enumerate(demonstrations):
        buckets: dict[OutputKey, int] = defaultdict(int)
        for program in programs:
            buckets[_prediction(program, demo.input, execute)] += 1
        score = len(programs) - max(buckets.values(), default=0)
        if score > best_score:
            best_index, best_score = index, score
    return best_index


def _family_masses(
    programs: Iterable[Program],
    *,
    family_priors: Mapping[str, float] | None,
) -> dict[str, float]:
    by_family: dict[str, list[Program]] = defaultdict(list)
    for program in programs:
        by_family[program.family].append(program)
    if not by_family:
        return {}
    if family_priors is None:
        return {family: 1.0 / len(by_family) for family in by_family}
    raw = {family: max(0.0, float(family_priors.get(family, 0.0)))
           for family in by_family}
    total = fsum(raw.values())
    return (
        {family: value / total for family, value in raw.items()}
        if total > 0.0
        else {family: 1.0 / len(by_family) for family in by_family}
    )


def posterior_outputs(
    programs: Iterable[Program],
    test_input: Any,
    execute: Executor,
    *,
    family_priors: Mapping[str, float] | None = None,
) -> tuple[tuple[OutputKey, float], ...]:
    """Aggregate survivor mass by exact test-output class.

    Program multiplicity cannot inflate a family's mass: priors are normalized
    within each family before the family prior is applied.  This is the key
    anti-vote-counting property needed when a decoder emits correlated copies.
    """

    programs = tuple(programs)
    if not programs:
        return ()
    masses = _family_masses(programs, family_priors=family_priors)
    by_family: dict[str, list[Program]] = defaultdict(list)
    for program in programs:
        by_family[program.family].append(program)
    output_mass: dict[OutputKey, float] = defaultdict(float)
    for family, family_programs in by_family.items():
        raw = {
            program.program_id: max(0.0, program.prior_weight)
            for program in family_programs
        }
        total = fsum(raw.values())
        if total <= 0.0:
            total = float(len(family_programs))
            raw = {program.program_id: 1.0 for program in family_programs}
        for program in family_programs:
            output_mass[_prediction(program, test_input, execute)] += (
                masses[family] * raw[program.program_id] / total
            )
    return tuple(sorted(output_mass.items(), key=lambda item: (-item[1], repr(item[0]))))


def semantic_posterior_outputs(
    programs: Iterable[Program],
    test_input: Any,
    execute: Executor,
    *,
    family_priors: Mapping[str, float] | None = None,
) -> tuple[tuple[OutputKey, float], ...]:
    """Aggregate output mass after collapsing correlated programs.

    Within a family, programs that predict the same test output are one
    semantic hypothesis, not independent votes.  The class receives the
    strongest program prior in that family, with MDL acting as a prefix-code
    prior when lengths are supplied.  This is intentionally separate from
    :func:`posterior_outputs`: multiplicity can represent genuine independent
    derivations, so a caller must opt into the anti-correlation quotient.
    """

    programs = tuple(programs)
    if not programs:
        return ()
    masses = _family_masses(programs, family_priors=family_priors)
    by_family: dict[str, list[Program]] = defaultdict(list)
    for program in programs:
        by_family[program.family].append(program)

    output_mass: dict[OutputKey, float] = defaultdict(float)
    for family, family_programs in by_family.items():
        by_output: dict[OutputKey, list[Program]] = defaultdict(list)
        for program in family_programs:
            by_output[_prediction(program, test_input, execute)].append(program)
        class_scores = {
            output: max(
                max(0.0, program.prior_weight)
                * exp(-max(0.0, program.mdl_length) * log(2.0))
                for program in class_programs
            )
            for output, class_programs in by_output.items()
        }
        total = fsum(class_scores.values())
        if total <= 0.0:
            total = float(len(class_scores))
            class_scores = {output: 1.0 for output in class_scores}
        for output, score in class_scores.items():
            output_mass[output] += masses[family] * score / total
    return tuple(sorted(output_mass.items(), key=lambda item: (-item[1], repr(item[0]))))


def cegis_solve(
    programs: Iterable[Program],
    demonstrations: Iterable[Demonstration],
    test_inputs: Iterable[Any],
    execute: Executor,
    *,
    family_priors: Mapping[str, float] | None = None,
    collapse_correlated: bool = False,
) -> CEGISResult:
    """Filter the exact version space and return independent top-two actions."""

    survivors = filter_version_space(programs, demonstrations, execute)
    aggregate = (
        semantic_posterior_outputs
        if collapse_correlated
        else posterior_outputs
    )
    selected = tuple(
        tuple(output for output, _ in aggregate(
            survivors, test_input, execute, family_priors=family_priors
        )[:2])
        for test_input in test_inputs
    )
    return CEGISResult(
        survivors=tuple(program.program_id for program in survivors),
        selected_outputs=selected,
    )


if __name__ == "__main__":
    def execute(program: Program, value: int) -> int:
        return value + (1 if program.program_id == "right" else 0)

    result = cegis_solve(
        [Program("right", "synth"), Program("wrong", "copy")],
        [Demonstration(0, 1)], [2], execute,
    )
    assert result.survivors == ("right",)
    print("cegis_version_space selftest: PASS")
