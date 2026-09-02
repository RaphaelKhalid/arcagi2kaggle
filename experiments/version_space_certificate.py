"""Finite version-space certificates for ARC test ambiguity.

A demo-consistent program is not necessarily a uniquely identified rule.  This
module makes that distinction explicit: after exact demo filtering, it counts
distinct semantic outputs on each test input and reports whether the surviving
version space forces one output, fits inside the official two-attempt action,
or remains irreducibly wider than two outputs under the configured posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Callable, Iterable, Mapping

try:
    from experiments.cegis_version_space import (
        Demonstration,
        Executor,
        Program,
        filter_version_space,
        semantic_posterior_outputs,
    )
except ModuleNotFoundError:  # direct ``python experiments/version_space_certificate.py``
    from cegis_version_space import (
        Demonstration,
        Executor,
        Program,
        filter_version_space,
        semantic_posterior_outputs,
    )


OutputKey = Any


@dataclass(frozen=True)
class TestVersionSpace:
    """The exact semantic uncertainty induced by one test input."""

    test_index: int
    output_classes: tuple[OutputKey, ...]
    output_masses: tuple[tuple[OutputKey, float], ...]

    @property
    def ambiguity_count(self) -> int:
        return len(self.output_classes)

    @property
    def forced_output(self) -> OutputKey | None:
        return self.output_classes[0] if len(self.output_classes) == 1 else None

    @property
    def top2_posterior_mass(self) -> float:
        return sum(mass for _, mass in self.output_masses[:2])

    @property
    def pass2_covers_entire_version_space(self) -> bool:
        """Whether two outputs cover every surviving semantic class."""

        return self.ambiguity_count <= 2


@dataclass(frozen=True)
class VersionSpaceCertificate:
    """Proof summary after exact demo filtering.

    ``demo_verified`` means every survivor reproduces every supplied demo.
    ``task_forced`` is stronger: every survivor also predicts the same output
    on every supplied test input.  Neither property claims hidden-label
    correctness when the candidate library is incomplete.
    """

    survivors: tuple[str, ...]
    tests: tuple[TestVersionSpace, ...]

    @property
    def demo_verified(self) -> bool:
        return bool(self.survivors)

    @property
    def task_forced(self) -> bool:
        return self.demo_verified and all(test.forced_output is not None for test in self.tests)

    @property
    def pass2_covers_entire_task_version_space(self) -> bool:
        return self.demo_verified and all(
            test.pass2_covers_entire_version_space for test in self.tests
        )


def certify_version_space(
    programs: Iterable[Program],
    demonstrations: Iterable[Demonstration],
    test_inputs: Iterable[Any],
    execute: Executor,
    *,
    family_priors: Mapping[str, float] | None = None,
) -> VersionSpaceCertificate:
    """Filter programs on demos and certify semantic test ambiguity exactly."""

    survivors = filter_version_space(programs, demonstrations, execute)
    tests: list[TestVersionSpace] = []
    for test_index, test_input in enumerate(test_inputs):
        posterior = semantic_posterior_outputs(
            survivors, test_input, execute, family_priors=family_priors
        )
        classes = tuple(output for output, _ in posterior)
        tests.append(TestVersionSpace(test_index, classes, posterior))
    return VersionSpaceCertificate(
        survivors=tuple(program.program_id for program in survivors),
        tests=tuple(tests),
    )


def posterior_is_complete(test: TestVersionSpace, *, tolerance: float = 1e-9) -> bool:
    """Check that the selected two-class posterior has no residual mass."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    return isclose(test.top2_posterior_mass, 1.0, rel_tol=0.0, abs_tol=tolerance)


if __name__ == "__main__":
    def execute(program: Program, value: int) -> int:
        return value + (1 if program.program_id == "right" else 0)

    certificate = certify_version_space(
        [Program("right", "synth")],
        [Demonstration(0, 1)],
        [2],
        execute,
    )
    assert certificate.task_forced
    print("version_space_certificate selftest: PASS")
