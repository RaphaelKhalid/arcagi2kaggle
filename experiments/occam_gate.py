"""Occam/PAC-Bayes control for demo-consistent ARC programs.

This is a conservative ranking aid, not a claim that ARC demonstrations are
i.i.d. samples.  Under the explicit within-task sampling assumption and a
prefix-free program code, a union-bound/Occam argument gives an upper bound on
unseen-example error from empirical demo error plus a description-length
penalty.  The bound is useful as a gate against arbitrary complex programs;
the version-space and proof certificate remain the hard correctness gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Iterable


@dataclass(frozen=True)
class OccamCandidate:
    program_id: str
    failures: int
    mdl_bits: float


@dataclass(frozen=True)
class OccamScore:
    program_id: str
    empirical_error: float
    complexity_penalty: float
    upper_error: float


def complexity_penalty(
    mdl_bits: float,
    n_examples: int,
    *,
    delta: float = 0.05,
) -> float:
    """Return the finite-sample Occam penalty in a bounded error rate.

    For a prefix-free code, the penalty is the Hoeffding term for prior mass
    proportional to ``2**(-mdl_bits)``.  It is clipped at one because an error
    rate is bounded.  This function does not silently accept a non-positive
    sample count or invalid confidence level.
    """

    if mdl_bits < 0:
        raise ValueError("mdl_bits must be non-negative")
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return min(
        1.0,
        sqrt((mdl_bits * log(2.0) + log(2.0 / delta)) / (2.0 * n_examples)),
    )


def score_occam(
    candidate: OccamCandidate,
    n_examples: int,
    *,
    delta: float = 0.05,
) -> OccamScore:
    """Score one candidate using observed demo failures plus complexity."""

    if not 0 <= candidate.failures <= n_examples:
        raise ValueError("failures must be between zero and n_examples")
    empirical = candidate.failures / n_examples
    penalty = complexity_penalty(candidate.mdl_bits, n_examples, delta=delta)
    return OccamScore(
        program_id=candidate.program_id,
        empirical_error=empirical,
        complexity_penalty=penalty,
        upper_error=min(1.0, empirical + penalty),
    )


def rank_by_occam(
    candidates: Iterable[OccamCandidate],
    n_examples: int,
    *,
    delta: float = 0.05,
) -> tuple[OccamScore, ...]:
    """Rank candidates by conservative unseen-demo error upper bound."""

    scores = [score_occam(candidate, n_examples, delta=delta)
              for candidate in candidates]
    return tuple(sorted(scores, key=lambda score: (
        score.upper_error, score.empirical_error, score.program_id
    )))


if __name__ == "__main__":
    score = score_occam(OccamCandidate("identity", 0, 4), 8)
    assert 0.0 <= score.upper_error <= 1.0
    print("occam_gate selftest: PASS", score)
