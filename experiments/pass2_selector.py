"""Replayable pass@2 selector for proof-carrying candidate records.

This is a CPU-only research prototype. It does not generate ARC outputs. It
implements the selection theorem from ``AUTORESEARCH_LOG_2026-09-01.md``:
aggregate candidates by exact output class, estimate class mass across solver
families, and select two distinct classes. The interface is intentionally
model-agnostic so saved NVARC/TRM/Leg-C records can be replayed later.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, fsum, log
from typing import Hashable, Iterable, Mapping, TypeVar


ClassKey = TypeVar("ClassKey", bound=Hashable)


@dataclass(frozen=True)
class Candidate:
    """A normalized proposal after deterministic validation."""

    output_hash: str
    family: str
    weight: float = 1.0
    mdl_length: float = 0.0
    hard_valid: bool = True
    correlation_group: str | None = None


@dataclass(frozen=True)
class OutputClassScore:
    output_hash: str
    score: float
    family_support: int
    best_mdl_length: float


@dataclass(frozen=True)
class TaskCandidate:
    """One program hypothesis evaluated on every test input of a task."""

    program_hash: str
    output_vector: tuple[str, ...]
    family: str
    weight: float = 1.0
    mdl_length: float = 0.0
    correlation_group: str | None = None


def _normalize_class_scores(
    scores: Mapping[ClassKey, float],
    alpha: float,
) -> dict[ClassKey, float]:
    """Normalize one class distribution with the selector's smoothing."""

    if not scores:
        return {}
    total = fsum(scores.values())
    if total <= 0.0:
        scores = {key: 1.0 for key in scores}
        total = float(len(scores))
    denominator = total + alpha * len(scores)
    return {
        key: (value + alpha) / denominator
        for key, value in scores.items()
    }


def _family_conditional_mass(
    family_candidates: list[Candidate],
    alpha: float,
    collapse_correlated: bool = False,
) -> dict[str, float]:
    """Return a smoothed distribution over observed output classes.

    Duplicate proposals are not treated as independent evidence: each family
    contributes a distribution whose total mass is one. This is a conservative
    approximation to effective-sample-size correction while candidate caches
    are still being standardized.
    """

    if not collapse_correlated:
        by_output: dict[str, float] = defaultdict(float)
        for candidate in family_candidates:
            by_output[candidate.output_hash] += max(0.0, candidate.weight)
        return _normalize_class_scores(by_output, alpha)

    # A correlation group is one lineage (checkpoint/prompt/temperature/
    # repair chain).  Collapse copies within it, then average lineages so a
    # family can contain multiple genuinely independent sources.
    by_group: dict[str, dict[str, float]] = defaultdict(dict)
    for candidate in family_candidates:
        group = candidate.correlation_group or "__default__"
        score = (
            max(0.0, candidate.weight)
            * exp(-max(0.0, candidate.mdl_length) * log(2.0))
        )
        by_group[group][candidate.output_hash] = max(
            by_group[group].get(candidate.output_hash, 0.0), score
        )
    distributions = [
        _normalize_class_scores(scores, alpha)
        for scores in by_group.values()
    ]
    outputs = {output for distribution in distributions for output in distribution}
    return {
        output: fsum(distribution.get(output, 0.0) for distribution in distributions)
        / len(distributions)
        for output in outputs
    }


def _family_vector_mass(
    family_candidates: list[TaskCandidate],
    alpha: float,
    collapse_correlated: bool,
) -> dict[tuple[str, ...], float]:
    """Return a conditional distribution over complete output vectors."""

    if not collapse_correlated:
        by_vector: dict[tuple[str, ...], float] = defaultdict(float)
        for candidate in family_candidates:
            by_vector[candidate.output_vector] += max(0.0, candidate.weight)
        return _normalize_class_scores(by_vector, alpha)

    grouped: dict[str, dict[tuple[str, ...], float]] = defaultdict(dict)
    for candidate in family_candidates:
        group = candidate.correlation_group or "__default__"
        score = (
            max(0.0, candidate.weight)
            * exp(-max(0.0, candidate.mdl_length) * log(2.0))
        )
        grouped[group][candidate.output_vector] = max(
            grouped[group].get(candidate.output_vector, 0.0), score
        )
    distributions = [
        _normalize_class_scores(scores, alpha)
        for scores in grouped.values()
    ]
    vectors = {vector for distribution in distributions for vector in distribution}
    return {
        vector: fsum(distribution.get(vector, 0.0) for distribution in distributions)
        / len(distributions)
        for vector in vectors
    }


def score_output_classes(
    candidates: Iterable[Candidate],
    *,
    family_priors: Mapping[str, float] | None = None,
    alpha: float = 0.25,
    collapse_correlated: bool = False,
) -> list[OutputClassScore]:
    """Score observed output classes with equal/evidence-weighted families.

    A family contributes at most its prior mass, regardless of how many
    correlated samples it emitted. This is intentionally less aggressive than
    a raw vote and is suitable for the first replay ablation.
    """

    by_family: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.hard_valid:
            by_family[candidate.family].append(candidate)
    if not by_family:
        return []

    if family_priors is None:
        prior = {family: 1.0 / len(by_family) for family in by_family}
    else:
        raw = {family: max(0.0, float(family_priors.get(family, 0.0)))
               for family in by_family}
        total = fsum(raw.values())
        prior = ({family: value / total for family, value in raw.items()}
                 if total > 0.0 else
                 {family: 1.0 / len(by_family) for family in by_family})

    class_scores: dict[str, float] = defaultdict(float)
    family_support: dict[str, set[str]] = defaultdict(set)
    best_mdl: dict[str, float] = {}
    for family, family_candidates in by_family.items():
        conditional = _family_conditional_mass(
            family_candidates, alpha, collapse_correlated
        )
        for output_hash, mass in conditional.items():
            class_scores[output_hash] += prior[family] * mass
            family_support[output_hash].add(family)
        for candidate in family_candidates:
            old = best_mdl.get(candidate.output_hash, float("inf"))
            best_mdl[candidate.output_hash] = min(old, candidate.mdl_length)

    return sorted(
        (
            OutputClassScore(
                output_hash=output_hash,
                score=score,
                family_support=len(family_support[output_hash]),
                best_mdl_length=best_mdl.get(output_hash, float("inf")),
            )
            for output_hash, score in class_scores.items()
        ),
        key=lambda item: (
            -item.score,
            -item.family_support,
            item.best_mdl_length,
            item.output_hash,
        ),
    )


def select_pass2(
    candidates: Iterable[Candidate],
    *,
    family_priors: Mapping[str, float] | None = None,
    alpha: float = 0.25,
    collapse_correlated: bool = False,
) -> tuple[str, ...]:
    """Return up to two distinct output hashes for submission attempts."""

    ranked = score_output_classes(
        candidates,
        family_priors=family_priors,
        alpha=alpha,
        collapse_correlated=collapse_correlated,
    )
    return tuple(item.output_hash for item in ranked[:2])


def select_task_program_pair(
    candidates: Iterable[TaskCandidate],
    *,
    family_priors: Mapping[str, float] | None = None,
    alpha: float = 0.25,
    collapse_correlated: bool = False,
) -> tuple[TaskCandidate, ...]:
    """Choose up to two shared-rule hypotheses as a coherence diagnostic.

    This constrained objective is useful when we want one coherent rule pair,
    but it is *not* the official optimum: Kaggle scores each test output
    independently, so the production selector should use
    ``select_task_output_pairs`` below. Programs are first collapsed by their
    complete output vector, so syntactic duplicates cannot inflate a class.
    """

    candidates = list(candidates)
    if not candidates:
        return ()
    vector_lengths = {len(candidate.output_vector) for candidate in candidates}
    if len(vector_lengths) != 1 or not next(iter(vector_lengths)):
        raise ValueError("all task candidates must have the same non-empty output vector")

    by_family: dict[str, list[TaskCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_family[candidate.family].append(candidate)

    if family_priors is None:
        prior = {family: 1.0 / len(by_family) for family in by_family}
    else:
        raw = {family: max(0.0, float(family_priors.get(family, 0.0)))
               for family in by_family}
        total = fsum(raw.values())
        prior = ({family: value / total for family, value in raw.items()}
                 if total > 0.0 else
                 {family: 1.0 / len(by_family) for family in by_family})

    vector_mass: dict[tuple[str, ...], float] = defaultdict(float)
    representative: dict[tuple[str, ...], TaskCandidate] = {}
    for family, family_candidates in by_family.items():
        by_vector = _family_vector_mass(
            family_candidates, alpha, collapse_correlated
        )
        for candidate in family_candidates:
            old = representative.get(candidate.output_vector)
            if old is None or candidate.mdl_length < old.mdl_length:
                representative[candidate.output_vector] = candidate
        for vector, mass in by_vector.items():
            vector_mass[vector] += prior[family] * mass

    vectors = list(vector_mass)
    best_pair: tuple[tuple[str, ...], ...] | None = None
    best_utility = float("-inf")
    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1:]:
            utility = sum(
                sum(
                    mass for vector, mass in vector_mass.items()
                    if vector[position] in {left[position], right[position]}
                )
                for position in range(len(left))
            )
            pair = (left, right)
            if (utility > best_utility or
                    (utility == best_utility and pair < (best_pair or pair))):
                best_utility = utility
                best_pair = pair

    if best_pair is None:
        return (representative[vectors[0]],)
    return tuple(representative[vector] for vector in best_pair)


def select_task_output_pairs(
    candidates: Iterable[TaskCandidate],
    *,
    family_priors: Mapping[str, float] | None = None,
    alpha: float = 0.25,
    collapse_correlated: bool = False,
) -> tuple[tuple[str, ...], ...]:
    """Select the official pass@2 pair independently for each test output.

    Candidate programs still induce a joint posterior over complete output
    vectors, which is valuable evidence.  But the official score is the sum
    of per-output indicators.  Therefore, for output position ``j``, the
    Bayes-optimal action is the two highest-mass marginal classes

        P_j(z) = sum_v P(v) * 1[v[j] == z].

    The returned tuple contains one (up to two) output-hash pair per position.
    This can choose combinations not emitted by a single program; that is
    allowed by the submission format and is optimal for the additive metric.
    """

    candidates = list(candidates)
    if not candidates:
        return ()
    vector_lengths = {len(candidate.output_vector) for candidate in candidates}
    if len(vector_lengths) != 1 or not next(iter(vector_lengths)):
        raise ValueError("all task candidates must have the same non-empty output vector")

    by_family: dict[str, list[TaskCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_family[candidate.family].append(candidate)

    if family_priors is None:
        prior = {family: 1.0 / len(by_family) for family in by_family}
    else:
        raw = {family: max(0.0, float(family_priors.get(family, 0.0)))
               for family in by_family}
        total = fsum(raw.values())
        prior = ({family: value / total for family, value in raw.items()}
                 if total > 0.0 else
                 {family: 1.0 / len(by_family) for family in by_family})

    vector_mass: dict[tuple[str, ...], float] = defaultdict(float)
    for family, family_candidates in by_family.items():
        by_vector = _family_vector_mass(
            family_candidates, alpha, collapse_correlated
        )
        for vector, mass in by_vector.items():
            vector_mass[vector] += prior[family] * mass

    n_positions = len(next(iter(vector_mass)))
    pairs: list[tuple[str, ...]] = []
    for position in range(n_positions):
        marginal: dict[str, float] = defaultdict(float)
        for vector, mass in vector_mass.items():
            marginal[vector[position]] += mass
        ranked = sorted(marginal, key=lambda output: (-marginal[output], output))
        pairs.append(tuple(ranked[:2]))
    return tuple(pairs)


if __name__ == "__main__":
    # A raw vote would choose ``wrong`` (ten correlated samples). Equal-family
    # aggregation recognizes the minority family and keeps its output in the
    # pass@2 set.
    demo = [
        *(Candidate("wrong", "dense") for _ in range(10)),
        Candidate("correct", "program"),
        Candidate("alternate", "recursive"),
    ]
    selected = select_pass2(
        demo, family_priors={"dense": 0.5, "program": 0.3, "recursive": 0.2}
    )
    assert selected == ("wrong", "correct"), selected
    task_pair = select_task_program_pair(
        [
            *(TaskCandidate(f"dense-{i}", ("wrong", "wrong"), "dense")
              for i in range(8)),
            TaskCandidate("program", ("correct", "correct"), "program"),
            TaskCandidate("recursive", ("alternate", "correct"), "recursive"),
        ],
        family_priors={"dense": 0.5, "program": 0.3, "recursive": 0.2},
    )
    assert {candidate.program_hash for candidate in task_pair} == {"dense-0", "program"}
    output_pairs = select_task_output_pairs(
        [
            TaskCandidate("v1", ("a", "x"), "program", weight=4.0),
            TaskCandidate("v2", ("a", "y"), "program", weight=3.0),
            TaskCandidate("v3", ("b", "z"), "program", weight=2.0),
            TaskCandidate("v4", ("c", "x"), "program", weight=1.0),
        ]
    )
    assert output_pairs == (("a", "b"), ("x", "y")), output_pairs
    print("pass2_selector selftest: PASS", selected,
          [c.program_hash for c in task_pair], output_pairs)
