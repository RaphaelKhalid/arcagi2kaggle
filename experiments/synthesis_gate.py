"""Conservative gate for combining partial ARC reasoning traces.

Candidate synthesis is useful only when two traces contain complementary,
non-conflicting constraints.  This module does not execute the merged program
and therefore never certifies a synthesis; it chooses whether to spend a
model call on one and returns the exact constraint union to replay through the
CEGIS/demo-proof layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping


@dataclass(frozen=True)
class TraceCandidate:
    candidate_id: str
    family: str
    clauses: Mapping[str, str]
    hard_verified: bool = False


@dataclass(frozen=True)
class SynthesisDecision:
    should_synthesize: bool
    pair: tuple[str, str] | None
    merged_clauses: tuple[tuple[str, str], ...]
    complementarity: float
    reason: str


def compatible_union(
    left: Mapping[str, str], right: Mapping[str, str]
) -> dict[str, str] | None:
    """Return the constraint union, or ``None`` if an overlapping key conflicts."""

    merged = dict(left)
    for key, value in right.items():
        if key in merged and merged[key] != value:
            return None
        merged[key] = value
    return merged


def complementarity(left: Mapping[str, str], right: Mapping[str, str]) -> float:
    """Jaccard distance over clause keys, with conflicts treated separately."""

    keys_left, keys_right = set(left), set(right)
    union = keys_left | keys_right
    if not union:
        return 0.0
    return len(keys_left ^ keys_right) / len(union)


def gate_synthesis(
    candidates: list[TraceCandidate] | tuple[TraceCandidate, ...],
    *,
    unresolved_mass: float,
    min_unresolved: float = 0.25,
    min_complementarity: float = 0.25,
) -> SynthesisDecision:
    """Choose at most one cross-family compatible synthesis pair."""

    if not 0.0 <= unresolved_mass <= 1.0:
        raise ValueError("unresolved_mass must lie in [0, 1]")
    if not 0.0 <= min_unresolved <= 1.0:
        raise ValueError("min_unresolved must lie in [0, 1]")
    if not 0.0 <= min_complementarity <= 1.0:
        raise ValueError("min_complementarity must lie in [0, 1]")
    if unresolved_mass < min_unresolved:
        return SynthesisDecision(False, None, (), 0.0, "posterior already resolved")
    if any(candidate.hard_verified for candidate in candidates):
        return SynthesisDecision(False, None, (), 0.0, "hard-verified candidate exists")

    best: tuple[float, tuple[str, str], dict[str, str]] | None = None
    for left, right in combinations(sorted(candidates, key=lambda item: item.candidate_id), 2):
        if left.family == right.family:
            continue
        merged = compatible_union(left.clauses, right.clauses)
        if merged is None:
            continue
        score = complementarity(left.clauses, right.clauses)
        pair = (left.candidate_id, right.candidate_id)
        if best is None or (score, tuple(reversed(pair))) > (best[0], tuple(reversed(best[1]))):
            best = (score, pair, merged)
    if best is None or best[0] < min_complementarity:
        return SynthesisDecision(False, None, (), 0.0 if best is None else best[0],
                                "no sufficiently complementary compatible pair")
    score, pair, merged = best
    return SynthesisDecision(
        True, pair, tuple(sorted(merged.items())), score,
        "compatible cross-family partial constraints",
    )


if __name__ == "__main__":
    decision = gate_synthesis(
        [TraceCandidate("a", "text", {"role": "marker"}),
         TraceCandidate("b", "code", {"effect": "move"})],
        unresolved_mass=1.0,
    )
    assert decision.should_synthesize
    print("synthesis_gate selftest: PASS", decision)
