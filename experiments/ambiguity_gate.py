"""Promotion diagnostics for finite version-space ambiguity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from experiments.version_space_certificate import VersionSpaceCertificate


@dataclass(frozen=True)
class AmbiguitySummary:
    """Aggregate hidden-input identification diagnostics over tasks."""

    total_positions: int
    unresolved_positions: int
    forced_positions: int
    pass2_coverable_positions: int
    mean_ambiguity: float

    @property
    def forced_rate(self) -> float:
        return self.forced_positions / self.total_positions if self.total_positions else 0.0

    @property
    def pass2_coverable_rate(self) -> float:
        return (
            self.pass2_coverable_positions / self.total_positions
            if self.total_positions else 0.0
        )

    @property
    def unresolved_rate(self) -> float:
        return self.unresolved_positions / self.total_positions if self.total_positions else 0.0


def summarize_certificates(
    certificates: Iterable[VersionSpaceCertificate],
) -> AmbiguitySummary:
    """Summarize only the finite candidate libraries supplied by the caller."""

    total = unresolved = forced = coverable = ambiguity_sum = 0
    for certificate in certificates:
        for test in certificate.tests:
            total += 1
            ambiguity_sum += test.ambiguity_count
            if not certificate.demo_verified:
                unresolved += 1
            elif test.forced_output is not None:
                forced += 1
            if certificate.demo_verified and test.pass2_covers_entire_version_space:
                coverable += 1
    return AmbiguitySummary(
        total_positions=total,
        unresolved_positions=unresolved,
        forced_positions=forced,
        pass2_coverable_positions=coverable,
        mean_ambiguity=ambiguity_sum / total if total else 0.0,
    )


def ambiguity_regressions(
    baseline: AmbiguitySummary,
    candidate: AmbiguitySummary,
    *,
    tolerance: float = 1e-12,
) -> tuple[str, ...]:
    """Return regressions that should block a candidate-cache promotion."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    reasons: list[str] = []
    if baseline.total_positions != candidate.total_positions:
        reasons.append("ambiguity summaries have different output counts")
        return tuple(reasons)
    if candidate.unresolved_rate > baseline.unresolved_rate + tolerance:
        reasons.append("candidate increases unresolved version-space positions")
    if candidate.forced_rate + tolerance < baseline.forced_rate:
        reasons.append("candidate loses forced-output positions")
    if candidate.pass2_coverable_rate + tolerance < baseline.pass2_coverable_rate:
        reasons.append("candidate loses pass@2-coverable positions")
    if candidate.mean_ambiguity > baseline.mean_ambiguity + tolerance:
        reasons.append("candidate increases mean semantic ambiguity")
    return tuple(reasons)
