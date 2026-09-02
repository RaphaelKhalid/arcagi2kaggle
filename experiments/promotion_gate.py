"""Reproducibility manifest and conservative fold-promotion gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from experiments.coverage_recovery import CoverageRecovery
from experiments.ambiguity_gate import AmbiguitySummary, ambiguity_regressions


@dataclass(frozen=True)
class RunManifest:
    """The immutable configuration needed to compare two candidate caches."""

    run_id: str
    competition: str
    code_revision: str
    model_artifacts: tuple[str, ...]
    gpu_count: int = 4
    wall_clock_seconds: int = 43_200
    safety_buffer_seconds: int = 600
    selector_mode: str = "collapse_correlated"
    fold_mode: str = "dev"

    def __post_init__(self) -> None:
        if not self.run_id or not self.competition or not self.code_revision:
            raise ValueError("run_id, competition, and code_revision are required")
        if not self.model_artifacts:
            raise ValueError("at least one model artifact is required")
        if not 1 <= self.gpu_count <= 4:
            raise ValueError("gpu_count must be in [1, 4]")
        if not 0 < self.wall_clock_seconds <= 43_200:
            raise ValueError("wall clock must be positive and at most 12 hours")
        if not 0 <= self.safety_buffer_seconds < self.wall_clock_seconds:
            raise ValueError("safety buffer must be smaller than wall clock")
        if self.selector_mode not in {"baseline", "collapse_correlated"}:
            raise ValueError("unknown selector mode")
        if self.fold_mode not in {"dev", "shadow_milestone"}:
            raise ValueError("unknown fold mode")

    @property
    def proposal_seconds(self) -> int:
        return self.wall_clock_seconds - self.safety_buffer_seconds

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reasons: tuple[str, ...]


def decide_promotion(
    baseline: CoverageRecovery,
    candidate: CoverageRecovery,
    *,
    require_shadow: bool = False,
    shadow_verified: bool = False,
    tolerance: float = 1e-12,
    baseline_ambiguity: AmbiguitySummary | None = None,
    candidate_ambiguity: AmbiguitySummary | None = None,
) -> PromotionDecision:
    """Promote only for coverage/recovery progress without score regression."""

    reasons: list[str] = []
    if baseline.total_positions != candidate.total_positions:
        reasons.append("development folds have different output counts")
    if baseline.total_tasks != candidate.total_tasks:
        reasons.append("development folds have different task counts")
    if require_shadow and not shadow_verified:
        reasons.append("shadow fold milestone verification is missing")
    if candidate.coverage_rate + tolerance < baseline.coverage_rate:
        reasons.append("candidate loses exact candidate coverage")
    if candidate.selector_recovery_rate + tolerance < baseline.selector_recovery_rate:
        reasons.append("candidate loses conditional selector recovery")
    if candidate.output_score + tolerance < baseline.output_score:
        reasons.append("candidate loses output-weighted score")
    if (baseline_ambiguity is None) != (candidate_ambiguity is None):
        reasons.append("ambiguity evidence is missing on one side")
    elif baseline_ambiguity is not None and candidate_ambiguity is not None:
        reasons.extend(ambiguity_regressions(
            baseline_ambiguity, candidate_ambiguity, tolerance=tolerance
        ))

    coverage_gain = candidate.coverage_rate > baseline.coverage_rate + tolerance
    recovery_gain = (
        candidate.selector_recovery_rate
        > baseline.selector_recovery_rate + tolerance
    )
    if not coverage_gain and not recovery_gain:
        reasons.append("no coverage or selector-recovery gain")
    return PromotionDecision(promote=not reasons, reasons=tuple(reasons))


if __name__ == "__main__":
    manifest = RunManifest("demo", "arc-agi-2", "local", ("model",))
    assert manifest.proposal_seconds == 42_600
    print("promotion_gate selftest: PASS")
