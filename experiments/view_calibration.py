"""Calibrate nuisance-view likelihoods before product-of-experts scoring.

ARC symmetries are valid transformations of the puzzle, but a checkpoint may
not assign comparable likelihoods to all serialized representations.  This
module estimates an additive per-view NLL offset from visible demonstrations
and applies it to candidate scores.  It is CPU-only and intentionally does
not generate grids or inspect hidden labels.
"""

from __future__ import annotations

from math import fsum, isfinite
from statistics import median
from typing import Iterable, Mapping


def estimate_view_offsets(
    view_demo_nlls: Mapping[str, Iterable[float]],
    *,
    shrinkage: float = 0.0,
) -> dict[str, float]:
    """Estimate additive NLL offsets relative to the per-demo view median.

    ``view_demo_nlls[view][demo]`` is the teacher-forced NLL of a visible
    demonstration answer under one serialized view.  A positive offset means
    that the view is systematically assigned larger NLL and should be
    corrected downward before PoE aggregation.  Shrinkage pulls offsets to
    zero when the task has too few demonstrations.
    """

    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must lie in [0, 1]")
    if not view_demo_nlls:
        raise ValueError("at least one view is required")
    values = {
        str(view): tuple(float(value) for value in nlls)
        for view, nlls in view_demo_nlls.items()
    }
    lengths = {len(nlls) for nlls in values.values()}
    if not lengths or 0 in lengths or len(lengths) != 1:
        raise ValueError("every view needs the same non-empty demo scores")
    if any(not isfinite(value) or value < 0.0
           for nlls in values.values() for value in nlls):
        raise ValueError("demo NLLs must be finite and non-negative")

    n_demos = next(iter(lengths))
    centers = tuple(
        median(values[view][demo] for view in values)
        for demo in range(n_demos)
    )
    scale = 1.0 - shrinkage
    return {
        view: scale * fsum(
            values[view][demo] - centers[demo]
            for demo in range(n_demos)
        ) / n_demos
        for view in sorted(values)
    }


def calibrated_poe_score(
    candidate_view_nlls: Mapping[str, float],
    view_offsets: Mapping[str, float],
    *,
    view_weights: Mapping[str, float] | None = None,
    missing_view_penalty: float = 0.0,
) -> float:
    """Return a calibrated weighted geometric-mean log score.

    The score is ``-sum_g w_g (NLL_g - offset_g)`` over present views.  Missing
    views are not fabricated; their absent weight incurs an optional explicit
    coverage penalty.  Weights are renormalized over present views so a view
    family does not dominate merely because another family failed to decode.
    """

    if missing_view_penalty < 0.0:
        raise ValueError("missing_view_penalty must be non-negative")
    if not candidate_view_nlls:
        raise ValueError("at least one candidate view score is required")
    if not view_offsets:
        raise ValueError("view offsets are required")
    weights = {
        view: 1.0 for view in view_offsets
    } if view_weights is None else {
        view: float(weight) for view, weight in view_weights.items()
    }
    if set(weights) != set(view_offsets):
        raise ValueError("weights must be provided for every calibrated view")
    if any(not isfinite(weight) or weight < 0.0 for weight in weights.values()):
        raise ValueError("view weights must be finite and non-negative")
    total_weight = fsum(weights.values())
    if total_weight <= 0.0:
        raise ValueError("view weights must have positive mass")

    present = [
        view for view in candidate_view_nlls
        if view in view_offsets and weights[view] > 0.0
    ]
    if not present:
        raise ValueError("candidate has no calibrated view support")
    for view in present:
        value = float(candidate_view_nlls[view])
        if not isfinite(value) or value < 0.0:
            raise ValueError("candidate NLLs must be finite and non-negative")
    present_weight = fsum(weights[view] for view in present)
    score = -fsum(
        weights[view] * (float(candidate_view_nlls[view]) - view_offsets[view])
        for view in present
    ) / present_weight
    missing_mass = (total_weight - present_weight) / total_weight
    return score - missing_view_penalty * missing_mass


def rank_candidates(
    candidates: Mapping[str, Mapping[str, float]],
    view_offsets: Mapping[str, float],
    *,
    view_weights: Mapping[str, float] | None = None,
    missing_view_penalty: float = 0.0,
) -> tuple[str, ...]:
    """Rank candidate IDs with deterministic tie-breaking."""

    scores = {
        candidate: calibrated_poe_score(
            nlls,
            view_offsets,
            view_weights=view_weights,
            missing_view_penalty=missing_view_penalty,
        )
        for candidate, nlls in candidates.items()
    }
    return tuple(sorted(scores, key=lambda candidate: (-scores[candidate], candidate)))


if __name__ == "__main__":
    offsets = estimate_view_offsets({"row": (1.0, 2.0), "column": (4.0, 5.0)})
    assert offsets == {"column": 1.5, "row": -1.5}
    print("view_calibration selftest: PASS")
