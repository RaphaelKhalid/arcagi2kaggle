"""Lineage-aware aggregation of demo-verified program predictions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, fsum, log
from typing import Any, Iterable


def freeze_prediction(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(freeze_prediction(item) for item in value)
    return value


@dataclass(frozen=True)
class VerifiedPrediction:
    """One program that passed every demonstration."""

    program_id: str
    prediction: Any
    mdl_length: float = 0.0
    correlation_group: str | None = None

    def __post_init__(self) -> None:
        if not self.program_id:
            raise ValueError("program_id is required")
        if self.mdl_length < 0:
            raise ValueError("mdl_length must be non-negative")


@dataclass(frozen=True)
class VerifiedOutputClass:
    prediction: Any
    mass: float
    representative_program: str
    witness_groups: tuple[str, ...]


def rank_verified_outputs(
    predictions: Iterable[VerifiedPrediction],
    *,
    tau: float = 1.0,
    collapse_correlated: bool = True,
) -> tuple[VerifiedOutputClass, ...]:
    """Rank verified output classes without raw correlated-vote inflation.

    Each correlation group contributes one normalized distribution over its
    semantic output classes.  Within a group, the best MDL witness for a class
    is retained.  With collapse disabled, every program becomes its own group,
    reproducing ordinary program-level evidence while retaining MDL weights.
    """

    if tau <= 0:
        raise ValueError("tau must be positive")
    items = tuple(predictions)
    if not items:
        return ()
    grouped: dict[str, list[VerifiedPrediction]] = defaultdict(list)
    for item in items:
        group = item.correlation_group if collapse_correlated else item.program_id
        grouped[group or item.program_id].append(item)

    class_mass: dict[Any, float] = defaultdict(float)
    representatives: dict[Any, VerifiedPrediction] = {}
    witness_groups: dict[Any, set[str]] = defaultdict(set)
    group_weight = 1.0 / len(grouped)
    for group, group_items in sorted(grouped.items()):
        by_output: dict[Any, list[VerifiedPrediction]] = defaultdict(list)
        for item in group_items:
            by_output[freeze_prediction(item.prediction)].append(item)
        class_scores = {
            output: max(exp(-item.mdl_length * log(2.0) / tau) for item in members)
            for output, members in by_output.items()
        }
        total = fsum(class_scores.values())
        for output, score in class_scores.items():
            class_mass[output] += group_weight * score / total
            witness_groups[output].add(group)
            representative = min(
                by_output[output], key=lambda item: (item.mdl_length, item.program_id)
            )
            old = representatives.get(output)
            if old is None or (representative.mdl_length, representative.program_id) < (
                old.mdl_length, old.program_id
            ):
                representatives[output] = representative

    ranked = sorted(class_mass, key=lambda output: (
        -class_mass[output], repr(output)
    ))
    return tuple(
        VerifiedOutputClass(
            prediction=output,
            mass=class_mass[output],
            representative_program=representatives[output].program_id,
            witness_groups=tuple(sorted(witness_groups[output])),
        )
        for output in ranked
    )


def select_verified_pass2(
    predictions: Iterable[VerifiedPrediction],
    *,
    tau: float = 1.0,
    collapse_correlated: bool = True,
) -> tuple[Any, ...]:
    """Return at most two verified output classes for the official action."""

    return tuple(item.prediction for item in rank_verified_outputs(
        predictions, tau=tau, collapse_correlated=collapse_correlated
    )[:2])
