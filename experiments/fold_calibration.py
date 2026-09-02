"""Fit leakage-safe family calibration from labeled training-fold records."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from experiments.candidate_records import CandidateRecord, grid_hash, normalize_grid
    from experiments.hierarchical_calibration import (
        Outcome,
        hierarchical_rate,
    )
    from experiments.structural_groups import task_features, task_position_features
except ModuleNotFoundError:  # direct ``python experiments/fold_calibration.py``
    from candidate_records import CandidateRecord, grid_hash, normalize_grid
    from hierarchical_calibration import Outcome, hierarchical_rate
    from structural_groups import task_features, task_position_features


@dataclass(frozen=True)
class FamilyCalibration:
    global_outcome: Outcome
    feature_outcomes: Mapping[tuple[str, str], Outcome]

    def predict(self, features: Mapping[str, str], *, shrinkage: float = 8.0) -> float:
        return hierarchical_rate(
            self.global_outcome, self.feature_outcomes, features,
            shrinkage=shrinkage,
        )


def fit_family_calibration(
    records: list[CandidateRecord],
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, list[Any]],
    *,
    eligible_positions: Mapping[str, set[tuple[str, int]]] | None = None,
) -> dict[str, FamilyCalibration]:
    """Estimate family coverage, counting each task-position once.

    Multiple views and repeated samples from one family are collapsed to an
    output-class set before scoring.  A task-position is a success when any
    class emitted by that family equals its labeled training solution.  By
    default every labeled position is eligible; callers running adaptive
    routing may provide a per-family eligibility mask so positions that were
    never scheduled do not estimate the family's intrinsic hit rate.
    """

    classes: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for record in records:
        classes[(record.family, record.task_id, record.test_index)].add(record.output_hash)
    accum_global: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    accum_features: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    # Every labeled task-position is an observation for every active family.
    # An absent record means the family failed to cover that position; omitting
    # it would estimate a conditional hit rate on "positions it attempted"
    # and systematically overstate sparse families.
    active_families = tuple(sorted({family for family, _, _ in classes}))
    for family in active_families:
        allowed = (
            None if eligible_positions is None
            else eligible_positions.get(family, set())
        )
        for task_id, expected_outputs in solutions.items():
            task = challenges.get(task_id)
            if task is None:
                continue
            for index, expected in enumerate(expected_outputs):
                if index >= len(task.get("test", [])):
                    continue
                if allowed is not None and (task_id, index) not in allowed:
                    continue
                truth = grid_hash(normalize_grid(expected))
                success = int(truth in classes.get((family, task_id, index), set()))
                accum_global[family][success] += 1
                for name, value in task_position_features(task, index).items():
                    accum_features[(family, name, value)][success] += 1
    result: dict[str, FamilyCalibration] = {}
    for family, (failures, successes) in accum_global.items():
        features = {
            (name, value): Outcome(successes=values[1], failures=values[0])
            for (candidate_family, name, value), values in accum_features.items()
            if candidate_family == family
        }
        result[family] = FamilyCalibration(
            global_outcome=Outcome(successes=successes, failures=failures),
            feature_outcomes=features,
        )
    return result


def target_family_rates(
    calibrations: Mapping[str, FamilyCalibration],
    target_challenges: Mapping[str, Mapping[str, Any]],
    *,
    shrinkage: float = 8.0,
) -> dict[str, float]:
    """Average calibrated family rates over target task-positions."""

    result: dict[str, float] = {}
    for family, calibration in calibrations.items():
        estimates = []
        for task in target_challenges.values():
            features = task_features(task)
            estimates.extend(
                calibration.predict(features, shrinkage=shrinkage)
                for _ in task.get("test", [])
            )
        if estimates:
            result[family] = sum(estimates) / len(estimates)
    return result


def target_family_position_rates(
    calibrations: Mapping[str, FamilyCalibration],
    target_challenges: Mapping[str, Mapping[str, Any]],
    *,
    shrinkage: float = 8.0,
) -> dict[str, dict[tuple[str, int], float]]:
    """Project fold-calibrated rates onto each visible target position.

    The returned keys identify a task and test index, while values use only
    that task's challenge-visible input features.  No hidden target outputs
    are read here.  Keeping the position index makes complementary-lane
    routing possible without pretending all test positions share one rate.
    """

    result: dict[str, dict[tuple[str, int], float]] = {}
    for family, calibration in calibrations.items():
        family_rates: dict[tuple[str, int], float] = {}
        for task_id, task in target_challenges.items():
            for index, _ in enumerate(task.get("test", [])):
                features = task_position_features(task, index)
                family_rates[(task_id, index)] = calibration.predict(
                    features, shrinkage=shrinkage
                )
        if family_rates:
            result[family] = family_rates
    return result


if __name__ == "__main__":
    print("fold_calibration selftest: PASS")
