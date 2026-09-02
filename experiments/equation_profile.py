"""Data-facing audit for constant versus reference-relative effect equations."""

from __future__ import annotations

from collections import Counter
import json
import sys
from typing import Any, Mapping

try:
    from experiments.effect_equations import (
        AnchorObservation,
        ConstantOffset,
        RolePlusOffset,
        fit_equations,
    )
    from experiments.object_correspondence import top_k_correspondences
    from experiments.object_deltas import extract_objects, normalize_grid
except ModuleNotFoundError:  # direct ``python experiments/equation_profile.py``
    from effect_equations import AnchorObservation, ConstantOffset, RolePlusOffset, fit_equations
    from object_correspondence import top_k_correspondences
    from object_deltas import extract_objects, normalize_grid


def _single_move_reference_observations(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> tuple[AnchorObservation, ...] | None:
    observations: list[AnchorObservation] = []
    for pair in task.get("train", []):
        source = extract_objects(normalize_grid(pair["input"]))
        target = extract_objects(normalize_grid(pair["output"]))
        try:
            correspondence = top_k_correspondences(
                pair["input"], pair["output"], k=k, max_objects=max_objects
            )[0]
        except (IndexError, ValueError):
            return None
        moves: list[tuple[int, int]] = []
        references: list[tuple[int, int]] = []
        for source_index, target_index in correspondence.pairs:
            left, right = source[source_index], target[target_index]
            if left.shape != right.shape or left.colored_shape != right.colored_shape:
                continue
            if left.anchor != right.anchor:
                moves.append((source_index, target_index))
            else:
                references.append((source_index, target_index))
        if len(moves) != 1 or not references:
            return None
        mover_source, mover_target = moves[0]
        reference_source, _ = sorted(references)[0]
        observations.append(AnchorObservation(
            source_role="A",
            target_anchor=target[mover_target].anchor,
            anchors={
                "A": source[mover_source].anchor,
                "B": source[reference_source].anchor,
            },
        ))
    return tuple(observations) if observations else None


def task_equation_profile(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, int | bool]:
    observations = _single_move_reference_observations(
        task, k=k, max_objects=max_objects
    )
    if observations is None:
        return {"eligible": False, "constant": False,
                "reference_relative": False, "reference_only": False}
    equations = fit_equations(observations)
    constant = any(isinstance(equation, ConstantOffset) for equation in equations)
    reference = any(isinstance(equation, RolePlusOffset) for equation in equations)
    return {
        "eligible": True,
        "constant": constant,
        "reference_relative": reference,
        "reference_only": reference and not constant,
    }


def dataset_equation_profile(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> dict[str, int]:
    summary = Counter({
        "tasks": 0,
        "eligible": 0,
        "constant_fit": 0,
        "reference_relative_fit": 0,
        "reference_only_fit": 0,
    })
    for task in challenges.values():
        summary["tasks"] += 1
        result = task_equation_profile(task, k=k, max_objects=max_objects)
        for key in ("eligible", "constant", "reference_relative", "reference_only"):
            summary[{
                "eligible": "eligible",
                "constant": "constant_fit",
                "reference_relative": "reference_relative_fit",
                "reference_only": "reference_only_fit",
            }[key]] += int(result[key])
    return dict(summary)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1], "r", encoding="utf-8") as handle:
            print(dataset_equation_profile(json.load(handle)))
    else:
        print("equation_profile selftest: PASS")
