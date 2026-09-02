"""CPU-only D8 candidate-recall and pass@2 replay on a checked-in split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

try:
    from experiments.candidate_records import CandidateRecord, normalize_grid
    from experiments.replay_harness import replay_score
except ModuleNotFoundError:  # direct ``python experiments/geometric_replay.py``
    from candidate_records import CandidateRecord, normalize_grid
    from replay_harness import replay_score


def rotate_clockwise(grid: tuple[tuple[int, ...], ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in zip(*grid[::-1]))


def d8_transforms(value: Any) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Return the eight geometric orbit members in stable order."""

    current = normalize_grid(value)
    result: dict[str, tuple[tuple[int, ...], ...]] = {}
    for index in range(4):
        result[f"rot{index * 90}"] = current
        result[f"flip_rot{index * 90}"] = tuple(
            tuple(reversed(row)) for row in current
        )
        current = rotate_clockwise(current)
    return result


def build_geometric_records(
    challenges: Mapping[str, Mapping[str, Any]],
) -> list[CandidateRecord]:
    records: list[CandidateRecord] = []
    for task_id, task in challenges.items():
        for test_index, test in enumerate(task["test"]):
            for name, output in d8_transforms(test["input"]).items():
                records.append(
                    CandidateRecord.from_output(
                        task_id=task_id,
                        test_index=test_index,
                        family="geometry",
                        candidate_id=f"{task_id}:{test_index}:{name}",
                        output=output,
                    )
                )
    return records


def d8_recall(
    challenges: Mapping[str, Mapping[str, Any]],
    solutions: Mapping[str, list[Any]],
) -> tuple[int, int]:
    """Count outputs whose truth is in the geometric candidate orbit."""

    covered = total = 0
    for task_id, task in challenges.items():
        for index, test in enumerate(task["test"]):
            total += 1
            truth = normalize_grid(solutions[task_id][index])
            if truth in d8_transforms(test["input"]).values():
                covered += 1
    return covered, total


def load_split(root: Path, split: str) -> tuple[dict[str, Any], dict[str, Any]]:
    challenges = json.loads(
        (root / f"arc-agi_{split}_challenges.json").read_text()
    )
    solutions = json.loads(
        (root / f"arc-agi_{split}_solutions.json").read_text()
    )
    return challenges, solutions


if __name__ == "__main__":
    data_root = Path("data/raw")
    challenges, solutions = load_split(data_root, "evaluation")
    records = build_geometric_records(challenges)
    covered, total = d8_recall(challenges, solutions)
    selected_score = replay_score(records, solutions=solutions)
    print(
        f"geometric replay: D8 recall={covered}/{total} ({covered / total:.4f}), "
        f"selector pass@2={selected_score:.4f}, records={len(records)}"
    )
