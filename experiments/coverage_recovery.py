"""Leakage-safe decomposition of candidate coverage and selector recovery.

The evaluator is intended for labeled training folds only.  It separates the
two controllable failure modes: the correct exact output was absent from the
candidate set, or it was present but the selector failed to retain it in the
two submitted attempts.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

from experiments.candidate_records import CandidateRecord, grid_hash, normalize_grid
from experiments.pass2_selector import select_pass2


@dataclass(frozen=True)
class CoverageRecovery:
    """Output-weighted and task-weighted replay diagnostics."""

    total_positions: int
    covered_positions: int
    selected_positions: int
    total_tasks: int
    fully_covered_tasks: int
    solved_tasks: int
    coverage_rate: float
    selector_recovery_rate: float
    output_score: float
    task_coverage_rate: float
    task_solve_rate: float


def evaluate_coverage_recovery(
    records: list[CandidateRecord],
    *,
    solutions: Mapping[str, list[Any]],
    family_priors: Mapping[str, float] | None = None,
    collapse_correlated: bool = False,
) -> CoverageRecovery:
    """Measure candidate recall and conditional pass@2 recovery on labeled data."""

    grouped: dict[tuple[str, int], list[CandidateRecord]] = defaultdict(list)
    for record in records:
        if record.task_id in solutions and 0 <= record.test_index < len(solutions[record.task_id]):
            grouped[(record.task_id, record.test_index)].append(record)

    total_positions = covered_positions = selected_positions = 0
    total_tasks = fully_covered_tasks = solved_tasks = 0
    for task_id, expected_outputs in solutions.items():
        if not expected_outputs:
            continue
        total_tasks += 1
        task_covered = True
        task_selected = True
        for index, expected in enumerate(expected_outputs):
            total_positions += 1
            truth = grid_hash(normalize_grid(expected))
            position_records = grouped[(task_id, index)]
            valid_hashes = {
                record.output_hash for record in position_records if record.hard_valid
            }
            covered = truth in valid_hashes
            selected_hashes = select_pass2(
                [record.as_selector_candidate() for record in position_records],
                family_priors=family_priors,
                collapse_correlated=collapse_correlated,
            )
            selected = truth in set(selected_hashes)
            covered_positions += int(covered)
            selected_positions += int(selected)
            task_covered = task_covered and covered
            task_selected = task_selected and selected
        fully_covered_tasks += int(task_covered)
        solved_tasks += int(task_selected)

    coverage_rate = covered_positions / total_positions if total_positions else 0.0
    recovery_rate = (
        selected_positions / covered_positions if covered_positions else 0.0
    )
    return CoverageRecovery(
        total_positions=total_positions,
        covered_positions=covered_positions,
        selected_positions=selected_positions,
        total_tasks=total_tasks,
        fully_covered_tasks=fully_covered_tasks,
        solved_tasks=solved_tasks,
        coverage_rate=coverage_rate,
        selector_recovery_rate=recovery_rate,
        output_score=(selected_positions / total_positions if total_positions else 0.0),
        task_coverage_rate=(
            fully_covered_tasks / total_tasks if total_tasks else 0.0
        ),
        task_solve_rate=solved_tasks / total_tasks if total_tasks else 0.0,
    )


if __name__ == "__main__":
    record = CandidateRecord.from_output(
        task_id="t", test_index=0, family="program", candidate_id="p", output=[[1]]
    )
    result = evaluate_coverage_recovery([record], solutions={"t": [[[1]]]})
    assert result.coverage_rate == 1.0 and result.selector_recovery_rate == 1.0
    print("coverage_recovery selftest: PASS")
