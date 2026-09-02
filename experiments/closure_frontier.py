"""Find the frontier between the verified object language and ARC deltas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from experiments.closed_object_executor import verified_closed_programs
from experiments.object_deltas import classify_delta, task_delta_profile


SUPPORTED_DELTA_LABELS = frozenset({
    "identity", "object_move", "object_recolor", "object_add", "object_delete",
})


@dataclass(frozen=True)
class ClosureFrontier:
    """A task-level closure diagnostic, independent of hidden labels."""

    task_id: str
    observed_labels: frozenset[str]
    unsupported_labels: frozenset[str]
    verified_program_count: int
    status: str

    @property
    def language_closed(self) -> bool:
        return not self.unsupported_labels

    @property
    def verified(self) -> bool:
        return self.verified_program_count > 0


def inspect_task(
    task_id: str,
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> ClosureFrontier:
    """Classify whether the current closed language can explain a task."""

    pairs = task.get("train", [])
    if not pairs:
        return ClosureFrontier(task_id, frozenset(), frozenset(), 0, "empty")
    observed = frozenset(task_delta_profile(task))
    unsupported = observed - SUPPORTED_DELTA_LABELS
    program_count = len(verified_closed_programs(
        task, k=k, max_objects=max_objects
    ))
    if unsupported:
        status = "language_gap"
    elif program_count:
        status = "closed_verified"
    else:
        status = "search_gap"
    return ClosureFrontier(task_id, observed, unsupported, program_count, status)


def inspect_dataset(
    challenges: Mapping[str, Mapping[str, Any]],
    *,
    k: int = 4,
    max_objects: int = 10,
) -> tuple[ClosureFrontier, ...]:
    """Return deterministic closure diagnostics in input mapping order."""

    return tuple(
        inspect_task(task_id, task, k=k, max_objects=max_objects)
        for task_id, task in challenges.items()
    )


if __name__ == "__main__":
    task = {"train": [{"input": [[0, 1]], "output": [[0, 2]]}]}
    result = inspect_task("demo", task)
    assert result.status == "closed_verified"
    print("closure_frontier selftest: PASS")
