"""Output-level value-per-second scheduling for an anytime ARC run.

The scorer awards points per test output, while the notebook has a hard wall
clock and four independent workers.  This module turns calibrated position
estimates into a deterministic non-preemptive task plan.  It does not infer
the estimates and must be fed leakage-safe fold statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Iterable


@dataclass(frozen=True)
class PositionValue:
    unresolved_mass: float
    novelty_rate: float
    selector_recovery: float = 1.0

    @property
    def expected_gain(self) -> float:
        return self.unresolved_mass * self.novelty_rate * self.selector_recovery

    def __post_init__(self) -> None:
        if any(not 0.0 <= value <= 1.0 for value in (
            self.unresolved_mass, self.novelty_rate, self.selector_recovery
        )):
            raise ValueError("position probabilities must lie in [0, 1]")


@dataclass(frozen=True)
class TaskValue:
    task_id: str
    positions: tuple[PositionValue, ...]
    cost_seconds: float

    @property
    def expected_gain(self) -> float:
        return fsum(position.expected_gain for position in self.positions)

    @property
    def gain_per_second(self) -> float:
        return self.expected_gain / self.cost_seconds

    def __post_init__(self) -> None:
        if not self.task_id or self.cost_seconds <= 0.0:
            raise ValueError("task id and cost must be positive")


@dataclass(frozen=True)
class ScheduledTask:
    task: TaskValue
    worker: int
    start_seconds: float


def greedy_task_plan(
    tasks: Iterable[TaskValue],
    *,
    worker_count: int = 4,
    worker_seconds: float,
) -> tuple[ScheduledTask, ...]:
    """Pack tasks by expected output gain per second under a wall-clock cap."""

    if worker_count < 1 or worker_seconds <= 0.0:
        raise ValueError("worker_count and worker_seconds must be positive")
    ordered = sorted(
        tasks,
        key=lambda task: (
            -task.gain_per_second, -task.expected_gain,
            task.cost_seconds, task.task_id,
        ),
    )
    loads = [0.0] * worker_count
    result: list[ScheduledTask] = []
    for task in ordered:
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        if loads[worker] + task.cost_seconds > worker_seconds:
            continue
        result.append(ScheduledTask(task, worker, loads[worker]))
        loads[worker] += task.cost_seconds
    return tuple(result)


if __name__ == "__main__":
    plan = greedy_task_plan(
        [TaskValue("a", (PositionValue(1.0, 1.0),), 2.0)],
        worker_seconds=3.0,
    )
    assert plan and plan[0].worker == 0
    print("task_value_scheduler selftest: PASS")
