from __future__ import annotations

import unittest

from experiments.task_value_scheduler import (
    PositionValue,
    TaskValue,
    greedy_task_plan,
)


class TaskValueSchedulerTests(unittest.TestCase):
    def test_output_level_value_adds_positions(self) -> None:
        task = TaskValue(
            "t",
            (PositionValue(0.5, 0.8), PositionValue(1.0, 0.25)),
            2.0,
        )
        self.assertAlmostEqual(task.expected_gain, 0.65)
        self.assertAlmostEqual(task.gain_per_second, 0.325)

    def test_scheduler_prefers_value_density_and_balances_workers(self) -> None:
        tasks = [
            TaskValue("dense", (PositionValue(1.0, 1.0),), 1.0),
            TaskValue("sparse", (PositionValue(0.1, 1.0),), 1.0),
            TaskValue("long", (PositionValue(1.0, 1.0),), 3.0),
        ]
        plan = greedy_task_plan(tasks, worker_count=2, worker_seconds=2.0)
        self.assertEqual([item.task.task_id for item in plan], ["dense", "sparse"])
        self.assertEqual([item.worker for item in plan], [0, 1])

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PositionValue(1.1, 0.0)


if __name__ == "__main__":
    unittest.main()
