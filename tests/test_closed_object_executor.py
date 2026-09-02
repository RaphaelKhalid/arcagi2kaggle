from __future__ import annotations

import unittest

from experiments.closed_object_executor import (
    execute_closed_program,
    verified_closed_programs,
)


class ClosedObjectExecutorTests(unittest.TestCase):
    def test_verified_move_executes(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0], [0, 0, 0]],
            "output": [[0, 0, 1], [0, 0, 0]],
        }]}
        programs = verified_closed_programs(task)
        self.assertTrue(programs)
        self.assertEqual(
            execute_closed_program(programs[0], [[0, 1, 0], [0, 0, 0]]),
            ((0, 0, 1), (0, 0, 0)),
        )

    def test_verified_recolor_executes(self) -> None:
        task = {"train": [{
            "input": [[0, 1], [0, 0]],
            "output": [[0, 2], [0, 0]],
        }]}
        self.assertTrue(verified_closed_programs(task))

    def test_verified_delete_executes(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0]],
            "output": [[0, 0, 0]],
        }]}
        self.assertTrue(verified_closed_programs(task))

    def test_fixed_addition_is_closed(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0, 0, 0]],
            "output": [[0, 1, 0, 2, 0]],
        }]}
        self.assertTrue(verified_closed_programs(task))

    def test_shape_transform_is_rejected(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0], [0, 0, 0]],
            "output": [[0, 1, 1], [0, 0, 0]],
        }]}
        self.assertFalse(verified_closed_programs(task))

    def test_shape_transform_is_available_only_in_expanded_language(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0], [0, 0, 0]],
            "output": [[0, 1, 1], [0, 0, 0]],
        }]}
        programs = verified_closed_programs(task, allow_shape_transform=True)
        self.assertTrue(programs)
        self.assertEqual(
            execute_closed_program(programs[0], task["train"][0]["input"]),
            ((0, 1, 1), (0, 0, 0)),
        )

    def test_cross_demo_lookahead_mode_is_opt_in(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0], [0, 0, 0]],
            "output": [[0, 0, 1], [0, 0, 0]],
        }]}
        baseline = verified_closed_programs(task)
        lookahead = verified_closed_programs(task, minimum_cost_only=False)
        self.assertGreaterEqual(len(lookahead), len(baseline))


if __name__ == "__main__":
    unittest.main()
