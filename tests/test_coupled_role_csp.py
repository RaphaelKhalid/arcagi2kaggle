from __future__ import annotations

import unittest

from experiments.coupled_role_csp import dataset_coupled_profile


class CoupledRoleCspTests(unittest.TestCase):
    def test_no_move_task_is_not_promoted(self) -> None:
        task = {"train": [{
            "input": [[0, 1]],
            "output": [[0, 1]],
        }]}
        self.assertEqual(dataset_coupled_profile({"x": task})["compiled_tasks"], 0)

    def test_unmatched_addition_is_not_promoted(self) -> None:
        task = {"train": [{
            "input": [[0, 1]],
            "output": [[0, 1, 2]],
        }]}
        self.assertEqual(dataset_coupled_profile({"x": task})["compiled_tasks"], 0)


if __name__ == "__main__":
    unittest.main()
