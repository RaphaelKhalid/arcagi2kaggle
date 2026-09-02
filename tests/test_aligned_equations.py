from __future__ import annotations

import unittest

from experiments.aligned_equations import task_aligned_equation_profile


class AlignedEquationTests(unittest.TestCase):
    def test_move_has_grounded_constant_equation(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0]],
            "output": [[0, 0, 1]],
        }]}
        result = task_aligned_equation_profile(task)
        self.assertEqual(result["with_equations"], 1)

    def test_shape_change_has_no_equation(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0]],
            "output": [[0, 1, 1]],
        }]}
        result = task_aligned_equation_profile(task)
        self.assertEqual(result["with_equations"], 0)


if __name__ == "__main__":
    unittest.main()
