from __future__ import annotations

import unittest

from experiments.equation_profile import task_equation_profile


class EquationProfileTests(unittest.TestCase):
    def test_reference_relative_fit_survives_layout_change(self) -> None:
        task = {"train": [
            {"input": [[0, 1, 0, 0, 2]],
             "output": [[0, 0, 1, 0, 2]]},
            {"input": [[1, 0, 0, 2, 0]],
             "output": [[0, 1, 0, 2, 0]]},
        ]}
        result = task_equation_profile(task)
        self.assertTrue(result["eligible"])
        self.assertTrue(result["reference_relative"])

    def test_no_stationary_reference_is_ineligible(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0]],
            "output": [[0, 0, 1]],
        }]}
        self.assertFalse(task_equation_profile(task)["eligible"])


if __name__ == "__main__":
    unittest.main()
