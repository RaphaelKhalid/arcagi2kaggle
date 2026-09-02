from __future__ import annotations

import unittest

from experiments.repair_prompt import build_repair_prompt, render_grid


class RepairPromptTests(unittest.TestCase):
    def test_prompt_contains_only_bounded_counterexample_context(self):
        prompt = build_repair_prompt("def transform(grid):\n    return grid", 2, [[0]], [[1]])
        self.assertIn("demonstration index 2", prompt)
        self.assertIn("Observed candidate output:\n0", prompt)
        self.assertIn("Required demonstration output:\n1", prompt)
        self.assertIn("hidden-test data", prompt)

    def test_exception_observation_is_explicit(self):
        prompt = build_repair_prompt("def transform(grid):\n    raise ValueError()", 1, None, [[2]])
        self.assertIn("execution raised an exception", prompt)

    def test_grid_and_source_bounds_are_enforced(self):
        self.assertEqual(render_grid([[1, 2], [3, 4]]), "12\n34")
        with self.assertRaises(ValueError):
            render_grid([[0] * 31] * 30)
        with self.assertRaises(ValueError):
            build_repair_prompt("x" * 13, 0, [[0]], [[0]], max_source_chars=12)

    def test_invalid_context_is_rejected(self):
        with self.assertRaises(ValueError):
            build_repair_prompt("def transform(grid):\n    return grid", -1, [[0]], [[0]])
        with self.assertRaises(ValueError):
            render_grid([[10]])


if __name__ == "__main__":
    unittest.main()
