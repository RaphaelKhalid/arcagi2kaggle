from __future__ import annotations

import unittest

from experiments.relational_families import (
    relational_action_families,
    task_relational_consensus,
)


class RelationalFamilyTests(unittest.TestCase):
    def test_move_is_quotiented_to_axis(self) -> None:
        families = relational_action_families(
            [[0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0]],
        )
        self.assertTrue(families)
        self.assertEqual(families[0][0][0], "move")
        self.assertEqual(families[0][0][4], "horizontal")

    def test_consensus_accepts_coarse_same_rule(self) -> None:
        task = {
            "train": [
                {"input": [[0, 1], [0, 0]], "output": [[0, 2], [0, 0]]},
                {"input": [[0, 1, 0], [0, 0, 0]],
                 "output": [[0, 2, 0], [0, 0, 0]]},
            ]
        }
        result = task_relational_consensus(task)
        self.assertTrue(result["stable_families"])

    def test_empty_task_is_explicit(self) -> None:
        result = task_relational_consensus({"train": []})
        self.assertEqual(result["n_demos"], 0)
        self.assertFalse(result["skipped"])


if __name__ == "__main__":
    unittest.main()
