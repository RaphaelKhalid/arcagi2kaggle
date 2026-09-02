from __future__ import annotations

import unittest

from experiments.aligned_clauses import (
    aligned_local_hypotheses,
    task_aligned_proof_profile,
)


class AlignedClauseTests(unittest.TestCase):
    def test_local_indices_survive_alignment(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0, 0, 2]],
            "output": [[0, 0, 1, 0, 2]],
        }]}
        hypotheses = aligned_local_hypotheses(task)
        self.assertTrue(hypotheses)
        self.assertEqual(
            {item.source_index for item in hypotheses[0].traces[0]},
            {0, 1},
        )

    def test_constant_move_can_be_grounded(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0]],
            "output": [[0, 0, 1]],
        }]}
        result = task_aligned_proof_profile(task)
        self.assertEqual(result["grounded_hypotheses"], 1)

    def test_shape_transform_is_not_grounded(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0]],
            "output": [[0, 1, 1]],
        }]}
        result = task_aligned_proof_profile(task)
        self.assertEqual(result["grounded_hypotheses"], 0)


if __name__ == "__main__":
    unittest.main()
