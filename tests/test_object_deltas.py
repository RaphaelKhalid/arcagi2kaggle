from __future__ import annotations

import unittest

from experiments.object_deltas import (
    classify_delta,
    extract_objects,
    task_consistency_profile,
)


class ObjectDeltaTests(unittest.TestCase):
    def test_recolor_and_move_are_separated(self) -> None:
        self.assertIn("object_recolor", classify_delta(
            [[0, 1, 0], [0, 0, 0]], [[0, 2, 0], [0, 0, 0]]
        ))
        self.assertIn("object_move", classify_delta(
            [[0, 1, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0]]
        ))

    def test_object_extraction_ignores_absolute_anchor(self) -> None:
        left = extract_objects([[0, 1], [0, 0]])[0]
        right = extract_objects([[0, 0], [1, 0]])[0]
        self.assertEqual(left.shape, right.shape)
        self.assertNotEqual(left.anchor, right.anchor)

    def test_task_consistency_requires_all_demos(self) -> None:
        challenges = {
            "a": {"train": [
                {"input": [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                 "output": [[0, 2, 0], [0, 0, 0], [0, 0, 0]]},
                {"input": [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
                 "output": [[0, 2, 0], [0, 0, 0], [0, 0, 0]]},
            ]}
        }
        profile = task_consistency_profile(challenges)
        self.assertEqual(profile["object_recolor"], 1)


if __name__ == "__main__":
    unittest.main()
