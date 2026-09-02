from __future__ import annotations

import unittest

from experiments.object_correspondence import top_k_correspondences


class ObjectCorrespondenceTests(unittest.TestCase):
    def test_moved_object_is_matched_globally(self) -> None:
        result = top_k_correspondences(
            [[0, 1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0]],
        )
        self.assertEqual(result[0].pairs, ((0, 0),))
        self.assertEqual(result[0].unmatched_source, ())
        self.assertEqual(result[0].unmatched_target, ())

    def test_unmatched_objects_are_explicit(self) -> None:
        result = top_k_correspondences([[0, 1, 0, 0, 0]], [[0, 1, 0, 2, 0]])
        self.assertTrue(result)
        self.assertEqual(len(result[0].unmatched_target), 1)

    def test_ambiguous_assignments_are_retained(self) -> None:
        result = top_k_correspondences(
            [[0, 1, 0, 1]], [[0, 2, 0, 2]], k=4
        )
        self.assertGreaterEqual(len(result), 2)

    def test_object_limit_is_explicit(self) -> None:
        with self.assertRaises(ValueError):
            top_k_correspondences([[0, 1, 0, 2, 0]], [[0, 1, 0, 2, 0]], max_objects=1)


if __name__ == "__main__":
    unittest.main()
