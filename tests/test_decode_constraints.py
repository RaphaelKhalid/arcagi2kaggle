from __future__ import annotations

import unittest

from experiments.decode_constraints import (
    GridConstraints,
    infer_constraints,
    parse_view_operations,
)


class DecodeConstraintTests(unittest.TestCase):
    def test_shared_shape_and_palette_are_inferred(self) -> None:
        constraints = infer_constraints([
            [[0, 1], [1, 0]],
            [[1, 0], [0, 1]],
        ])
        self.assertEqual(constraints, GridConstraints(2, 2, frozenset({0, 1})))
        self.assertTrue(constraints.validate([[1, 0], [0, 1]]))
        self.assertFalse(constraints.validate([[1, 0, 0], [0, 1, 0]]))

    def test_only_shared_invariants_survive(self) -> None:
        constraints = infer_constraints([
            [[0, 1], [1, 0]],
            [[0, 2], [2, 0]],
        ])
        self.assertEqual((constraints.height, constraints.width), (2, 2))
        self.assertIsNone(constraints.palette)

    def test_view_transform_and_token_bound(self) -> None:
        constraints = GridConstraints(2, 3, frozenset({0, 1}))
        view = constraints.transformed(("transpose", "permute1023456789"))
        self.assertEqual(view, GridConstraints(3, 2, frozenset({0, 1})))
        self.assertEqual(view.max_new_tokens(), 11)
        self.assertEqual(
            parse_view_operations("task.transpose.permute1023456789"),
            ("transpose", "permute1023456789"),
        )


if __name__ == "__main__":
    unittest.main()
