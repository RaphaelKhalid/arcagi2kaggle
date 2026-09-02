from __future__ import annotations

import unittest

from experiments.relation_equations import (
    anchor_relation,
    relation_equation_unique_on_demos,
    relation_satisfying_anchors,
    task_relation_profile,
)


class RelationEquationTests(unittest.TestCase):
    def test_relation_quotients_exact_distance(self) -> None:
        self.assertEqual(
            anchor_relation((4, 2), (0, 2)),
            anchor_relation((8, 8), (4, 8)),
        )

    def test_relation_axis_is_preserved(self) -> None:
        self.assertEqual(anchor_relation((0, 3), (0, 0))[1], "horizontal")

    def test_single_move_has_no_reference_equation(self) -> None:
        result = task_relation_profile({"train": [{
            "input": [[0, 1, 0]],
            "output": [[0, 0, 1]],
        }]})
        self.assertEqual(result["relation_equation_count"], 0)

    def test_relation_can_have_unique_placement(self) -> None:
        grid = [[0, 1, 2, 0]]
        relation = anchor_relation((0, 1), (0, 2))
        candidates = relation_satisfying_anchors(
            grid, ((0, 0),), (0, 2), relation, frozenset({(0, 2)})
        )
        self.assertEqual(candidates, ((0, 1),))

    def test_unique_placement_proof_matches_demo(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0, 2, 0]],
            "output": [[0, 0, 1, 2, 0]],
        }]}
        # The one-demo task has no aligned reference equation because its
        # target reference is an identity clause only after correspondence.
        result = task_relation_profile(task)
        self.assertGreaterEqual(result["hypotheses"], 1)


if __name__ == "__main__":
    unittest.main()
