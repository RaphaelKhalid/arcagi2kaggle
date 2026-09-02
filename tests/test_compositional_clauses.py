from __future__ import annotations

import unittest

from experiments.compositional_clauses import (
    ClauseFootprint,
    clause_footprints,
    footprints_are_disjoint,
    frame_condition_holds,
)
from experiments.object_correspondence import top_k_correspondences


class CompositionalClauseTests(unittest.TestCase):
    def test_frame_condition_covers_move(self) -> None:
        source = [[0, 1, 0, 0, 2]]
        target = [[0, 0, 1, 0, 2]]
        footprints = clause_footprints(
            source, target, top_k_correspondences(source, target)[0]
        )
        self.assertTrue(footprints_are_disjoint(footprints))
        self.assertTrue(frame_condition_holds(source, target, footprints))

    def test_disjointness_rejects_overlapping_footprints(self) -> None:
        footprints = (
            ClauseFootprint("move", frozenset({(0, 0)}), frozenset({(0, 1)}), 0, 0),
            ClauseFootprint("move", frozenset({(0, 0)}), frozenset({(0, 2)}), 1, 1),
        )
        self.assertFalse(footprints_are_disjoint(footprints))

    def test_frame_rejects_unexplained_change(self) -> None:
        source = [[0, 1, 0]]
        target = [[0, 2, 0]]
        footprints = (
            ClauseFootprint("move", frozenset({(0, 0)}), frozenset({(0, 0)}), 0, 0),
        )
        self.assertFalse(frame_condition_holds(source, target, footprints))


if __name__ == "__main__":
    unittest.main()
