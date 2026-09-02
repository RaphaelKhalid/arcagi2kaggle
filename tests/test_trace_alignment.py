from __future__ import annotations

import unittest

from experiments.graph_lgg import ActionObservation
from experiments.trace_alignment import (
    aligned_lgg_candidates,
    align_traces,
)


def _obs(kind: str, axis: str) -> ActionObservation:
    return ActionObservation(
        kind, axis, "same", "same", "same",
        frozenset({("area", 1)}), frozenset({("area", 1)}),
    )


class TraceAlignmentTests(unittest.TestCase):
    def test_assignment_recovers_permuted_actions(self) -> None:
        reference = (_obs("move", "horizontal"), _obs("recolor", "none"))
        other = (_obs("recolor", "none"), _obs("move", "horizontal"))
        alignment = align_traces(reference, other)[0]
        self.assertEqual(alignment.pairs, ((0, 1), (1, 0)))
        self.assertFalse(alignment.unmatched_reference)
        self.assertFalse(alignment.unmatched_other)

    def test_aligned_lgg_handles_permutation(self) -> None:
        reference = (_obs("move", "horizontal"), _obs("recolor", "none"))
        other = (_obs("recolor", "none"), _obs("move", "horizontal"))
        schemas = aligned_lgg_candidates((reference, other))
        self.assertTrue(schemas)
        self.assertEqual(
            tuple(schema.kind for schema in schemas[0]),
            ("move", "recolor"),
        )

    def test_unmatched_clause_blocks_full_lgg(self) -> None:
        self.assertFalse(aligned_lgg_candidates(
            ((_obs("move", "horizontal"),),
             (_obs("move", "horizontal"), _obs("delete", "none")))
        ))


if __name__ == "__main__":
    unittest.main()
