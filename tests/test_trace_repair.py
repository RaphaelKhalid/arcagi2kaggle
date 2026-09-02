from __future__ import annotations

import unittest

from experiments.trace_repair import (
    first_divergence,
    normalize_state,
    repair_targets,
    summarize_grid_diff,
    state_hash,
)


class TraceRepairTests(unittest.TestCase):
    def test_matching_traces_have_no_divergence(self) -> None:
        trace = [[[0]], [[1]]]
        self.assertIsNone(first_divergence(trace, trace))
        self.assertEqual(state_hash(normalize_state([[0]])),
                         state_hash(((0,),)))

    def test_first_divergence_maps_to_its_causal_operation(self) -> None:
        self.assertEqual(
            first_divergence([[[0]], [[1]], [[1]]], [[[0]], [[1]], [[2]]]), 2
        )
        self.assertEqual(repair_targets(2, ["paint", "move"]), ("move",))

    def test_input_and_trace_length_mismatches_are_not_hidden(self) -> None:
        self.assertEqual(repair_targets(0, ["paint"]), ("input",))
        self.assertEqual(repair_targets(2, ["paint"]), ("trace_length",))

    def test_grid_diff_reports_coordinates_and_bounds(self) -> None:
        diff = summarize_grid_diff([[0, 1], [2, 0]], [[0, 0, 3]])
        self.assertEqual(diff.changed_count, 4)
        self.assertEqual(diff.changed_cells[0], (0, 1, 1, 0))
        self.assertIn("shape observed=2x2 expected=1x3", diff.render())

    def test_grid_diff_is_bounded(self) -> None:
        diff = summarize_grid_diff(
            [[1, 1], [1, 1]], [[0, 0], [0, 0]], max_changes=2
        )
        self.assertEqual(len(diff.changed_cells), 2)
        self.assertTrue(diff.truncated)


if __name__ == "__main__":
    unittest.main()
