from __future__ import annotations

import unittest

from experiments.holistic_context import TraceEvidence, select_trace_evidence


class HolisticContextTests(unittest.TestCase):
    def test_every_output_class_gets_a_witness_before_flood_fill(self):
        evidence = [
            TraceEvidence("a1", "a", "same", "a" * 40, lineage="l1"),
            TraceEvidence("a2", "a", "same", "a" * 40, lineage="l1"),
            TraceEvidence("b1", "b", "rare", "b" * 40, lineage="l2"),
        ]
        result = select_trace_evidence(evidence, max_chars=400, per_trace_chars=40)
        self.assertEqual(result.covered_output_hashes, ("a", "b"))
        self.assertEqual({item.output_hash for item in result.selected}, {"a", "b"})

    def test_remaining_budget_prefers_new_family_and_lineage(self):
        evidence = [
            TraceEvidence("a1", "a", "f1", "x", lineage="l1"),
            TraceEvidence("b1", "b", "f1", "x", lineage="l1"),
            TraceEvidence("b2", "b", "f2", "x", lineage="l2"),
        ]
        result = select_trace_evidence(evidence, max_chars=230, per_trace_chars=1)
        self.assertEqual(
            {item.candidate_id for item in result.selected},
            {"a1", "b1", "b2"},
        )

    def test_too_small_budget_fails_closed(self):
        evidence = [
            TraceEvidence("a", "a", "f", "trace"),
            TraceEvidence("b", "b", "f", "trace"),
        ]
        with self.assertRaises(ValueError):
            select_trace_evidence(evidence, max_chars=10, per_trace_chars=5)

    def test_required_class_without_trace_fails_closed(self):
        with self.assertRaises(ValueError):
            select_trace_evidence(
                [TraceEvidence("a", "a", "f", "trace")],
                max_chars=100,
                required_output_hashes=("a", "missing"),
            )


if __name__ == "__main__":
    unittest.main()
