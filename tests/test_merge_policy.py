from __future__ import annotations

import unittest

from experiments.merge_policy import gated_pair_merge


class MergePolicyTests(unittest.TestCase):
    def test_weak_verified_candidate_does_not_displace_baseline(self):
        decision = gated_pair_merge(
            ("base-a", "base-b"),
            {"base-a": 0.45, "base-b": 0.40, "verified": 0.10},
        )
        self.assertEqual(decision.attempts, ("base-a", "base-b"))
        self.assertFalse(decision.promoted)

    def test_stronger_candidate_replaces_weakest_slot(self):
        decision = gated_pair_merge(
            ("base-a", "base-b"),
            {"base-a": 0.45, "base-b": 0.20, "verified": 0.40},
        )
        self.assertEqual(decision.attempts, ("base-a", "verified"))
        self.assertTrue(decision.promoted)

    def test_ties_preserve_baseline(self):
        decision = gated_pair_merge(
            ("base-a", "base-b"),
            {"base-a": 0.5, "base-b": 0.2, "verified": 0.2},
        )
        self.assertEqual(decision.attempts, ("base-a", "base-b"))


if __name__ == "__main__":
    unittest.main()
