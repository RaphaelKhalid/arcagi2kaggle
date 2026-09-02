from __future__ import annotations

import unittest

from experiments.reflective_budget import (
    independent_pair_probability,
    paired_success_probability,
    second_stage_decision,
)


class ReflectiveBudgetTests(unittest.TestCase):
    def test_equal_cost_repair_wins_exactly_when_q_exceeds_p(self):
        self.assertFalse(second_stage_decision(
            0.4, 0.4, fresh_seconds=1, repair_seconds=1
        ).use_repair)
        self.assertTrue(second_stage_decision(
            0.4, 0.41, fresh_seconds=1, repair_seconds=1
        ).use_repair)

    def test_cost_adjusted_marginal_rule(self):
        decision = second_stage_decision(
            0.3, 0.45, fresh_seconds=1, repair_seconds=2
        )
        self.assertFalse(decision.use_repair)
        self.assertAlmostEqual(decision.repair_gain_per_second, 0.225)

    def test_correlated_repairs_are_discounted(self):
        decision = second_stage_decision(
            0.3, 0.8, fresh_seconds=1, repair_seconds=1, repair_novelty=0.25
        )
        self.assertEqual(decision.repair_gain, 0.2)
        self.assertFalse(decision.use_repair)

    def test_pair_formula_matches_theorem(self):
        p, q = 0.2, 0.6
        self.assertAlmostEqual(paired_success_probability(p, q), 0.68)
        self.assertAlmostEqual(independent_pair_probability(p), 0.36)

    def test_invalid_rates_and_costs_are_rejected(self):
        with self.assertRaises(ValueError):
            second_stage_decision(-0.1, 0.2, fresh_seconds=1, repair_seconds=1)
        with self.assertRaises(ValueError):
            second_stage_decision(0.1, 1.2, fresh_seconds=1, repair_seconds=1)
        with self.assertRaises(ValueError):
            second_stage_decision(0.1, 0.2, fresh_seconds=1, repair_seconds=1,
                                  repair_novelty=1.1)
        with self.assertRaises(ValueError):
            second_stage_decision(0.1, 0.2, fresh_seconds=0, repair_seconds=1)


if __name__ == "__main__":
    unittest.main()
