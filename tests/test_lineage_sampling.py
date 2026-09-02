import unittest

from experiments.lineage_sampling import (
    balanced_lineage_plan,
    lineage_effective_sample_size,
    max_lineage_ess,
)


class LineageSamplingTests(unittest.TestCase):
    def test_independent_samples_recover_total_count(self):
        self.assertAlmostEqual(
            lineage_effective_sample_size((2, 3, 3), rho=0.0), 8.0
        )

    def test_perfect_correlation_recovers_lineage_count(self):
        self.assertAlmostEqual(
            lineage_effective_sample_size((2, 2, 2), rho=1.0), 3.0
        )

    def test_balancing_maximizes_ess_at_fixed_lineage_count(self):
        balanced = balanced_lineage_plan(8, 4, rho=0.6)
        imbalanced = lineage_effective_sample_size((5, 1, 1, 1), rho=0.6)
        self.assertEqual(balanced.sample_counts, (2, 2, 2, 2))
        self.assertGreater(balanced.effective_sample_size, imbalanced)

    def test_one_lineage_per_sample_is_ess_upper_bound(self):
        plan = max_lineage_ess(8, rho=0.9)
        self.assertEqual(plan.sample_counts, (1,) * 8)
        self.assertAlmostEqual(plan.effective_sample_size, 8.0)

    def test_invalid_allocations_are_rejected(self):
        with self.assertRaises(ValueError):
            balanced_lineage_plan(3, 4)
        with self.assertRaises(ValueError):
            lineage_effective_sample_size((2, 0), rho=0.5)
        with self.assertRaises(ValueError):
            lineage_effective_sample_size((1, 1), rho=1.1)
