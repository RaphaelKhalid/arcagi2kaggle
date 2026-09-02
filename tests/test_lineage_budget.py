import unittest

from experiments.lineage_budget import choose_lineage_budget


class LineageBudgetTests(unittest.TestCase):
    def test_high_correlation_justifies_more_lineages(self) -> None:
        independent = choose_lineage_budget(
            20.0, 1.0, 1.0, rho=0.0, max_lineages=20
        )
        correlated = choose_lineage_budget(
            20.0, 1.0, 1.0, rho=0.9, max_lineages=20
        )
        self.assertLessEqual(independent.lineage_count, correlated.lineage_count)

    def test_balancing_is_preserved(self) -> None:
        plan = choose_lineage_budget(
            11.0, 1.0, 1.0, rho=0.7, max_lineages=5
        )
        self.assertLessEqual(max(plan.sample_counts) - min(plan.sample_counts), 1)
        self.assertEqual(sum(plan.sample_counts), plan.total_samples)

    def test_zero_setup_cost_can_use_all_samples_without_correlation(self) -> None:
        plan = choose_lineage_budget(
            5.0, 1.0, 0.0, rho=0.0, max_lineages=5
        )
        self.assertEqual(plan.total_samples, 5)
        self.assertEqual(plan.lineage_count, 1)

    def test_invalid_budget_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            choose_lineage_budget(1.0, 2.0, 0.0)
        with self.assertRaises(ValueError):
            choose_lineage_budget(10.0, 1.0, 0.0, rho=1.1)


if __name__ == "__main__":
    unittest.main()
