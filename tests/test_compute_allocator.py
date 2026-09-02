import unittest

from experiments.compute_allocator import (
    FamilyBudget,
    PosteriorFamily,
    beta_marginal_gain,
    greedy_plan,
    greedy_posterior_plan,
    marginal_discovery_gain,
)


class ComputeAllocatorTests(unittest.TestCase):
    def test_independent_marginal_gain_has_geometric_decay(self) -> None:
        self.assertAlmostEqual(marginal_discovery_gain(0.5, 0), 0.5)
        self.assertAlmostEqual(marginal_discovery_gain(0.5, 1), 0.25)

    def test_greedy_plan_respects_budget(self) -> None:
        plan = greedy_plan((
            FamilyBudget("cheap", 0.2, 1),
            FamilyBudget("slow", 0.6, 5),
        ), budget_seconds=3)
        self.assertEqual(plan, ("cheap", "cheap", "cheap"))

    def test_high_hit_rate_family_enters_when_affordable(self) -> None:
        plan = greedy_plan((
            FamilyBudget("cheap", 0.1, 1),
            FamilyBudget("strong", 0.8, 2),
        ), budget_seconds=2)
        self.assertEqual(plan, ("strong",))

    def test_invalid_family_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FamilyBudget("bad", 1.1, 1)
        with self.assertRaises(ValueError):
            FamilyBudget("bad", 0.1, 0)

    def test_beta_posterior_recovers_uniform_prior_mean(self) -> None:
        self.assertAlmostEqual(beta_marginal_gain(1.0, 1.0, 0), 0.5)
        self.assertAlmostEqual(beta_marginal_gain(2.0, 2.0, 0), 0.5)

    def test_posterior_planner_combines_evidence_and_cost(self) -> None:
        plan = greedy_posterior_plan((
            PosteriorFamily("untested", 0, 0, 1, max_candidates=1),
            PosteriorFamily("proven", 8, 2, 1, max_candidates=1),
        ), budget_seconds=2)
        self.assertEqual(plan, ("proven", "untested"))

    def test_invalid_posterior_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PosteriorFamily("bad", -1, 0, 1)
        with self.assertRaises(ValueError):
            beta_marginal_gain(0, 1, 0)


if __name__ == "__main__":
    unittest.main()
