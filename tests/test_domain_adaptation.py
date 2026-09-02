import unittest

from experiments.domain_adaptation import (
    GroupOutcome,
    effective_sample_size,
    importance_weights,
    smoothed_distribution,
    target_success_rate,
)


class DomainAdaptationTests(unittest.TestCase):
    def test_smoothed_distribution_is_normalized(self) -> None:
        distribution = smoothed_distribution({"a": 3, "b": 1})
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertGreater(distribution["a"], distribution["b"])

    def test_target_shift_changes_importance_weights(self) -> None:
        weights = importance_weights({"a": 9, "b": 1}, {"a": 1, "b": 9})
        self.assertGreater(weights["b"], weights["a"])
        self.assertLessEqual(max(weights.values()), 5.0)

    def test_unseen_target_group_falls_back_to_pooled_rate(self) -> None:
        estimate = target_success_rate(
            {"a": GroupOutcome(8, 2)}, {"a": 1, "b": 1}
        )
        self.assertGreater(estimate, 0.5)
        self.assertLess(estimate, 1.0)

    def test_groupwise_beta_estimate_is_target_weighted(self) -> None:
        estimate = target_success_rate(
            {"a": GroupOutcome(10, 0), "b": GroupOutcome(0, 10)},
            {"a": 1, "b": 3},
        )
        self.assertLess(estimate, 0.5)

    def test_effective_sample_size_detects_concentrated_weights(self) -> None:
        self.assertAlmostEqual(effective_sample_size({"a": 1, "b": 1}), 2.0)
        self.assertLess(effective_sample_size({"a": 5, "b": 0}), 2.0)


if __name__ == "__main__":
    unittest.main()
