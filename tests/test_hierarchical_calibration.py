import unittest

from experiments.hierarchical_calibration import (
    Outcome,
    hierarchical_rate,
    posterior_mean,
)


class HierarchicalCalibrationTests(unittest.TestCase):
    def test_beta_posterior_is_bounded(self) -> None:
        self.assertGreater(posterior_mean(Outcome(0, 0)), 0.0)
        self.assertLess(posterior_mean(Outcome(0, 0)), 1.0)

    def test_supported_feature_moves_global_rate(self) -> None:
        rate = hierarchical_rate(
            Outcome(5, 5), {("shape", "small"): Outcome(8, 2)},
            {"shape": "small"},
        )
        self.assertGreater(rate, 0.5)

    def test_sparse_feature_is_shrunk_toward_global(self) -> None:
        rate = hierarchical_rate(
            Outcome(50, 50), {("shape", "rare"): Outcome(1, 0)},
            {"shape": "rare"}, shrinkage=100.0,
        )
        self.assertGreater(rate, 0.49)
        self.assertLess(rate, 0.55)

    def test_missing_feature_uses_global_posterior(self) -> None:
        self.assertAlmostEqual(
            hierarchical_rate(Outcome(7, 3), {}, {"shape": "unknown"}),
            posterior_mean(Outcome(7, 3),
                           prior_alpha=1.0, prior_beta=1.0),
        )

    def test_invalid_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            hierarchical_rate(Outcome(1, 1), {}, {}, shrinkage=0.0)
        with self.assertRaises(ValueError):
            posterior_mean(Outcome(1, 1), prior_alpha=0.0)


if __name__ == "__main__":
    unittest.main()
