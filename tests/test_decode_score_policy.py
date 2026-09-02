import math
import unittest

from experiments.decode_score_policy import (
    DecodeScorePolicy,
    current_absolute_policy,
)


class DecodeScorePolicyTests(unittest.TestCase):
    def test_absolute_policy_matches_current_cutoff(self) -> None:
        policy = current_absolute_policy()
        self.assertAlmostEqual(policy.budget, -math.log(0.2))
        self.assertTrue(policy.accepts(1.0, 100))
        self.assertFalse(policy.accepts(policy.budget, 100))

    def test_absolute_policy_has_stricter_long_grid_requirement(self) -> None:
        policy = current_absolute_policy()
        short = policy.absolute_mean_likelihood_requirement(100)
        long = policy.absolute_mean_likelihood_requirement(900)
        self.assertLess(short, long)
        self.assertAlmostEqual(short, 0.9840344434, places=8)
        self.assertAlmostEqual(long, 0.9982133336, places=8)

    def test_mean_nll_is_length_normalized(self) -> None:
        policy = DecodeScorePolicy("mean_nll", 0.02)
        self.assertTrue(policy.accepts(1.0, 100))
        self.assertTrue(policy.accepts(9.0, 900))
        self.assertFalse(policy.accepts(2.0, 100))

    def test_mdl_penalty_and_partial_bound(self) -> None:
        policy = DecodeScorePolicy("mdl", 4.0, length_penalty=0.01)
        self.assertTrue(policy.accepts(2.0, 100))
        self.assertFalse(policy.accepts(3.0, 200))
        self.assertEqual(policy.optimistic_partial_score(1.0, 10, 100), 2.0)

    def test_invalid_policy_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DecodeScorePolicy("mean_nll", 0.0)
        with self.assertRaises(ValueError):
            DecodeScorePolicy("absolute", 1.0, length_penalty=0.1)
        with self.assertRaises(ValueError):
            DecodeScorePolicy("mdl", 1.0, length_penalty=-0.1)


if __name__ == "__main__":
    unittest.main()
