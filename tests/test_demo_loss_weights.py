import unittest
from math import fsum

from experiments.demo_loss_weights import demo_loss_weights


class DemoLossWeightTests(unittest.TestCase):
    def test_demo_balancing_gives_equal_span_mass(self) -> None:
        weights = demo_loss_weights((2, 4), demo_balance=1.0)
        self.assertAlmostEqual(fsum(weights[0]), fsum(weights[1]))
        self.assertAlmostEqual(fsum(map(fsum, weights)), 1.0)

    def test_token_balancing_is_proportional_to_length(self) -> None:
        weights = demo_loss_weights((2, 4), demo_balance=0.0)
        self.assertAlmostEqual(fsum(weights[1]), 2.0 * fsum(weights[0]))

    def test_interpolation_preserves_normalization(self) -> None:
        for balance in (0.0, 0.25, 0.5, 1.0):
            weights = demo_loss_weights((1, 3, 5), demo_balance=balance)
            self.assertAlmostEqual(fsum(map(fsum, weights)), 1.0)

    def test_invalid_lengths_and_balance_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            demo_loss_weights(())
        with self.assertRaises(ValueError):
            demo_loss_weights((2, 0))
        with self.assertRaises(ValueError):
            demo_loss_weights((2,), demo_balance=1.1)


if __name__ == "__main__":
    unittest.main()
