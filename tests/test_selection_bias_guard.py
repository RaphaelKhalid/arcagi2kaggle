import unittest

from experiments.selection_bias_guard import (
    two_attempt_action_count,
    uniform_pass2_lower_bound,
)


class SelectionBiasGuardTests(unittest.TestCase):
    def test_action_count_includes_singletons_and_pairs(self) -> None:
        self.assertEqual(two_attempt_action_count(1), 1)
        self.assertEqual(two_attempt_action_count(3), 6)

    def test_more_searched_classes_pay_a_larger_penalty(self) -> None:
        small = uniform_pass2_lower_bound(9, 10, 2)
        large = uniform_pass2_lower_bound(9, 10, 20)
        self.assertGreater(small, large)

    def test_bound_is_zero_safe(self) -> None:
        self.assertEqual(uniform_pass2_lower_bound(0, 10, 4), 0.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            two_attempt_action_count(0)
        with self.assertRaises(ValueError):
            uniform_pass2_lower_bound(1, 0, 2)
        with self.assertRaises(ValueError):
            uniform_pass2_lower_bound(1, 2, 2, delta=1.0)


if __name__ == "__main__":
    unittest.main()
