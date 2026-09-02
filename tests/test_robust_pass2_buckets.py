import unittest

from experiments.robust_pass2_buckets import (
    robust_pair_mass,
    select_robust_pass2,
)


class RobustPass2BucketTests(unittest.TestCase):
    def test_bucketwise_tv_bound_mixes_conditional_pair_mass(self) -> None:
        lower = robust_pair_mass(
            {"small": {"a": 0.8, "b": 0.2}, "large": {"a": 0.1, "c": 0.9}},
            {"small": 0.75, "large": 0.25},
            ("a", "b"),
            shift_radius={"small": 0.1, "large": 0.2},
        )
        self.assertAlmostEqual(lower, 0.75 * 0.9 + 0.25 * 0.0)

    def test_unseen_target_bucket_is_zero_not_pooled(self) -> None:
        lower = robust_pair_mass(
            {"small": {"a": 1.0}},
            {"small": 0.5, "unseen": 0.5},
            ("a",),
        )
        self.assertAlmostEqual(lower, 0.5)

    def test_selector_can_prefer_complementary_pair(self) -> None:
        selected = select_robust_pass2(
            {"x": {"a": 0.9, "b": 0.1}, "y": {"a": 0.1, "b": 0.9}},
            {"x": 0.5, "y": 0.5},
        )
        self.assertEqual(selected, ("a", "b"))

    def test_empty_candidates_and_invalid_inputs(self) -> None:
        self.assertEqual(select_robust_pass2({}, {"x": 1.0}), ())
        with self.assertRaises(ValueError):
            robust_pair_mass({"x": {"a": 0.5}}, {"x": 1.0}, ("a",))
        with self.assertRaises(ValueError):
            robust_pair_mass({"x": {"a": 1.0}}, {"x": 1.0}, ())
        with self.assertRaises(ValueError):
            robust_pair_mass({"x": {"a": 1.0}}, {"x": 1.0}, ("a",),
                             shift_radius={"x": 1.1})


if __name__ == "__main__":
    unittest.main()
