import unittest

from experiments.shift_robust_router import (
    ShiftRobustLane,
    groupwise_target_lower_rate,
    rank_shift_robust_lanes,
    wilson_lower_bound,
)


class ShiftRobustRouterTests(unittest.TestCase):
    def test_wilson_bound_is_conservative_and_zero_safe(self) -> None:
        self.assertEqual(wilson_lower_bound(0, 0), 0.0)
        lower = wilson_lower_bound(8, 10)
        self.assertGreater(lower, 0.25)
        self.assertLess(lower, 0.8)

    def test_more_evidence_tightens_the_same_rate_bound(self) -> None:
        self.assertGreater(
            wilson_lower_bound(80, 100), wilson_lower_bound(8, 10)
        )

    def test_shift_radius_can_only_reduce_target_lower_bound(self) -> None:
        source = ShiftRobustLane("a", 8, 10, 1.0)
        shifted = ShiftRobustLane("b", 8, 10, 1.0, shift_radius=0.2)
        self.assertLess(shifted.target_lower_rate, source.target_lower_rate)
        floor = ShiftRobustLane("c", 0, 1, 1.0, shift_radius=1.0)
        self.assertEqual(floor.target_lower_rate, 0.0)

    def test_routing_prefers_conservative_gain_per_second(self) -> None:
        route = rank_shift_robust_lanes([
            ShiftRobustLane("slow", 9, 10, 10.0),
            ShiftRobustLane("fast", 8, 10, 1.0),
        ], unresolved_mass=0.5)
        self.assertEqual(route.lane_names[0], "fast")
        self.assertGreater(route.lower_gain_per_second[0],
                           route.lower_gain_per_second[1])

    def test_invalid_counts_and_rates_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            wilson_lower_bound(2, 1)
        with self.assertRaises(ValueError):
            ShiftRobustLane("", 0, 1, 1.0)
        with self.assertRaises(ValueError):
            ShiftRobustLane("a", 0, 1, 1.0, shift_radius=1.1)

    def test_unseen_target_bucket_receives_zero_lower_mass(self) -> None:
        lane = ShiftRobustLane("small", 8, 10, 1.0)
        weighted = groupwise_target_lower_rate(
            {"small": lane}, {"small": 0.75, "unseen": 0.25}
        )
        self.assertAlmostEqual(weighted, 0.75 * lane.target_lower_rate)

    def test_single_bucket_matches_its_lane_bound(self) -> None:
        lane = ShiftRobustLane("all", 8, 10, 1.0, shift_radius=0.1)
        self.assertAlmostEqual(
            groupwise_target_lower_rate({"all": lane}, {"all": 1.0}),
            lane.target_lower_rate,
        )

    def test_invalid_target_mass_is_rejected(self) -> None:
        lane = ShiftRobustLane("a", 1, 2, 1.0)
        with self.assertRaises(ValueError):
            groupwise_target_lower_rate({"a": lane}, {})
        with self.assertRaises(ValueError):
            groupwise_target_lower_rate({"a": lane}, {"a": -0.1, "b": 1.1})
        with self.assertRaises(ValueError):
            groupwise_target_lower_rate({"a": lane}, {"a": 0.9})


if __name__ == "__main__":
    unittest.main()
