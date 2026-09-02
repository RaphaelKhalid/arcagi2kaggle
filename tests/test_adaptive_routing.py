import unittest
from math import log

from experiments.adaptive_routing import (
    decide_route,
    decide_route_positionwise,
    distribution_stats,
)


class AdaptiveRoutingTests(unittest.TestCase):
    def test_entropy_detects_concentration(self):
        concentrated = distribution_stats({"a": 0.99, "b": 0.01})
        diffuse = distribution_stats({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25})
        self.assertLess(concentrated.entropy, diffuse.entropy)
        self.assertLess(concentrated.effective_classes, diffuse.effective_classes)

    def test_current_utility_is_top_two_mass(self):
        decision = decide_route(
            [{"a": 0.5, "b": 0.3, "c": 0.2}, {"x": 1.0}],
            novelty_rate=1.0, lane_seconds=1.0,
        )
        self.assertAlmostEqual(decision.current_utility, 1.8)
        self.assertAlmostEqual(decision.unresolved_mass, 0.2)

    def test_unknown_reserve_prevents_false_certainty(self):
        stats = distribution_stats({"wrong_consensus": 1.0}, unknown_mass=0.2)
        self.assertAlmostEqual(stats.top_two_mass, 0.8)
        self.assertAlmostEqual(stats.entropy, -(0.8 * log(0.8) + 0.2 * log(0.2)))
        decision = decide_route(
            [{"wrong_consensus": 1.0}],
            novelty_rate=1.0, lane_seconds=1.0, unknown_mass=0.2,
        )
        self.assertAlmostEqual(decision.unresolved_mass, 0.2)

    def test_unknown_reserve_can_be_position_specific(self):
        decision = decide_route(
            [{"a": 1.0}, {"b": 1.0}],
            novelty_rate=1.0, lane_seconds=1.0, unknown_mass=[0.1, 0.3],
        )
        self.assertAlmostEqual(decision.unresolved_mass, 0.4)

    def test_expected_gain_is_additive_across_test_outputs(self):
        decision = decide_route(
            [{"a": 0.5, "b": 0.3, "c": 0.2}] * 2,
            novelty_rate=0.5, selector_recovery=0.8, lane_seconds=10.0,
        )
        self.assertAlmostEqual(decision.expected_gain, 0.16)
        self.assertTrue(decision.continue_search)

    def test_threshold_can_stop_when_lane_is_not_worth_cost(self):
        decision = decide_route(
            [{"a": 0.99, "b": 0.01, "c": 0.0}],
            novelty_rate=1.0, lane_seconds=10.0, min_gain_per_second=0.001,
        )
        self.assertFalse(decision.continue_search)

    def test_positionwise_rates_spend_value_on_complementary_positions(self):
        decision = decide_route_positionwise(
            [{}, {}],
            novelty_rates=(1.0, 0.0),
            lane_seconds=1.0,
        )
        self.assertAlmostEqual(decision.unresolved_mass, 2.0)
        self.assertAlmostEqual(decision.expected_gain, 1.0)

    def test_positionwise_rate_lengths_are_guarded(self):
        with self.assertRaises(ValueError):
            decide_route_positionwise(
                [{"a": 1.0}], novelty_rates=(), lane_seconds=1.0
            )

    def test_invalid_rates_and_costs_are_rejected(self):
        with self.assertRaises(ValueError):
            decide_route([], novelty_rate=1.1, lane_seconds=1.0)
        with self.assertRaises(ValueError):
            decide_route([], novelty_rate=0.1, lane_seconds=0.0)
        with self.assertRaises(ValueError):
            decide_route(
                [{"a": 1.0}], novelty_rate=0.1, lane_seconds=1.0,
                unknown_mass=[0.1, 0.2],
            )
