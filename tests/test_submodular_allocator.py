import unittest

from experiments.submodular_allocator import (
    ProposalLane,
    expected_coverage,
    greedy_plan,
    marginal_coverage,
)


class SubmodularAllocatorTests(unittest.TestCase):
    def test_repeated_lane_has_diminishing_residual_gain(self):
        lane = ProposalLane("same", (0.8, 0.8), 1.0)
        self.assertGreater(marginal_coverage(lane, []), marginal_coverage(lane, [lane]))

    def test_complementary_lane_is_preferred_after_duplicate(self):
        first = ProposalLane("first", (0.9, 0.1), 1.0)
        duplicate = ProposalLane("duplicate", (0.9, 0.1), 1.0)
        complementary = ProposalLane("complementary", (0.1, 0.9), 1.0)
        self.assertGreater(
            marginal_coverage(complementary, [first]),
            marginal_coverage(duplicate, [first]),
        )

    def test_greedy_plan_respects_budget_and_improves_coverage(self):
        lanes = [
            ProposalLane("a", (0.8, 0.1), 1.0),
            ProposalLane("b", (0.1, 0.8), 1.0),
            ProposalLane("expensive", (1.0, 1.0), 3.0),
        ]
        selected = greedy_plan(lanes, 2.0)
        self.assertEqual([lane.name for lane in selected], ["a", "b"])
        self.assertGreater(expected_coverage(selected), expected_coverage([lanes[0]]))

    def test_invalid_rates_or_shape_are_rejected(self):
        with self.assertRaises(ValueError):
            greedy_plan([ProposalLane("bad", (1.1,), 1.0)], 1.0)
        with self.assertRaises(ValueError):
            expected_coverage([
                ProposalLane("a", (0.5,), 1.0),
                ProposalLane("b", (0.5, 0.5), 1.0),
            ])
