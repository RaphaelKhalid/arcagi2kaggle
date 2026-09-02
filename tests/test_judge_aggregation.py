import unittest

from experiments.judge_aggregation import (
    aggregate_ranked_pairs,
    position_debiased_aggregation,
)


class JudgeAggregationTests(unittest.TestCase):
    def test_weighted_council_can_recover_minority_first_choice(self) -> None:
        pairs = [("minority", "modal"), ("minority", "modal"), ("modal", "other")]
        self.assertEqual(set(aggregate_ranked_pairs(pairs)), {"minority", "modal"})

    def test_second_slot_is_distinct(self) -> None:
        self.assertEqual(aggregate_ranked_pairs(
            [("a", "b"), ("a", "b"), ("b", "a")]
        ), ("a", "b"))

    def test_debiased_control_ignores_rank(self) -> None:
        pairs = [("a", "b"), ("b", "a"), ("b", "c")]
        self.assertEqual(position_debiased_aggregation(pairs), ("b", "a"))

    def test_malformed_pairs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_ranked_pairs([("a",)])
        with self.assertRaises(ValueError):
            aggregate_ranked_pairs([("a", "a")])

    def test_correlated_judge_group_cannot_flood_the_council(self) -> None:
        pairs = [("modal", "modal-alt")] * 10 + [("minority", "other")]
        groups = ["same-prompt"] * 10 + ["independent"]
        selected = aggregate_ranked_pairs(pairs, judge_groups=groups)
        self.assertIn("minority", selected)

    def test_judge_group_shape_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_ranked_pairs([("a", "b")], judge_groups=[])


if __name__ == "__main__":
    unittest.main()
