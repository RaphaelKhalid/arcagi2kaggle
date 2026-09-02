import unittest

from experiments.pass2_selector import Candidate, TaskCandidate
from experiments.robust_selector import (
    robust_select_pass2,
    robust_select_task_output_pairs,
    score_robust_classes,
)


class RobustSelectorTests(unittest.TestCase):
    def test_fixed_mixture_has_exact_class_mass(self):
        scores = score_robust_classes(
            [Candidate("a", "fast"), Candidate("b", "slow")],
            family_prior_intervals={"fast": (0.7, 0.7), "slow": (0.3, 0.3)},
            alpha=0.0,
        )
        self.assertEqual([(item.output_hash, item.lower_mass, item.upper_mass) for item in scores],
                         [("a", 0.7, 0.7), ("b", 0.3, 0.3)])

    def test_worst_case_pair_uses_interval_simplex_not_point_prior(self):
        candidates = [
            Candidate("a", "dominant"),
            Candidate("b", "middle"),
            Candidate("c", "tail"),
        ]
        selection = robust_select_pass2(
            candidates,
            family_prior_intervals={
                "dominant": (0.6, 0.6),
                "middle": (0.2, 0.4),
                "tail": (0.0, 0.2),
            },
            alpha=0.0,
        )
        self.assertEqual(selection.outputs, ("a", "b"))
        self.assertAlmostEqual(selection.worst_case_mass, 0.8)

    def test_invalid_or_infeasible_intervals_are_rejected(self):
        with self.assertRaises(ValueError):
            score_robust_classes(
                [Candidate("a", "fast")],
                family_prior_intervals={"fast": (0.8, 0.7)},
            )
        with self.assertRaises(ValueError):
            score_robust_classes(
                [Candidate("a", "fast"), Candidate("b", "slow")],
                family_prior_intervals={"fast": (0.8, 0.9), "slow": (0.0, 0.0)},
            )

    def test_correlation_collapse_applies_before_robust_mixture(self):
        candidates = [
            *(Candidate("wrong", "decoder", correlation_group="one") for _ in range(10)),
            Candidate("right", "decoder", correlation_group="one"),
            Candidate("other", "independent", correlation_group="two"),
        ]
        selection = robust_select_pass2(
            candidates,
            family_prior_intervals={"decoder": (0.7, 0.7), "independent": (0.3, 0.3)},
            collapse_correlated=True,
        )
        self.assertEqual(set(selection.outputs), {"right", "wrong"})

    def test_task_pairs_optimize_one_shared_prior_adversary(self):
        selection = robust_select_task_output_pairs(
            [
                TaskCandidate("a", ("a", "x"), "dominant"),
                TaskCandidate("b", ("b", "y"), "middle"),
                TaskCandidate("c", ("c", "z"), "tail"),
            ],
            family_prior_intervals={
                "dominant": (0.6, 0.6),
                "middle": (0.2, 0.4),
                "tail": (0.0, 0.2),
            },
            alpha=0.0,
        )
        self.assertEqual(selection.pairs, (("a", "b"), ("x", "y")))
        self.assertAlmostEqual(selection.worst_case_mass, 1.6)

    def test_task_action_cap_is_explicit(self):
        candidates = [
            TaskCandidate("a", ("a",), "f1"),
            TaskCandidate("b", ("b",), "f2"),
            TaskCandidate("c", ("c",), "f3"),
        ]
        with self.assertRaises(ValueError):
            robust_select_task_output_pairs(
                candidates,
                family_prior_intervals={
                    "f1": (1 / 3, 1 / 3), "f2": (1 / 3, 1 / 3),
                    "f3": (1 / 3, 1 / 3),
                },
                alpha=0.0,
                max_joint_actions=2,
            )
