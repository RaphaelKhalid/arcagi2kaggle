from __future__ import annotations

import unittest

from experiments.pass2_selector import (
    Candidate,
    TaskCandidate,
    score_output_classes,
    select_pass2,
    select_task_output_pairs,
    select_task_program_pair,
)


class Pass2SelectorTests(unittest.TestCase):
    def test_hard_invalid_candidates_are_ignored(self) -> None:
        selected = select_pass2(
            [
                Candidate("invalid", "dense", weight=100.0, hard_valid=False),
                Candidate("valid", "program"),
            ]
        )
        self.assertEqual(selected, ("valid",))

    def test_correlated_votes_do_not_overrule_family_support(self) -> None:
        ranked = score_output_classes(
            [
                *(Candidate("wrong", "dense") for _ in range(100)),
                Candidate("correct", "program"),
                Candidate("correct", "recursive"),
            ]
        )
        self.assertEqual(ranked[0].output_hash, "correct")
        self.assertEqual(
            select_pass2(
                [
                    Candidate("wrong", "dense"),
                    Candidate("correct", "program"),
                    Candidate("correct", "recursive"),
                ]
            ),
            ("correct", "wrong"),
        )

    def test_semantic_switch_removes_within_family_copy_count(self) -> None:
        candidates = [
            *(Candidate("wrong", "decoder") for _ in range(20)),
            Candidate("right", "decoder"),
            Candidate("other", "independent"),
        ]
        baseline = score_output_classes(
            candidates,
            family_priors={"decoder": 0.7, "independent": 0.3},
        )
        semantic = score_output_classes(
            candidates,
            family_priors={"decoder": 0.7, "independent": 0.3},
            collapse_correlated=True,
        )
        self.assertEqual(baseline[0].output_hash, "wrong")
        self.assertEqual(
            {item.output_hash for item in semantic[:2]}, {"right", "wrong"}
        )

    def test_correlation_groups_preserve_independent_lineages(self) -> None:
        candidates = [
            *(Candidate("wrong", "decoder", correlation_group="checkpoint-a")
              for _ in range(20)),
            Candidate("right", "decoder", correlation_group="checkpoint-b"),
        ]
        ranked = score_output_classes(
            candidates, collapse_correlated=True, alpha=0.0
        )
        self.assertEqual({item.output_hash for item in ranked}, {"right", "wrong"})
        self.assertAlmostEqual(ranked[0].score, ranked[1].score)

    def test_semantic_switch_reaches_independent_output_pairing(self) -> None:
        candidates = [
            *(TaskCandidate(
                "wrong-copy", ("wrong", "wrong"), "decoder",
                correlation_group="checkpoint-a",
            ) for _ in range(20)),
            TaskCandidate(
                "right", ("right", "right"), "decoder",
                correlation_group="checkpoint-a",
            ),
            TaskCandidate("other", ("other", "other"), "independent"),
        ]
        pairs = select_task_output_pairs(
            candidates,
            family_priors={"decoder": 0.7, "independent": 0.3},
            collapse_correlated=True,
        )
        self.assertEqual(pairs, (("right", "wrong"), ("right", "wrong")))

    def test_multi_test_pair_is_selected_as_shared_vectors(self) -> None:
        pair = select_task_program_pair(
            [
                TaskCandidate("dense", ("wrong", "wrong"), "dense", weight=8.0),
                TaskCandidate("program", ("correct", "correct"), "program"),
                TaskCandidate("recursive", ("alternate", "correct"), "recursive"),
            ],
            family_priors={"dense": 0.5, "program": 0.3, "recursive": 0.2},
        )
        self.assertEqual({candidate.program_hash for candidate in pair},
                         {"dense", "program"})

    def test_official_metric_uses_independent_marginal_pairs(self) -> None:
        pairs = select_task_output_pairs(
            [
                TaskCandidate("v1", ("a", "x"), "program", weight=4.0),
                TaskCandidate("v2", ("a", "y"), "program", weight=3.0),
                TaskCandidate("v3", ("b", "z"), "program", weight=2.0),
                TaskCandidate("v4", ("c", "x"), "program", weight=1.0),
            ]
        )
        self.assertEqual(pairs, (("a", "b"), ("x", "y")))


if __name__ == "__main__":
    unittest.main()
