import unittest

from experiments.verified_ensemble import (
    VerifiedPrediction,
    rank_verified_outputs,
    select_verified_pass2,
)


class VerifiedEnsembleTests(unittest.TestCase):
    def test_correlated_vote_flooding_does_not_hide_independent_class(self):
        predictions = [
            VerifiedPrediction(f"a{i}", [[1]], correlation_group="batch-a")
            for i in range(10)
        ] + [VerifiedPrediction("b", [[2]], correlation_group="batch-b")]
        ranked = rank_verified_outputs(predictions)
        self.assertEqual({item.prediction for item in ranked[:2]}, {((1,),), ((2,),)})
        self.assertAlmostEqual(sum(item.mass for item in ranked), 1.0)

    def test_mdl_selects_representative_inside_semantic_class(self):
        ranked = rank_verified_outputs([
            VerifiedPrediction("long", [[1]], mdl_length=8, correlation_group="g"),
            VerifiedPrediction("short", [[1]], mdl_length=1, correlation_group="g"),
        ])
        self.assertEqual(ranked[0].representative_program, "short")

    def test_two_attempt_boundary_and_validation(self):
        predictions = [
            VerifiedPrediction("a", [[1]]),
            VerifiedPrediction("b", [[2]]),
            VerifiedPrediction("c", [[3]]),
        ]
        self.assertEqual(len(select_verified_pass2(predictions)), 2)
        with self.assertRaises(ValueError):
            rank_verified_outputs(predictions, tau=0)


if __name__ == "__main__":
    unittest.main()
