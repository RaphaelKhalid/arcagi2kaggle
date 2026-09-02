import math
import unittest

from experiments.augmentation_scoring import (
    augmentation_log_marginal,
    best_class_by_augmentation_marginal,
    records_to_marginal_candidates,
)
from experiments.candidate_records import CandidateRecord


class AugmentationScoringTests(unittest.TestCase):
    def test_uniform_group_marginal_is_log_mean_exp(self) -> None:
        expected = math.log((math.exp(-1.0) + math.exp(-3.0)) / 2.0)
        self.assertAlmostEqual(augmentation_log_marginal((1.0, 3.0)), expected)

    def test_marginal_rewards_one_strong_view_more_than_mean_nll(self) -> None:
        self.assertGreater(augmentation_log_marginal((0.0, 8.0)),
                           augmentation_log_marginal((4.0, 4.0)))

    def test_classes_rank_by_their_best_representative(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="a", candidate_id="a",
                output=[[0]], augmentation_nlls=(2.0, 2.0),
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="b", candidate_id="b",
                output=[[1]], augmentation_nlls=(0.0, 9.0),
            ),
        ]
        self.assertEqual(best_class_by_augmentation_marginal(records)[0],
                         records[1].output_hash)

    def test_selector_candidates_use_relative_marginal_weights(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="a", candidate_id="a",
                output=[[0]], augmentation_nlls=(0.0, 0.0),
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="a", candidate_id="b",
                output=[[1]], augmentation_nlls=(2.0, 2.0),
            ),
        ]
        candidates = records_to_marginal_candidates(records)
        self.assertEqual(len(candidates), 2)
        self.assertAlmostEqual(max(candidate.weight for candidate in candidates), 1.0)
        self.assertLess(min(candidate.weight for candidate in candidates), 1.0)


if __name__ == "__main__":
    unittest.main()
