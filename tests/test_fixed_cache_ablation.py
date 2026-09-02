from __future__ import annotations

import unittest

from experiments.candidate_records import CandidateRecord
from experiments.fixed_cache_ablation import fixed_cache_report


class FixedCacheAblationTests(unittest.TestCase):
    def test_counts_unique_classes_and_family_overlap(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="a", candidate_id="a1",
                output=[[1]],
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="a", candidate_id="a2",
                output=[[1]],
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="b", candidate_id="b1",
                output=[[2]],
            ),
        ]
        report = fixed_cache_report(records)
        self.assertEqual(report.raw_records, 3)
        self.assertEqual(report.positions_with_records, 1)
        self.assertEqual(report.output_classes, 2)
        self.assertEqual(report.family_records, {"a": 2, "b": 1})
        self.assertEqual(report.family_pairwise_class_jaccard["a|b"], 0.0)

    def test_labeled_recall_is_separate_from_inventory(self) -> None:
        records = [CandidateRecord.from_output(
            task_id="t", test_index=0, family="a", candidate_id="a",
            output=[[2]],
        )]
        report = fixed_cache_report(records, solutions={"t": [[[2]]]})
        self.assertEqual(report.candidate_recall, (1, 1))
        self.assertEqual(report.selected_score, 1.0)

    def test_unlabeled_report_has_no_accuracy_claim(self) -> None:
        report = fixed_cache_report([])
        self.assertIsNone(report.candidate_recall)
        self.assertIsNone(report.selected_score)


if __name__ == "__main__":
    unittest.main()
