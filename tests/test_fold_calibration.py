import unittest

from experiments.candidate_records import CandidateRecord
from experiments.fold_calibration import (
    fit_family_calibration,
    target_family_position_rates,
    target_family_rates,
)


class FoldCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.challenges = {
            "a": {"train": [], "test": [{"input": [[0, 1]]}]},
            "b": {"train": [], "test": [{"input": [[0, 1], [0, 0]]}]},
        }
        self.solutions = {"a": [[[0, 1]]], "b": [[[1, 0], [0, 0]]]}

    def test_repeated_views_count_once_per_position(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="a", test_index=0, family="nvarc", candidate_id="v0",
                output=[[0, 1]],
            ),
            CandidateRecord.from_output(
                task_id="a", test_index=0, family="nvarc", candidate_id="v1",
                output=[[0, 1]],
            ),
        ]
        calibration = fit_family_calibration(records, self.challenges, self.solutions)["nvarc"]
        self.assertEqual(calibration.global_outcome.successes, 1)
        self.assertEqual(calibration.global_outcome.failures, 1)

    def test_missing_candidate_positions_count_as_failures(self) -> None:
        records = [CandidateRecord.from_output(
            task_id="a", test_index=0, family="sparse", candidate_id="p",
            output=[[0, 1]],
        )]
        calibration = fit_family_calibration(records, self.challenges, self.solutions)["sparse"]
        self.assertEqual(calibration.global_outcome.successes, 1)
        self.assertEqual(calibration.global_outcome.failures, 1)

    def test_eligibility_mask_excludes_unscheduled_positions(self) -> None:
        records = [CandidateRecord.from_output(
            task_id="a", test_index=0, family="adaptive", candidate_id="p",
            output=[[0, 1]],
        )]
        calibration = fit_family_calibration(
            records,
            self.challenges,
            self.solutions,
            eligible_positions={"adaptive": {("a", 0)}},
        )["adaptive"]
        self.assertEqual(calibration.global_outcome.successes, 1)
        self.assertEqual(calibration.global_outcome.failures, 0)

    def test_family_rate_is_available_for_unlabeled_target_inputs(self) -> None:
        records = [CandidateRecord.from_output(
            task_id="a", test_index=0, family="program", candidate_id="p",
            output=[[0, 1]],
        )]
        calibrations = fit_family_calibration(records, self.challenges, self.solutions)
        rates = target_family_rates(calibrations, self.challenges)
        self.assertIn("program", rates)
        self.assertGreater(rates["program"], 0.0)

    def test_position_rates_preserve_target_task_and_test_index(self) -> None:
        records = [CandidateRecord.from_output(
            task_id="a", test_index=0, family="program", candidate_id="p",
            output=[[0, 1]],
        )]
        calibrations = fit_family_calibration(records, self.challenges, self.solutions)
        rates = target_family_position_rates(calibrations, self.challenges)
        self.assertEqual(set(rates["program"]), {("a", 0), ("b", 0)})
        self.assertTrue(all(0.0 < value < 1.0 for value in rates["program"].values()))

    def test_fold_fit_learns_position_features_used_by_projection(self) -> None:
        records = [CandidateRecord.from_output(
            task_id="a", test_index=0, family="program", candidate_id="p",
            output=[[0, 1]],
        )]
        calibration = fit_family_calibration(
            records, self.challenges, self.solutions
        )["program"]
        self.assertIn(("position_area", "le_25"), calibration.feature_outcomes)

    def test_wrong_class_is_a_failure(self) -> None:
        records = [CandidateRecord.from_output(
            task_id="b", test_index=0, family="nvarc", candidate_id="wrong",
            output=[[0, 0], [0, 0]],
        )]
        calibration = fit_family_calibration(records, self.challenges, self.solutions)["nvarc"]
        self.assertEqual(calibration.global_outcome.successes, 0)
        self.assertEqual(calibration.global_outcome.failures, 2)


if __name__ == "__main__":
    unittest.main()
