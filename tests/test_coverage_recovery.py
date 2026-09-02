import unittest

from experiments.candidate_records import CandidateRecord
from experiments.coverage_recovery import evaluate_coverage_recovery


class CoverageRecoveryTests(unittest.TestCase):
    def test_empty_candidates_are_coverage_failures(self):
        result = evaluate_coverage_recovery([], solutions={"t": [[[1]], [[2]]]})
        self.assertEqual((result.total_positions, result.covered_positions), (2, 0))
        self.assertEqual(result.output_score, 0.0)

    def test_exact_candidate_is_covered_and_recovered(self):
        record = CandidateRecord.from_output(
            task_id="t", test_index=0, family="program", candidate_id="p", output=[[1]]
        )
        result = evaluate_coverage_recovery([record], solutions={"t": [[[1]]]})
        self.assertEqual((result.coverage_rate, result.selector_recovery_rate), (1.0, 1.0))

    def test_present_truth_can_still_be_selector_failure(self):
        records = [
            *(
                CandidateRecord.from_output(
                    task_id="t", test_index=0, family="decoder",
                    candidate_id=f"wrong-{i}", output=[[0]],
                    correlation_group="checkpoint-a",
                )
                for i in range(20)
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="decoder", candidate_id="right",
                output=[[1]], correlation_group="checkpoint-a",
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="independent", candidate_id="other",
                output=[[2]],
            ),
        ]
        result = evaluate_coverage_recovery(records, solutions={"t": [[[1]]]})
        self.assertEqual(result.coverage_rate, 1.0)
        self.assertEqual(result.selector_recovery_rate, 0.0)

    def test_collapsed_selector_recovers_present_truth(self):
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="decoder", candidate_id="wrong",
                output=[[0]], correlation_group="checkpoint-a",
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="decoder", candidate_id="right",
                output=[[1]], correlation_group="checkpoint-a",
            ),
        ]
        result = evaluate_coverage_recovery(
            records, solutions={"t": [[[1]]]}, collapse_correlated=True
        )
        self.assertEqual(result.selector_recovery_rate, 1.0)

    def test_output_and_task_weighting_are_reported_separately(self):
        records = [
            CandidateRecord.from_output(
                task_id="a", test_index=0, family="program", candidate_id="a0", output=[[1]]
            ),
            CandidateRecord.from_output(
                task_id="a", test_index=1, family="program", candidate_id="a1", output=[[2]]
            ),
            CandidateRecord.from_output(
                task_id="b", test_index=0, family="program", candidate_id="b0", output=[[0]]
            ),
        ]
        result = evaluate_coverage_recovery(
            records, solutions={"a": [[[1]], [[2]]], "b": [[[9]]]}
        )
        self.assertEqual(result.output_score, 2 / 3)
        self.assertEqual(result.task_solve_rate, 0.5)
        self.assertEqual(result.task_coverage_rate, 0.5)
