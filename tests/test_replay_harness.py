from __future__ import annotations

import unittest

from experiments.candidate_records import CandidateRecord
from experiments.replay_harness import build_submission, replay_score


class ReplayHarnessTests(unittest.TestCase):
    def test_replay_uses_selector_and_official_score(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="dense", candidate_id="wrong",
                output=[[0]], weight=10,
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="program", candidate_id="right",
                output=[[1]],
            ),
        ]
        self.assertEqual(replay_score(records, solutions={"t": [[[1]]]}), 1.0)

    def test_uncovered_outputs_are_not_dropped(self) -> None:
        submission = build_submission([], n_test_by_task={"a": 1, "b": 2})
        self.assertEqual(set(submission), {"a", "b"})
        self.assertEqual(len(submission["b"]), 2)
        self.assertEqual(submission["a"][0]["attempt_1"], [[0]])

    def test_output_weighting_is_preserved(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="a", test_index=0, family="program", candidate_id="a",
                output=[[1]],
            ),
            CandidateRecord.from_output(
                task_id="b", test_index=0, family="program", candidate_id="b",
                output=[[2]],
            ),
        ]
        self.assertEqual(
            replay_score(records, solutions={"a": [[[1]], [[9]]], "b": [[[2]]]}),
            2 / 3,
        )

    def test_collapsed_replay_uses_lineage_quotient_at_submission_boundary(self) -> None:
        records = [
            *(
                CandidateRecord.from_output(
                    task_id="t", test_index=0, family="decoder",
                    candidate_id=f"wrong-{index}", output=[[0]],
                    correlation_group="checkpoint-a",
                )
                for index in range(20)
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
        solutions = {"t": [[[1]]]}
        self.assertEqual(
            replay_score(records, solutions=solutions), 0.0
        )
        self.assertEqual(
            replay_score(records, solutions=solutions, collapse_correlated=True), 1.0
        )


if __name__ == "__main__":
    unittest.main()
