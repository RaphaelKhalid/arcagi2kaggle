"""Tests for arc.score: the official metric and its breakdowns."""

import unittest

from arc.score import score_submission

CHALLENGES = {
    "t1": {"train": [], "test": [{"input": [[0]]}]},
    "t2": {"train": [], "test": [{"input": [[0]]}, {"input": [[0]]}]},
}
SOLUTIONS = {
    "t1": [[[1, 2], [3, 4]]],
    "t2": [[[5]], [[6, 7]]],
}


def entry(a1, a2):
    return {"attempt_1": a1, "attempt_2": a2}


class ScoreTests(unittest.TestCase):
    def test_perfect_submission_scores_one(self) -> None:
        sub = {
            "t1": [entry([[1, 2], [3, 4]], [[0]])],
            "t2": [entry([[5]], [[5]]), entry([[6, 7]], [[6, 7]])],
        }
        report = score_submission(sub, CHALLENGES, SOLUTIONS)
        self.assertEqual(report.score, 1.0)
        self.assertEqual(report.correct_outputs, 3)
        self.assertEqual(report.total_outputs, 3)
        self.assertTrue(all(t.solved for t in report.per_task.values()))

    def test_all_wrong_scores_zero(self) -> None:
        sub = {
            "t1": [entry([[9]], [[8]])],
            "t2": [entry([[9]], [[9]]), entry([[9]], [[9]])],
        }
        report = score_submission(sub, CHALLENGES, SOLUTIONS)
        self.assertEqual(report.score, 0.0)

    def test_dimension_mismatch_is_wrong(self) -> None:
        # Right cells, wrong shape: [[1,2],[3,4]] vs [[1,2,3,4]] etc.
        sub = {
            "t1": [entry([[1, 2, 3, 4]], [[1], [2], [3], [4]])],
            "t2": [entry([[5]], [[5]]), entry([[6], [7]], [[6, 7, 0]])],
        }
        report = score_submission(sub, CHALLENGES, SOLUTIONS)
        self.assertEqual(report.correct_outputs, 1)  # only t2[0]
        self.assertAlmostEqual(report.score, 1 / 3)

    def test_attempt_2_rescue(self) -> None:
        # attempt_1 wrong, attempt_2 exact: still 1 point.
        sub = {
            "t1": [entry([[0]], [[1, 2], [3, 4]])],
            "t2": [entry([[0]], [[0]]), entry([[0]], [[0]])],
        }
        report = score_submission(sub, CHALLENGES, SOLUTIONS)
        self.assertEqual(report.correct_outputs, 1)
        self.assertEqual(
            report.attempt_breakdown,
            {"attempt_1_only": 0, "attempt_2_only": 1, "both": 0},
        )

    def test_attempt_breakdown(self) -> None:
        sub = {
            "t1": [entry([[1, 2], [3, 4]], [[1, 2], [3, 4]])],  # both
            "t2": [entry([[5]], [[0]]), entry([[0]], [[6, 7]])],  # a1 only, a2 only
        }
        report = score_submission(sub, CHALLENGES, SOLUTIONS)
        self.assertEqual(
            report.attempt_breakdown,
            {"attempt_1_only": 1, "attempt_2_only": 1, "both": 1},
        )
        self.assertEqual(report.score, 1.0)

    def test_per_task_breakdown(self) -> None:
        sub = {
            "t1": [entry([[9]], [[9]])],
            "t2": [entry([[5]], [[9]]), entry([[9]], [[9]])],
        }
        report = score_submission(sub, CHALLENGES, SOLUTIONS)
        self.assertEqual(report.per_task["t1"].correct, 0)
        self.assertEqual(report.per_task["t2"].correct, 1)
        self.assertFalse(report.per_task["t2"].solved)

    def test_missing_pieces_score_zero_not_raise(self) -> None:
        # Defensive scoring: absent task / short entry list / missing attempts.
        sub = {"t2": [entry([[5]], [[5]])]}  # t1 absent, t2 short by one
        report = score_submission(sub, CHALLENGES, SOLUTIONS)
        self.assertEqual(report.correct_outputs, 1)
        self.assertEqual(report.total_outputs, 3)

    def test_bad_solutions_raise(self) -> None:
        with self.assertRaises(ValueError):
            score_submission({}, CHALLENGES, {"t1": SOLUTIONS["t1"]})
        with self.assertRaises(ValueError):
            score_submission(
                {}, CHALLENGES, {"t1": SOLUTIONS["t1"], "t2": SOLUTIONS["t2"][:1]}
            )


if __name__ == "__main__":
    unittest.main()
