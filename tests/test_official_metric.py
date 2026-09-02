from __future__ import annotations

import unittest

from experiments.official_metric import official_pass2_score


class OfficialMetricTests(unittest.TestCase):
    def test_score_is_output_weighted_and_pass2_exact(self) -> None:
        score = official_pass2_score(
            {
                "a": [
                    {"attempt_1": [[1]], "attempt_2": [[0]]},
                    {"attempt_1": [[0]], "attempt_2": [[0]]},
                ],
                "b": [{"attempt_1": [[2]], "attempt_2": [[0]]}],
            },
            {"a": [[[1]], [[2]]], "b": [[[2]]]},
        )
        self.assertEqual(score, 2 / 3)

    def test_missing_task_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            official_pass2_score({}, {"a": [[[0]]]})

    def test_missing_attempt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            official_pass2_score(
                {"a": [{"attempt_1": [[0]]}]}, {"a": [[[0]]]}
            )


if __name__ == "__main__":
    unittest.main()
