import math
import unittest

from kaggle_nemotron_probe.arc_decoder import (
    getter_log_evidence,
    score_kgmon,
    score_log_evidence,
)


def guess(solution, beam_score, augmentation_scores):
    return {
        "solution": solution,
        "beam_score": beam_score,
        "score_aug": augmentation_scores,
    }


class ArcDecoderEvidenceTests(unittest.TestCase):
    def test_log_evidence_marginalizes_in_probability_space(self):
        value = getter_log_evidence([
            guess([[1]], 1.0, [1.0, 3.0]),
            guess([[1]], 3.0, [1.0, 3.0]),
        ])
        expected_beam = math.log((math.exp(-1.0) + math.exp(-3.0)) / 2.0)
        expected_aug = math.log((math.exp(-1.0) + math.exp(-3.0)) / 2.0)
        self.assertAlmostEqual(value, expected_beam + expected_aug)

    def test_duplicate_class_copies_do_not_change_log_evidence(self):
        item = guess([[1]], 2.0, [2.0, 2.0])
        self.assertAlmostEqual(
            getter_log_evidence([item]),
            getter_log_evidence([item, item, item]),
        )

    def test_log_evidence_rejects_support_count_flooding(self):
        guesses = {
            "strong": guess([[1]], 0.5, [0.5, 0.5]),
        }
        guesses.update({
            f"flood-{index}": guess([[0]], 2.0, [2.0, 2.0])
            for index in range(20)
        })
        self.assertEqual(score_log_evidence(guesses)[0], [[1]])
        self.assertEqual(score_kgmon(guesses)[0], [[0]])

    def test_empty_evidence_is_negative_infinity(self):
        self.assertEqual(getter_log_evidence([]), float("-inf"))


if __name__ == "__main__":
    unittest.main()
