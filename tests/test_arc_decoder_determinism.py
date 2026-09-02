import unittest

from kaggle_nemotron_probe.arc_decoder import score_kgmon, score_log_evidence


def item(grid):
    return {"solution": grid, "beam_score": 1.0, "score_aug": [1.0, 1.0]}


class ArcDecoderDeterminismTests(unittest.TestCase):
    def test_kgmon_ties_use_canonical_grid_order(self):
        forward = {"b": item([[1]]), "a": item([[0]])}
        reverse = {"a": item([[0]]), "b": item([[1]])}
        self.assertEqual(score_kgmon(forward), [[[0]], [[1]]])
        self.assertEqual(score_kgmon(reverse), [[[0]], [[1]]])

    def test_log_evidence_ties_use_canonical_grid_order(self):
        forward = {"b": item([[1]]), "a": item([[0]])}
        reverse = {"a": item([[0]]), "b": item([[1]])}
        self.assertEqual(score_log_evidence(forward), [[[0]], [[1]]])
        self.assertEqual(score_log_evidence(reverse), [[[0]], [[1]]])


if __name__ == "__main__":
    unittest.main()
