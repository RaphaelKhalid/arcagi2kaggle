import unittest

from experiments.occam_gate import (
    OccamCandidate,
    complexity_penalty,
    rank_by_occam,
    score_occam,
)


class OccamGateTests(unittest.TestCase):
    def test_penalty_is_bounded_and_decreases_with_more_demos(self):
        self.assertLess(
            complexity_penalty(8, 100), complexity_penalty(8, 10)
        )
        self.assertLessEqual(complexity_penalty(10_000, 1), 1.0)

    def test_short_exact_program_beats_long_exact_program(self):
        ranked = rank_by_occam(
            [OccamCandidate("short", 0, 4), OccamCandidate("long", 0, 40)],
            20,
        )
        self.assertEqual(ranked[0].program_id, "short")

    def test_empirical_failure_can_override_small_complexity_difference(self):
        ranked = rank_by_occam(
            [OccamCandidate("fit", 0, 0), OccamCandidate("fail", 1, 0)],
            100,
        )
        self.assertEqual(ranked[0].program_id, "fit")

    def test_invalid_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            score_occam(OccamCandidate("bad", 3, 1), 2)
