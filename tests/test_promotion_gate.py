import unittest

from experiments.coverage_recovery import CoverageRecovery
from experiments.promotion_gate import RunManifest, decide_promotion


def metric(coverage, recovery, score, *, positions=10, tasks=5):
    return CoverageRecovery(
        total_positions=positions,
        covered_positions=round(coverage * positions),
        selected_positions=round(score * positions),
        total_tasks=tasks,
        fully_covered_tasks=0,
        solved_tasks=0,
        coverage_rate=coverage,
        selector_recovery_rate=recovery,
        output_score=score,
        task_coverage_rate=0.0,
        task_solve_rate=0.0,
    )


class PromotionGateTests(unittest.TestCase):
    def test_manifest_enforces_four_gpu_twelve_hour_boundary(self):
        manifest = RunManifest("r", "arc", "abc", ("m",))
        self.assertEqual(manifest.proposal_seconds, 42_600)
        with self.assertRaises(ValueError):
            RunManifest("r", "arc", "abc", ("m",), gpu_count=5)
        with self.assertRaises(ValueError):
            RunManifest("r", "arc", "abc", ("m",), wall_clock_seconds=43_201)

    def test_coverage_gain_promotes_without_score_regression(self):
        decision = decide_promotion(metric(0.5, 0.8, 0.4), metric(0.6, 0.8, 0.48))
        self.assertTrue(decision.promote)

    def test_selector_gain_promotes_without_coverage_regression(self):
        decision = decide_promotion(metric(0.8, 0.5, 0.4), metric(0.8, 0.6, 0.48))
        self.assertTrue(decision.promote)

    def test_score_only_gain_is_not_enough(self):
        decision = decide_promotion(metric(0.5, 0.8, 0.4), metric(0.5, 0.8, 0.5))
        self.assertFalse(decision.promote)
        self.assertIn("no coverage or selector-recovery gain", decision.reasons)

    def test_regression_and_missing_shadow_block_promotion(self):
        decision = decide_promotion(
            metric(0.8, 0.8, 0.64), metric(0.7, 0.9, 0.63),
            require_shadow=True,
        )
        self.assertFalse(decision.promote)
        self.assertIn("candidate loses exact candidate coverage", decision.reasons)
        self.assertIn("shadow fold milestone verification is missing", decision.reasons)
