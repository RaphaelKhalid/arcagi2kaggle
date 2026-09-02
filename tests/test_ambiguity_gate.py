import unittest

from experiments.ambiguity_gate import ambiguity_regressions, summarize_certificates
from experiments.cegis_version_space import Demonstration, Program
from experiments.promotion_gate import decide_promotion
from experiments.version_space_certificate import certify_version_space
from experiments.coverage_recovery import CoverageRecovery


class AmbiguityGateTests(unittest.TestCase):
    @staticmethod
    def execute(program: Program, value: int) -> int:
        return value + {"plus": 1, "copy": 0, "minus": -1}[program.program_id]

    @staticmethod
    def metric() -> CoverageRecovery:
        return CoverageRecovery(1, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0)

    def certificate(self, ids, demos=()):
        return certify_version_space(
            [Program(program_id, "dsl") for program_id in ids],
            demos, [10], self.execute,
        )

    def test_summary_distinguishes_forced_coverable_and_unresolved(self) -> None:
        summary = summarize_certificates([
            self.certificate(["plus"], [Demonstration(0, 1)]),
            self.certificate(["plus", "copy"], []),
            self.certificate([], [],),
        ])
        self.assertEqual(summary.total_positions, 3)
        self.assertEqual(summary.forced_positions, 1)
        self.assertEqual(summary.pass2_coverable_positions, 2)
        self.assertEqual(summary.unresolved_positions, 1)

    def test_ambiguity_regression_is_explicit(self) -> None:
        baseline = summarize_certificates([self.certificate(["plus"])])
        candidate = summarize_certificates([self.certificate(["plus", "copy", "minus"])])
        reasons = ambiguity_regressions(baseline, candidate)
        self.assertIn("candidate loses forced-output positions", reasons)
        self.assertIn("candidate loses pass@2-coverable positions", reasons)

    def test_promotion_can_require_ambiguity_non_regression(self) -> None:
        baseline = summarize_certificates([self.certificate(["plus"])])
        candidate = summarize_certificates([self.certificate(["plus", "copy", "minus"])])
        decision = decide_promotion(
            self.metric(), self.metric(),
            baseline_ambiguity=baseline, candidate_ambiguity=candidate,
        )
        self.assertFalse(decision.promote)
        self.assertIn("candidate loses forced-output positions", decision.reasons)


if __name__ == "__main__":
    unittest.main()
