import unittest

from experiments.cegis_version_space import Demonstration, Program
from experiments.version_space_certificate import (
    certify_version_space,
    posterior_is_complete,
)


class VersionSpaceCertificateTests(unittest.TestCase):
    @staticmethod
    def execute(program: Program, value: int) -> int:
        if program.program_id == "plus":
            return value + 1
        if program.program_id == "minus":
            return value - 1
        return value

    def test_demo_consistency_does_not_imply_forced_hidden_output(self) -> None:
        certificate = certify_version_space(
            [
                Program("plus", "dsl", mdl_length=1),
                Program("minus", "dsl", mdl_length=1),
                Program("copy", "dsl", mdl_length=1),
            ],
            [],
            [10],
            self.execute,
        )
        test = certificate.tests[0]
        self.assertTrue(certificate.demo_verified)
        self.assertFalse(certificate.task_forced)
        self.assertEqual(test.ambiguity_count, 3)
        self.assertFalse(test.pass2_covers_entire_version_space)
        self.assertAlmostEqual(test.top2_posterior_mass, 2.0 / 3.0)

    def test_exact_demo_filter_can_prove_a_forced_output(self) -> None:
        certificate = certify_version_space(
            [
                Program("plus", "dsl"),
                Program("minus", "dsl"),
            ],
            [Demonstration(0, 1)],
            [10],
            self.execute,
        )
        test = certificate.tests[0]
        self.assertEqual(certificate.survivors, ("plus",))
        self.assertTrue(certificate.task_forced)
        self.assertEqual(test.forced_output, 11)
        self.assertTrue(posterior_is_complete(test))

    def test_two_surviving_outputs_are_complete_for_pass_two(self) -> None:
        certificate = certify_version_space(
            [Program("plus", "dsl"), Program("copy", "dsl")],
            [],
            [10],
            self.execute,
        )
        test = certificate.tests[0]
        self.assertFalse(certificate.task_forced)
        self.assertTrue(test.pass2_covers_entire_version_space)
        self.assertTrue(posterior_is_complete(test))


if __name__ == "__main__":
    unittest.main()
