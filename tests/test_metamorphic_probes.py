import unittest

from experiments.cegis_version_space import Program
from experiments.metamorphic_probes import (
    color_swap_probe,
    d8_probes,
    evaluate_probe,
    translation_probes,
)


class MetamorphicProbeTests(unittest.TestCase):
    @staticmethod
    def execute(program: Program, grid):
        if program.program_id == "coordinate":
            return tuple(tuple(2 if (r, c) == (0, 0) else cell
                               for c, cell in enumerate(row))
                         for r, row in enumerate(grid))
        return grid

    def test_identity_is_d8_equivariant(self) -> None:
        evidence = [evaluate_probe(
            Program("identity", "dsl"), [[[0, 1], [0, 0]]], self.execute, probe
        ) for probe in d8_probes()]
        self.assertTrue(evidence)
        self.assertTrue(all(item.checked == 1 and item.passed == 1 for item in evidence))

    def test_coordinate_memorization_fails_translation(self) -> None:
        evidence = evaluate_probe(
            Program("coordinate", "neural"), [[[0, 1], [0, 0]]],
            self.execute, translation_probes()[0],
        )
        self.assertEqual(evidence.checked, 1)
        self.assertEqual(evidence.passed, 0)

    def test_color_probe_is_evidence_not_a_hard_gate(self) -> None:
        evidence = evaluate_probe(
            Program("identity", "dsl"), [[[0, 1]]], self.execute,
            color_swap_probe(1, 2),
        )
        self.assertEqual(evidence.passed, 1)
        self.assertIn("soft", evidence.justification)


if __name__ == "__main__":
    unittest.main()
