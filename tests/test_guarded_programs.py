import unittest

from experiments.guarded_programs import (
    GuardedBranch,
    compose_guarded_program,
)


class GuardedProgramTests(unittest.TestCase):
    @staticmethod
    def identity(grid):
        return grid

    @staticmethod
    def flip(grid):
        return tuple(tuple(reversed(row)) for row in grid)

    @staticmethod
    def has_marker(grid):
        return any(cell == 2 for row in grid for cell in row)

    def branch(self):
        return GuardedBranch("has_marker", self.has_marker, "flip", self.flip)

    def test_piecewise_rule_has_exact_truth_table(self):
        result = compose_guarded_program(
            [self.branch()], "identity", self.identity,
            [([[0, 2, 0]], [[0, 2, 0]]), ([[1, 0, 0]], [[1, 0, 0]])],
        )
        self.assertIsNotNone(result)
        program, proof = result
        self.assertTrue(proof.branch_is_exclusive)
        self.assertEqual(proof.truth_table[0], ("has_marker->flip",))
        self.assertEqual(proof.truth_table[1], ("fallback->identity",))
        self.assertEqual(program.fallback_name, "identity")

    def test_overlapping_guards_are_not_proof_carrying(self):
        overlap = GuardedBranch("always", lambda grid: True, "identity", self.identity)
        result = compose_guarded_program(
            [self.branch(), overlap], "identity", self.identity,
            [([[0, 2, 0]], [[0, 2, 0]])],
        )
        self.assertIsNone(result)

    def test_wrong_fallback_fails_exact_demo_replay(self):
        result = compose_guarded_program(
            [self.branch()], "flip", self.flip,
            [([[1, 0, 0]], [[1, 0, 0]])],
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
