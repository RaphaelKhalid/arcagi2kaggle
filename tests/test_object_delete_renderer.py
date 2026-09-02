from __future__ import annotations

import unittest

from experiments.object_delete_renderer import fit_delete_programs


class ObjectDeleteRendererTests(unittest.TestCase):
    def test_exact_single_object_deletion(self) -> None:
        programs = fit_delete_programs([
            ([[0, 1, 0], [0, 0, 2]], [[0, 0, 0], [0, 0, 2]]),
            ([[0, 4, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]),
        ])
        self.assertTrue(any(program.name in {"delete_area_border", "delete_shape_border"}
                            for program in programs))

    def test_ambiguous_guard_abstains_at_execution(self) -> None:
        programs = fit_delete_programs([
            ([[0, 1, 0], [0, 0, 2]], [[0, 0, 0], [0, 0, 2]]),
        ])
        self.assertTrue(programs)
        self.assertTrue(all(
            program.apply([[0, 0, 0, 0], [0, 1, 3, 0], [0, 0, 0, 0]]) is None
            for program in programs
        ))

    def test_non_deletion_is_rejected(self) -> None:
        self.assertEqual(fit_delete_programs([
            ([[0, 1]], [[0, 2]])
        ]), ())


if __name__ == "__main__":
    unittest.main()
