from __future__ import annotations

import unittest

from experiments.motif_panel_grammar import (
    fit_motif_panel_programs,
    motif_transforms,
)


class MotifPanelGrammarTests(unittest.TestCase):
    def test_dimension_preserving_transforms_are_available(self) -> None:
        transforms = motif_transforms(((1, 2), (3, 4)))
        self.assertEqual(transforms["flip_h"], ((2, 1), (4, 3)))
        self.assertEqual(transforms["rot180"], ((4, 3), (2, 1)))
        self.assertEqual(transforms["anti_transpose"], ((4, 2), (3, 1)))

    def test_row_template_recovers_alternating_panel(self) -> None:
        programs = fit_motif_panel_programs([
            (
                [[1, 2], [3, 4]],
                [[1, 2, 1, 2], [3, 4, 3, 4],
                 [2, 1, 2, 1], [4, 3, 4, 3],
                 [1, 2, 1, 2], [3, 4, 3, 4]],
            ),
            (
                [[5, 6], [7, 8]],
                [[5, 6, 5, 6], [7, 8, 7, 8],
                 [6, 5, 6, 5], [8, 7, 8, 7],
                 [5, 6, 5, 6], [7, 8, 7, 8]],
            ),
        ])
        self.assertTrue(any("row_template" in program.name for program in programs))
        self.assertEqual(programs[0].apply([[9, 0], [1, 2]])[:2],
                         ((9, 0, 9, 0), (1, 2, 1, 2)))

    def test_non_panel_output_is_rejected(self) -> None:
        self.assertEqual(fit_motif_panel_programs([([[1, 2]], [[1, 2, 3]])]), ())

    def test_bounded_matrix_template_recovers_mixed_panels(self) -> None:
        programs = fit_motif_panel_programs([
            (
                [[1, 2], [3, 4]],
                [[4, 3, 3, 4], [2, 1, 1, 2],
                 [2, 1, 1, 2], [4, 3, 3, 4]],
            ),
        ])
        self.assertTrue(any("matrix_template" in program.name for program in programs))

    def test_budget_is_explicit(self) -> None:
        self.assertEqual(fit_motif_panel_programs([([[1]], [[1]])], max_programs=0), ())


if __name__ == "__main__":
    unittest.main()
