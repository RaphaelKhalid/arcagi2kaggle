from __future__ import annotations

import unittest

from experiments.grid_renderer_grammar import (
    crop_content,
    dedupe_adjacent_cols,
    dedupe_adjacent_rows,
    fit_renderer_programs,
    remove_empty_cols,
    remove_empty_rows,
    trim_empty_border,
)


class GridRendererGrammarTests(unittest.TestCase):
    def test_dynamic_renderers(self) -> None:
        grid = ((0, 0, 2, 2), (0, 0, 0, 0), (3, 3, 0, 0))
        self.assertEqual(trim_empty_border(grid), grid)
        self.assertEqual(remove_empty_rows(grid), ((0, 0, 2, 2), (3, 3, 0, 0)))
        self.assertEqual(remove_empty_cols(((0, 2, 0), (0, 0, 0))), ((2,), (0,)))
        self.assertEqual(crop_content(((0, 0, 0), (0, 2, 0), (0, 0, 0))), ((2,),))

    def test_adjacent_deduplication(self) -> None:
        grid = ((1, 1, 2, 2), (1, 1, 2, 2), (3, 3, 4, 4))
        self.assertEqual(dedupe_adjacent_rows(grid), ((1, 1, 2, 2), (3, 3, 4, 4)))
        self.assertEqual(dedupe_adjacent_cols(grid), ((1, 2), (1, 2), (3, 4)))

    def test_palette_composition_is_demo_verified(self) -> None:
        programs = fit_renderer_programs([
            ([[0, 1, 0]], [[7, 8, 7]]),
            ([[1, 0, 1]], [[8, 7, 8]]),
        ])
        self.assertTrue(any(program.name == "palette_map" for program in programs))

    def test_empty_budget_is_closed(self) -> None:
        self.assertEqual(fit_renderer_programs([([[1]], [[1]])], max_programs=0), ())


if __name__ == "__main__":
    unittest.main()
