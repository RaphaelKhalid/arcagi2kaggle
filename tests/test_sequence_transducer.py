from __future__ import annotations

import unittest

from experiments.sequence_transducer import (
    fit_sequence_programs,
)


class SequenceTransducerTests(unittest.TestCase):
    def test_row_lexicographic_sort(self) -> None:
        programs = fit_sequence_programs([
            ([[0, 2], [0, 1]], [[0, 1], [0, 2]]),
            ([[3, 4], [3, 2]], [[3, 2], [3, 4]]),
        ])
        self.assertTrue(any(program.name == "rows_lex_asc" for program in programs))

    def test_column_density_sort(self) -> None:
        programs = fit_sequence_programs([
            ([[1, 0, 0], [1, 2, 0]], [[0, 0, 1], [0, 2, 1]]),
        ])
        self.assertTrue(any(program.name == "cols_density_asc" for program in programs))

    def test_shape_change_is_rejected(self) -> None:
        self.assertEqual(fit_sequence_programs([([[1, 0]], [[1]])]), ())

    def test_empty_input_is_closed(self) -> None:
        self.assertEqual(fit_sequence_programs([]), ())


if __name__ == "__main__":
    unittest.main()
