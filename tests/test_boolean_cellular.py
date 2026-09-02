from __future__ import annotations

import unittest

from experiments.boolean_cellular import fit_boolean_programs


class BooleanCellularTests(unittest.TestCase):
    def test_threshold_rule_replays(self) -> None:
        programs = fit_boolean_programs([
            (
                [[0, 1, 0], [1, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ),
        ])
        self.assertTrue(any(program.name == "count_ge_3" for program in programs))

    def test_shape_change_is_rejected(self) -> None:
        self.assertEqual(fit_boolean_programs([([[1]], [[1, 1]])]), ())

    def test_empty_input_is_closed(self) -> None:
        self.assertEqual(fit_boolean_programs([]), ())


if __name__ == "__main__":
    unittest.main()
