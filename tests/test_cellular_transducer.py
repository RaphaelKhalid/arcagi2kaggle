from __future__ import annotations

import unittest

from experiments.cellular_transducer import (
    BOUNDARY,
    context_at,
    fit_cellular_program,
    fit_cellular_programs,
)


class CellularTransducerTests(unittest.TestCase):
    def test_boundary_is_explicit(self) -> None:
        context = context_at(((1, 2), (3, 4)), 0, 0,
                             offsets=((0, 0), (0, -1), (-1, 0)))
        self.assertEqual(context, (1, BOUNDARY, BOUNDARY))

    def test_local_rule_replays_all_demos(self) -> None:
        program = fit_cellular_program([
            ([[0, 1, 0]], [[0, 2, 0]]),
            ([[1, 0, 1]], [[2, 0, 2]]),
        ])
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(program.apply([[1, 0, 1]]), ((2, 0, 2),))

    def test_conflicting_context_is_rejected(self) -> None:
        self.assertIsNone(fit_cellular_program([
            ([[0, 1, 0]], [[0, 2, 0]]),
            ([[0, 1, 0]], [[0, 3, 0]]),
        ]))

    def test_unseen_context_abstains(self) -> None:
        program = fit_cellular_program([([[0]], [[1]])])
        self.assertIsNotNone(program)
        assert program is not None
        self.assertIsNone(program.apply([[2]]))

    def test_identity_fallback_totalizes_unseen_context(self) -> None:
        program = fit_cellular_program([([[0]], [[1]])])
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(program.apply_with_fallback([[2]], "identity"), ((2,),))

    def test_background_fallback_totalizes_unseen_context(self) -> None:
        program = fit_cellular_program([([[0]], [[1]])])
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(program.apply_with_fallback([[2]], "background"), ((2,),))

    def test_role_mode_lifts_a_color_permutation(self) -> None:
        program = fit_cellular_program([
            ([[0, 1, 0]], [[0, 1, 0]]),
            ([[7, 3, 7]], [[7, 3, 7]]),
        ], radius=0, color_mode="role")
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(program.apply([[4, 9, 4]]), ((4, 9, 4),))

    def test_role_mode_rejects_new_output_color(self) -> None:
        self.assertIsNone(fit_cellular_program(
            [([[0, 1]], [[0, 2]])], radius=0, color_mode="role"
        ))

    def test_radius_variants_are_bounded(self) -> None:
        programs = fit_cellular_programs([([[1]], [[1]])], max_radius=2)
        self.assertLessEqual(len(programs), 12)


if __name__ == "__main__":
    unittest.main()
