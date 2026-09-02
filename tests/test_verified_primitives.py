from __future__ import annotations

import unittest

from experiments.verified_primitives import (
    crop_content,
    fit_verified_primitives,
    gravity,
    tile,
    upscale,
)


class VerifiedPrimitiveTests(unittest.TestCase):
    def test_basic_operators(self) -> None:
        self.assertEqual(tile(((1, 2),), 2, 2), ((1, 2, 1, 2), (1, 2, 1, 2)))
        self.assertEqual(upscale(((1, 2),), 2, 2), ((1, 1, 2, 2), (1, 1, 2, 2)))
        self.assertEqual(crop_content(((0, 0, 0), (0, 2, 0), (0, 0, 0))), ((2,),))

    def test_gravity_preserves_non_background_order(self) -> None:
        self.assertEqual(gravity(((0, 1, 0), (2, 0, 0)), "right"),
                         ((0, 0, 1), (0, 0, 2)))

    def test_color_map_is_verified_across_all_demos(self) -> None:
        fitted = fit_verified_primitives([
            ([[0, 1]], [[0, 2]]),
            ([[1, 0]], [[2, 0]]),
        ])
        self.assertTrue(any(primitive.family == "palette" for primitive in fitted))

    def test_inconsistent_color_map_is_rejected(self) -> None:
        fitted = fit_verified_primitives([
            ([[0, 1]], [[0, 2]]),
            ([[0, 1]], [[0, 3]]),
        ])
        self.assertFalse(any(primitive.family == "palette" for primitive in fitted))


if __name__ == "__main__":
    unittest.main()
