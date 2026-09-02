import unittest

from experiments.role_palette_transport import fit_role_palette_transports


class RolePaletteTransportTests(unittest.TestCase):
    def test_swaps_colors_without_color_id_assumption(self) -> None:
        pairs = [
            (
                ((1, 0, 2), (0, 0, 0)),
                ((2, 0, 1), (0, 0, 0)),
            ),
            (
                ((3, 0, 4), (0, 0, 0)),
                ((4, 0, 3), (0, 0, 0)),
            ),
        ]
        candidates = fit_role_palette_transports(pairs)
        self.assertTrue(any(candidate.permutation == (1, 0) for candidate in candidates))
        swapped = next(candidate for candidate in candidates if candidate.permutation == (1, 0))
        self.assertEqual(swapped.apply(((7, 0, 8), (0, 0, 0))),
                         ((8, 0, 7), (0, 0, 0)))

    def test_multicolor_objects_are_not_assumed_to_be_monochrome(self) -> None:
        pairs = [
            (((1, 2, 0), (0, 0, 0)), ((2, 1, 0), (0, 0, 0))),
        ]
        self.assertEqual(fit_role_palette_transports(pairs), ())

    def test_geometry_change_is_rejected(self) -> None:
        pairs = [
            (((1, 0), (0, 0)), ((0, 1), (0, 0))),
        ]
        self.assertEqual(fit_role_palette_transports(pairs), ())


if __name__ == "__main__":
    unittest.main()
