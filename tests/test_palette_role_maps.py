import unittest

from experiments.palette_role_maps import fit_role_map, fit_role_maps


class PaletteRoleMapTests(unittest.TestCase):
    def test_same_color_can_recolor_by_border_role(self) -> None:
        pairs = [
            (
                ((1, 0, 1), (0, 0, 0), (0, 0, 0)),
                ((2, 0, 3), (0, 0, 0), (0, 0, 0)),
            ),
            (
                ((1, 0, 1), (0, 0, 0), (0, 0, 0)),
                ((2, 0, 3), (0, 0, 0), (0, 0, 0)),
            ),
        ]
        candidate = fit_role_map(pairs, level=1, require_conditioned=True)
        self.assertIsNotNone(candidate)
        self.assertTrue(candidate.conditioned)
        self.assertEqual(candidate.apply(((1, 0, 1), (0, 0, 0), (0, 0, 0))),
                         ((2, 0, 3), (0, 0, 0), (0, 0, 0)))

    def test_rejects_global_map_when_conditioning_required(self) -> None:
        pairs = [
            (
                ((1, 0, 1), (0, 0, 0), (0, 0, 0)),
                ((2, 0, 2), (0, 0, 0), (0, 0, 0)),
            ),
        ]
        self.assertIsNone(fit_role_map(pairs, require_conditioned=True))

    def test_unseen_test_role_abstains(self) -> None:
        pairs = [
            (((1, 0), (0, 0)), ((2, 0), (0, 0))),
        ]
        candidate = fit_role_map(pairs, level=1, require_conditioned=False)
        self.assertIsNotNone(candidate)
        with self.assertRaises(ValueError):
            candidate.apply(((0, 0), (0, 1)))

    def test_all_levels_are_proof_gated(self) -> None:
        pairs = [
            (((1, 0, 1), (0, 0, 0)), ((2, 0, 3), (0, 0, 0))),
        ]
        self.assertTrue(all(candidate.conditioned for candidate in fit_role_maps(pairs)))


if __name__ == "__main__":
    unittest.main()
