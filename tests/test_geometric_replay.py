from __future__ import annotations

import unittest

from experiments.geometric_replay import d8_transforms, rotate_clockwise


class GeometricReplayTests(unittest.TestCase):
    def test_rotation_is_clockwise_and_shape_aware(self) -> None:
        self.assertEqual(rotate_clockwise(((1, 2, 3), (4, 5, 6))),
                         ((4, 1), (5, 2), (6, 3)))

    def test_d8_has_eight_named_members(self) -> None:
        transforms = d8_transforms([[1, 0], [0, 1]])
        self.assertEqual(len(transforms), 8)
        self.assertEqual(set(transforms), {
            "rot0", "rot90", "rot180", "rot270",
            "flip_rot0", "flip_rot90", "flip_rot180", "flip_rot270",
        })

    def test_d8_preserves_cell_multiset(self) -> None:
        transforms = d8_transforms([[1, 0, 2], [0, 3, 0]])
        expected = sorted([1, 2, 3])
        for grid in transforms.values():
            self.assertEqual(sorted(cell for row in grid for cell in row
                                    if cell != 0), expected)


if __name__ == "__main__":
    unittest.main()
