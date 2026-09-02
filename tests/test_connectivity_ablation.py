from __future__ import annotations

import unittest

from experiments.connectivity_ablation import (
    extract_objects,
    profile_grids,
)


class ConnectivityAblationTests(unittest.TestCase):
    def test_diagonal_touch_is_only_one_object_under_eight_connectivity(self) -> None:
        grid = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
        self.assertEqual(len(extract_objects(grid, connectivity=4)), 2)
        self.assertEqual(len(extract_objects(grid, connectivity=8)), 1)

    def test_orthogonal_components_agree(self) -> None:
        grid = [[1, 1, 0], [0, 0, 0], [0, 0, 2]]
        profile = profile_grids([grid])
        self.assertEqual(profile.changed_grids, 0)
        self.assertEqual(profile.diagonal_merges, 0)

    def test_invalid_connectivity_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            extract_objects([[1]], connectivity=6)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
