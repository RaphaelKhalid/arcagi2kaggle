from __future__ import annotations

import unittest
from collections import Counter

from experiments.structural_groups import (
    distribution_shift,
    task_features,
    task_position_features,
    total_variation,
)


class StructuralGroupTests(unittest.TestCase):
    def test_features_use_visible_inputs_and_are_stable(self) -> None:
        task = {
            "train": [{"input": [[0, 1], [0, 0]], "output": [[9]]}] * 2,
            "test": [{"input": [[0, 1], [1, 0]]}],
        }
        features = task_features(task)
        self.assertEqual(features["demos"], "le_2")
        self.assertEqual(features["tests"], "1")
        self.assertEqual(features["test_components"], "le_3")

    def test_position_features_distinguish_visible_test_inputs(self) -> None:
        task = {
            "train": [],
            "test": [
                {"input": [[0, 1]]},
                {"input": [[0, 1, 0, 1, 0]] * 6},
            ],
        }
        first = task_position_features(task, 0)
        second = task_position_features(task, 1)
        self.assertNotEqual(first["position_area"], second["position_area"])

    def test_total_variation_properties(self) -> None:
        self.assertEqual(total_variation(Counter({"x": 2}), Counter({"x": 1})), 0.0)
        self.assertEqual(total_variation(Counter({"x": 1}), Counter({"y": 1})), 1.0)

    def test_distribution_shift_is_featurewise(self) -> None:
        left = {"a": {"train": [], "test": [{"input": [[0]]}]}}
        right = {"b": {"train": [], "test": [{"input": [[0, 1, 0, 0, 0],
                                                     [0, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 0],
                                                     [0, 0, 0, 0, 0]]}]}}
        shift = distribution_shift(left, right)
        self.assertEqual(shift["test_area"], 1.0)


if __name__ == "__main__":
    unittest.main()
