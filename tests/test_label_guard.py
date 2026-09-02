from __future__ import annotations

import unittest

from experiments.label_guard import assert_label_free_hidden


class LabelGuardTests(unittest.TestCase):
    def test_none_is_allowed(self) -> None:
        assert_label_free_hidden({"h": {}})

    def test_disjoint_solution_mapping_is_allowed(self) -> None:
        assert_label_free_hidden({"h": {}}, {"evaluation": []})

    def test_overlapping_solution_mapping_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assert_label_free_hidden({"h": {}}, {"h": [[1]]})


if __name__ == "__main__":
    unittest.main()
