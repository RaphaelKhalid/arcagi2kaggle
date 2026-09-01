"""Tests for arc.tasks grid helpers and loaders."""

import json
import tempfile
import unittest
from pathlib import Path

from arc.tasks import (
    Task,
    grid_dims,
    grid_error,
    grid_palette,
    grids_equal,
    is_valid_grid,
    load_tasks,
)


class GridHelperTests(unittest.TestCase):
    def test_valid_grids(self) -> None:
        self.assertTrue(is_valid_grid([[0]]))
        self.assertTrue(is_valid_grid([[1, 2], [3, 4]]))
        self.assertTrue(is_valid_grid([[9] * 30 for _ in range(30)]))

    def test_invalid_type(self) -> None:
        self.assertIn("must be a list", grid_error("nope"))
        self.assertIn("must be a list", grid_error(None))
        self.assertIn("row 0 must be a list", grid_error([1, 2]))

    def test_empty_and_oversized(self) -> None:
        self.assertIn("rows", grid_error([]))
        self.assertIn("columns", grid_error([[]]))
        self.assertIn("rows", grid_error([[0]] * 31))
        self.assertIn("columns", grid_error([[0] * 31]))

    def test_ragged(self) -> None:
        self.assertIn("ragged", grid_error([[1, 2], [3]]))

    def test_bad_cells(self) -> None:
        self.assertIn("must be an int", grid_error([[1, "2"]]))
        self.assertIn("must be an int", grid_error([[1.0]]))
        self.assertIn("must be an int", grid_error([[True]]))
        self.assertIn("0-9", grid_error([[10]]))
        self.assertIn("0-9", grid_error([[-1]]))

    def test_dims_palette_equal(self) -> None:
        g = [[1, 2, 2], [3, 1, 0]]
        self.assertEqual(grid_dims(g), (2, 3))
        self.assertEqual(grid_palette(g), frozenset({0, 1, 2, 3}))
        self.assertTrue(grids_equal(g, [[1, 2, 2], [3, 1, 0]]))
        self.assertFalse(grids_equal(g, [[1, 2], [3, 1]]))


class LoadTasksTests(unittest.TestCase):
    def _write(self, tmp: Path, name: str, obj: object) -> Path:
        path = tmp / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def test_load_with_and_without_solutions(self) -> None:
        challenges = {
            "aaa": {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]]}, {"input": [[4]]}],
            }
        }
        solutions = {"aaa": [[[5]], [[6]]]}
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ch = self._write(tmp, "ch.json", challenges)
            so = self._write(tmp, "so.json", solutions)

            tasks = load_tasks(ch)
            self.assertEqual(list(tasks), ["aaa"])
            task = tasks["aaa"]
            self.assertIsInstance(task, Task)
            self.assertEqual(task.num_test, 2)
            self.assertIsNone(task.test_outputs)

            tasks = load_tasks(ch, so)
            self.assertEqual(tasks["aaa"].test_outputs, [[[5]], [[6]]])

    def test_load_rejects_mismatched_solutions(self) -> None:
        challenges = {"aaa": {"train": [], "test": [{"input": [[1]]}]}}
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ch = self._write(tmp, "ch.json", challenges)
            so_missing = self._write(tmp, "so1.json", {})
            so_short = self._write(tmp, "so2.json", {"aaa": []})
            with self.assertRaises(ValueError):
                load_tasks(ch, so_missing)
            with self.assertRaises(ValueError):
                load_tasks(ch, so_short)

    def test_load_rejects_malformed_grid(self) -> None:
        challenges = {"aaa": {"train": [], "test": [{"input": [[1, 99]]}]}}
        with tempfile.TemporaryDirectory() as d:
            ch = self._write(Path(d), "ch.json", challenges)
            with self.assertRaises(ValueError):
                load_tasks(ch)


if __name__ == "__main__":
    unittest.main()
