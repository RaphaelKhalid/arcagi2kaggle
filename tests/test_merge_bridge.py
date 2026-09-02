from __future__ import annotations

import unittest

from experiments.merge_bridge import compile_bridge, execute_bridge


class MergeBridgeTests(unittest.TestCase):
    def test_horizontal_bridge_replays_and_transfers_geometry(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0, 1, 0]],
            "output": [[0, 1, 3, 1, 0]],
        }]}
        program = compile_bridge(task)
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(
            execute_bridge(program, [[0, 2, 0, 2, 0]]),
            ((0, 2, 3, 2, 0),),
        )

    def test_vertical_bridge_replays(self) -> None:
        task = {"train": [{
            "input": [[0, 1], [0, 0], [0, 1]],
            "output": [[0, 1], [0, 3], [0, 1]],
        }]}
        program = compile_bridge(task)
        self.assertIsNotNone(program)

    def test_diagonal_objects_are_not_silently_bridged(self) -> None:
        task = {"train": [{
            "input": [[1, 0], [0, 1]],
            "output": [[1, 0], [0, 1]],
        }]}
        self.assertIsNone(compile_bridge(task))


if __name__ == "__main__":
    unittest.main()
