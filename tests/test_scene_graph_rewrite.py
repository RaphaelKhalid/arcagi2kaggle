from __future__ import annotations

import unittest

from experiments.scene_graph_rewrite import (
    compile_scene_rewrite,
    execute_scene_rewrite,
)


class SceneGraphRewriteTests(unittest.TestCase):
    def test_reference_anchored_addition_replays(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 2]],
            "output": [[0, 1, 0, 3, 0], [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 2]],
        }]}
        program = compile_scene_rewrite(task)
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(
            execute_scene_rewrite(
                program, [[0, 4, 0, 0, 0], [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 5]]
            ),
            ((0, 4, 0, 3, 0), (0, 0, 0, 0, 0),
             (0, 0, 0, 0, 5)),
        )

    def test_transform_is_grounded_and_replays(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0], [0, 0, 0]],
            "output": [[0, 1, 0], [0, 1, 1]],
        }]}
        program = compile_scene_rewrite(task)
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(
            execute_scene_rewrite(program, [[0, 2, 0], [0, 0, 0]]),
            ((0, 1, 0), (0, 1, 1)),
        )

    def test_ambiguous_reference_is_rejected(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0, 1]],
            "output": [[0, 1, 3, 1]],
        }]}
        self.assertIsNone(compile_scene_rewrite(task))


if __name__ == "__main__":
    unittest.main()
