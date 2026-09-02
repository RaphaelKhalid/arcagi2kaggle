from __future__ import annotations

import unittest

from experiments.frame_role_executor import (
    compile_frame_program,
    execute_frame_program,
)


class FrameRoleExecutorTests(unittest.TestCase):
    def test_composes_two_recolors(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0, 0, 2]],
            "output": [[0, 3, 0, 0, 4]],
        }]}
        program = compile_frame_program(task)
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(
            execute_frame_program(program, [[0, 5, 0, 0, 6]]),
            ((0, 3, 0, 0, 4),),
        )

    def test_composes_move_and_recolor(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0, 0, 2]],
            "output": [[0, 0, 1, 0, 3]],
        }]}
        program = compile_frame_program(task)
        self.assertIsNotNone(program)

    def test_transform_is_rejected(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0]],
            "output": [[0, 1, 1]],
        }]}
        self.assertIsNone(compile_frame_program(task))

    def test_connectivity_choice_is_stored_and_executable(self) -> None:
        task = {"train": [{
            "input": [[0, 1], [2, 0]],
            "output": [[0, 3], [4, 0]],
        }]}
        program = compile_frame_program(task, connectivity=8)
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(program.connectivity, 8)
        self.assertEqual(
            execute_frame_program(program, [[0, 5], [6, 0]]),
            ((0, 3), (4, 0)),
        )


if __name__ == "__main__":
    unittest.main()
