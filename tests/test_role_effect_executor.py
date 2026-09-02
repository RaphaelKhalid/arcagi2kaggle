from __future__ import annotations

import unittest

from experiments.role_effect_executor import (
    compile_role_effect,
    execute_role_effect,
)


class RoleEffectExecutorTests(unittest.TestCase):
    def test_move_uses_selected_role_not_literal_grid_position(self) -> None:
        task = {"train": [
            {"input": [[0, 1, 0], [0, 0, 0]],
             "output": [[0, 0, 1], [0, 0, 0]]},
            {"input": [[0, 0, 0], [2, 0, 0]],
             "output": [[0, 0, 0], [0, 2, 0]]},
        ]}
        program = compile_role_effect(task)
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(
            execute_role_effect(program, [[0, 0, 0], [3, 0, 0]]),
            ((0, 0, 0), (0, 3, 0)),
        )

    def test_recolor_is_closed(self) -> None:
        task = {"train": [{
            "input": [[0, 1], [0, 0]],
            "output": [[0, 2], [0, 0]],
        }]}
        self.assertIsNotNone(compile_role_effect(task))

    def test_ambiguous_role_is_rejected(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0, 1]],
            "output": [[0, 2, 0, 2]],
        }]}
        self.assertIsNone(compile_role_effect(task))


if __name__ == "__main__":
    unittest.main()
