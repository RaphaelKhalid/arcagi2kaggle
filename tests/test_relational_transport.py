from __future__ import annotations

import unittest

from experiments.relational_transport import (
    compile_transport,
    execute_transport,
)


class RelationalTransportTests(unittest.TestCase):
    def test_move_relative_to_reference(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0, 0, 2], [0, 0, 0, 0, 0]],
            "output": [[0, 0, 1, 0, 2], [0, 0, 0, 0, 0]],
        }]}
        program = compile_transport(task)
        self.assertIsNotNone(program)
        assert program is not None
        self.assertEqual(
            execute_transport(program, [[0, 3, 0, 0, 2], [0, 0, 0, 0, 0]]),
            ((0, 0, 3, 0, 2), (0, 0, 0, 0, 0)),
        )

    def test_no_stationary_reference_is_rejected(self) -> None:
        task = {"train": [{
            "input": [[0, 1, 0]],
            "output": [[0, 0, 1]],
        }]}
        self.assertIsNone(compile_transport(task))


if __name__ == "__main__":
    unittest.main()
