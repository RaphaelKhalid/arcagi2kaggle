from __future__ import annotations

import unittest

from experiments.graph_lgg import ActionSchema
from experiments.guarded_roles import (
    schema_has_unique_roles,
    select_roles,
    top1_lgg_for_task,
)


class GuardedRoleTests(unittest.TestCase):
    def test_guard_selects_single_object(self) -> None:
        guard = frozenset({("area", 1), ("scene_degree", 1), ("border_top", True)})
        self.assertEqual(select_roles([[0, 1, 0], [0, 0, 2]], guard), (0,))

    def test_empty_guard_is_not_a_proof(self) -> None:
        schema = ActionSchema(
            "move", "horizontal", "same", "same", "same",
            frozenset(), frozenset(),
        )
        self.assertFalse(schema_has_unique_roles(
            [[0, 1]], [[0, 1]], schema
        ))

    def test_single_action_task_builds_lgg(self) -> None:
        result = top1_lgg_for_task({
            "train": [{
                "input": [[0, 1], [0, 0]],
                "output": [[0, 2], [0, 0]],
            }]
        })
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result[0]), 1)


if __name__ == "__main__":
    unittest.main()
