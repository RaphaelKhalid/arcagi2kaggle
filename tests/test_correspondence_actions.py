from __future__ import annotations

import unittest

from experiments.correspondence_actions import (
    action_family,
    actions_for_correspondence,
    correspondence_action_families,
    task_action_consensus,
)
from experiments.object_correspondence import top_k_correspondences
from experiments.object_correspondence import Correspondence


class CorrespondenceActionTests(unittest.TestCase):
    def test_move_is_typed(self) -> None:
        source = [[0, 1, 0], [0, 0, 0]]
        target = [[0, 0, 1], [0, 0, 0]]
        correspondence = top_k_correspondences(source, target)[0]
        actions = actions_for_correspondence(source, target, correspondence)
        self.assertEqual(actions[0].kind, "move")
        self.assertEqual(actions[0].displacement, (0, 1))

    def test_recolor_is_typed(self) -> None:
        source = [[0, 1], [0, 0]]
        target = [[0, 2], [0, 0]]
        families = correspondence_action_families(source, target)
        self.assertTrue(families)
        self.assertEqual(families[0][0][0], "recolor")

    def test_unmatched_objects_become_actions(self) -> None:
        source = [[0, 1, 0, 0, 0]]
        target = [[0, 1, 0, 2, 0]]
        correspondence = top_k_correspondences(source, target)[0]
        kinds = {action.kind for action in actions_for_correspondence(
            source, target, correspondence
        )}
        self.assertIn("add", kinds)

    def test_consensus_requires_every_demo(self) -> None:
        task = {
            "train": [
                {"input": [[0, 1], [0, 0]], "output": [[0, 2], [0, 0]]},
                {"input": [[0, 1, 0], [0, 0, 0]],
                 "output": [[0, 2, 0], [0, 0, 0]]},
            ]
        }
        result = task_action_consensus(task)
        self.assertEqual(result["n_demos"], 2)
        self.assertTrue(result["stable_families"])

    def test_family_omits_object_indices(self) -> None:
        source = [[0, 1], [0, 0]]
        target = [[0, 2], [0, 0]]
        correspondence = top_k_correspondences(source, target)[0]
        family = action_family(actions_for_correspondence(
            source, target, correspondence
        ))
        self.assertNotIn(0, family[0])

    def test_index_validation_checks_each_coordinate(self) -> None:
        with self.assertRaises(ValueError):
            actions_for_correspondence(
                [[0, 1], [0, 0]],
                [[0, 1], [0, 0]],
                Correspondence(
                    pairs=((0, 0),),
                    unmatched_source=(),
                    unmatched_target=(1,),
                    cost=0,
                ),
            )


if __name__ == "__main__":
    unittest.main()
