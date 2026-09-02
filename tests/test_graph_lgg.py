from __future__ import annotations

import unittest

from experiments.graph_lgg import (
    ActionObservation,
    lgg_observations,
    role_predicates,
)


class GraphLggTests(unittest.TestCase):
    def test_common_fields_survive_and_variable_fields_drop(self) -> None:
        first = ActionObservation(
            "move", "horizontal", "same", "same", "same",
            frozenset({("area", 1), ("border_left", True)}),
            frozenset({("area", 1)}),
        )
        second = ActionObservation(
            "move", "horizontal", "same", "expanded", "changed",
            frozenset({("area", 1), ("border_left", True)}),
            frozenset({("area", 2)}),
        )
        schema = lgg_observations(((first,), (second,)))
        self.assertIsNotNone(schema)
        assert schema is not None
        self.assertEqual(schema[0].kind, "move")
        self.assertEqual(schema[0].motion_axis, "horizontal")
        self.assertIsNone(schema[0].area_relation)
        self.assertEqual(schema[0].source_guard, frozenset({("area", 1), ("border_left", True)}))
        self.assertEqual(schema[0].target_guard, frozenset())

    def test_unequal_trace_lengths_are_not_generalized(self) -> None:
        observation = ActionObservation(
            "identity", "none", "same", "same", "same",
            frozenset(), frozenset(),
        )
        self.assertIsNone(lgg_observations(((observation,), ())))

    def test_role_predicates_are_id_free(self) -> None:
        predicates = role_predicates([[0, 1, 0], [0, 0, 2]], 0)
        self.assertIn(("area", 1), predicates)
        self.assertIn(("scene_degree", 1), predicates)
        self.assertNotIn(("color", 1), predicates)


if __name__ == "__main__":
    unittest.main()
