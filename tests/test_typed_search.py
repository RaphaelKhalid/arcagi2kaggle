import unittest

from experiments.typed_search import Primitive, enumerate_typed_paths


class TypedSearchTests(unittest.TestCase):
    def test_ill_typed_compositions_are_pruned(self):
        paths = enumerate_typed_paths(
            [
                Primitive("objects", "grid", "objects"),
                Primitive("flip", "grid", "grid"),
                Primitive("paint", "objects", "grid"),
            ],
            start_type="grid", goal_type="grid", max_cost=2,
        )
        self.assertEqual(
            [path.steps for path in paths],
            [("flip",), ("flip", "flip"), ("objects", "paint")],
        )

    def test_cost_order_is_deterministic(self):
        paths = enumerate_typed_paths(
            [Primitive("slow", "grid", "grid", 3), Primitive("fast", "grid", "grid", 1)],
            start_type="grid", goal_type="grid", max_cost=3,
        )
        self.assertEqual(
            [path.steps for path in paths[:2]],
            [("fast",), ("fast", "fast")],
        )

    def test_duplicate_symbolic_path_is_not_emitted_twice(self):
        paths = enumerate_typed_paths(
            [Primitive("flip", "grid", "grid"), Primitive("flip", "grid", "grid", 2)],
            start_type="grid", goal_type="grid", max_cost=2,
        )
        self.assertEqual([path.steps for path in paths], [("flip",), ("flip", "flip")])

    def test_invalid_bounds_and_costs_are_rejected(self):
        with self.assertRaises(ValueError):
            enumerate_typed_paths([], start_type="grid", goal_type="grid", max_cost=-1)
        with self.assertRaises(ValueError):
            enumerate_typed_paths(
                [Primitive("bad", "grid", "grid", 0)],
                start_type="grid", goal_type="grid", max_cost=1,
            )
