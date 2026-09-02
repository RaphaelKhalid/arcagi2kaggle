import unittest

from experiments.guarded_programs import GuardedBranch
from experiments.guard_search import (
    NamedAction,
    feature_predicates,
    search_one_guard,
)


class GuardSearchTests(unittest.TestCase):
    @staticmethod
    def identity(grid):
        return grid

    @staticmethod
    def flip(grid):
        return tuple(tuple(reversed(row)) for row in grid)

    def test_feature_vocabulary_is_finite_and_deterministic(self):
        inputs = [[[0, 2, 0]], [[1, 0, 0]]]
        names = [item.name for item in feature_predicates(inputs)]
        self.assertEqual(names, [item.name for item in feature_predicates(inputs)])
        self.assertIn("contains=2", names)
        self.assertIn("width=3", names)

    def test_bounded_search_recovers_contextual_rule(self):
        actions = [
            NamedAction("flip", self.flip),
            NamedAction("identity", self.identity),
        ]
        result = search_one_guard(
            [([[0, 2, 1]], [[1, 2, 0]]), ([[1, 0, 0]], [[1, 0, 0]])],
            feature_predicates([[[0, 2, 1]], [[1, 0, 0]]]),
            actions,
        )
        self.assertTrue(result)
        self.assertTrue(any(
            item[0].branches[0].guard_name == "contains=2"
            and item[0].branches[0].action_name == "flip"
            and item[0].fallback_name == "identity"
            for item in result
        ))

    def test_search_bound_is_enforced(self):
        actions = [NamedAction("identity", self.identity)]
        with self.assertRaises(ValueError):
            search_one_guard([], [], actions, max_candidates=0)


if __name__ == "__main__":
    unittest.main()
