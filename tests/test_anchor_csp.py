from __future__ import annotations

import unittest

from experiments.anchor_csp import (
    AnchorSystem,
    RelationConstraint,
    solve_anchor_system,
)


class AnchorCspTests(unittest.TestCase):
    def test_fixed_reference_can_make_relation_unique(self) -> None:
        system = AnchorSystem(
            grid_shape=(1, 5),
            shapes={"A": ((0, 0),), "B": ((0, 0),)},
            relations=(RelationConstraint(
                "A", "B", ((0, -1), "horizontal", 1)
            ),),
            fixed={"B": (0, 4)},
        )
        result = solve_anchor_system(system)
        self.assertEqual(result.solutions, ({"A": (0, 3), "B": (0, 4)},))

    def test_ambiguous_system_returns_multiple_solutions(self) -> None:
        system = AnchorSystem(
            grid_shape=(3, 3),
            shapes={"A": ((0, 0),), "B": ((0, 0),)},
            fixed={"B": (1, 1)},
        )
        result = solve_anchor_system(system, max_solutions=2)
        self.assertEqual(len(result.solutions), 2)

    def test_blocked_cells_are_frame_constraints(self) -> None:
        system = AnchorSystem(
            grid_shape=(1, 4),
            shapes={"A": ((0, 0),)},
            blocked=frozenset({(0, 1)}),
        )
        result = solve_anchor_system(system, max_solutions=10)
        self.assertNotIn({"A": (0, 1)}, result.solutions)


if __name__ == "__main__":
    unittest.main()
