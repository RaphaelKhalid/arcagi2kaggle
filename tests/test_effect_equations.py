from __future__ import annotations

import unittest

from experiments.effect_equations import (
    AnchorObservation,
    ConstantOffset,
    RelativeRoleOffset,
    RolePlusOffset,
    fit_equations,
)


class EffectEquationTests(unittest.TestCase):
    def test_constant_offset_fits(self) -> None:
        observations = (
            AnchorObservation("A", (1, 2), {"A": (0, 0), "B": (4, 4)}),
            AnchorObservation("A", (5, 3), {"A": (4, 1), "B": (2, 8)}),
        )
        equations = fit_equations(observations)
        self.assertIn(ConstantOffset((1, 2)), equations)

    def test_relative_role_offset_fits_varying_layout(self) -> None:
        observations = (
            AnchorObservation("A", (1, 1), {"A": (0, 0), "B": (1, 1), "C": (0, 0)}),
            AnchorObservation("A", (7, 4), {"A": (5, 2), "B": (3, 4), "C": (1, 2)}),
        )
        equations = fit_equations(observations)
        self.assertIn(RelativeRoleOffset("B", "C"), equations)

    def test_reference_offset_fits(self) -> None:
        observations = (
            AnchorObservation("A", (2, 3), {"A": (0, 0), "B": (2, 2)}),
            AnchorObservation("A", (5, 1), {"A": (9, 9), "B": (5, 0)}),
        )
        self.assertIn(RolePlusOffset("B", (0, 1)), fit_equations(observations))

    def test_inconsistent_equations_are_rejected(self) -> None:
        observations = (
            AnchorObservation("A", (1, 0), {"A": (0, 0), "B": (1, 1)}),
            AnchorObservation("A", (4, 4), {"A": (2, 2), "B": (3, 3)}),
        )
        self.assertFalse(fit_equations(observations))


if __name__ == "__main__":
    unittest.main()
