import unittest

from experiments.synthesis_gate import (
    TraceCandidate,
    compatible_union,
    complementarity,
    gate_synthesis,
)


class SynthesisGateTests(unittest.TestCase):
    def test_compatible_constraints_are_unioned(self):
        self.assertEqual(
            compatible_union({"role": "marker"}, {"effect": "move"}),
            {"role": "marker", "effect": "move"},
        )

    def test_conflicting_constraints_are_rejected(self):
        self.assertIsNone(compatible_union({"effect": "move"}, {"effect": "recolor"}))

    def test_complementarity_is_high_for_disjoint_clause_sets(self):
        self.assertAlmostEqual(complementarity({"a": "1"}, {"b": "2"}), 1.0)
        self.assertAlmostEqual(complementarity({"a": "1"}, {"a": "1"}), 0.0)

    def test_gate_selects_cross_family_compatible_pair_when_uncertain(self):
        decision = gate_synthesis(
            [
                TraceCandidate("text", "text", {"role": "marker"}),
                TraceCandidate("code", "code", {"effect": "move"}),
            ],
            unresolved_mass=0.9,
        )
        self.assertTrue(decision.should_synthesize)
        self.assertEqual(decision.pair, ("code", "text"))
        self.assertEqual(decision.merged_clauses, (("effect", "move"), ("role", "marker")))

    def test_gate_stops_when_verified_or_resolved(self):
        verified = gate_synthesis(
            [TraceCandidate("v", "dsl", {"x": "y"}, hard_verified=True)],
            unresolved_mass=1.0,
        )
        resolved = gate_synthesis(
            [TraceCandidate("a", "text", {"a": "1"}), TraceCandidate("b", "code", {"b": "2"})],
            unresolved_mass=0.1,
        )
        self.assertFalse(verified.should_synthesize)
        self.assertFalse(resolved.should_synthesize)
