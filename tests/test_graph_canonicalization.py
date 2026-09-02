import unittest

from experiments.graph_canonicalization import (
    LabeledGraph,
    canonical_signature,
    collision_safe_cache_key,
)


class GraphCanonicalizationTests(unittest.TestCase):
    def test_exact_signature_ignores_node_ids(self):
        left = LabeledGraph.from_parts(
            ("a", "b", "c"), {"a": "red", "b": "blue", "c": "red"},
            [("a", "b", "near"), ("b", "c", "far")],
        )
        right = LabeledGraph.from_parts(
            ("z", "x", "y"), {"x": "blue", "y": "red", "z": "red"},
            [("z", "x", "near"), ("x", "y", "far")],
        )
        self.assertEqual(canonical_signature(left), canonical_signature(right))

    def test_non_isomorphic_small_graphs_are_separated(self):
        path = LabeledGraph.from_parts(("a", "b", "c"), edges=[("a", "b", "e"), ("b", "c", "e")])
        triangle = LabeledGraph.from_parts(("a", "b", "c"), edges=[("a", "b", "e"), ("b", "c", "e"), ("a", "c", "e")])
        self.assertNotEqual(canonical_signature(path), canonical_signature(triangle))

    def test_wl_signature_is_invariant_for_larger_permuted_graph(self):
        nodes = tuple(f"n{i}" for i in range(9))
        edges = [(nodes[i], nodes[i + 1], "e") for i in range(8)]
        permuted = tuple(reversed(nodes))
        remapped_edges = [(permuted[i], permuted[i + 1], "e") for i in range(8)]
        left = LabeledGraph.from_parts(nodes, edges=edges)
        right = LabeledGraph.from_parts(permuted, edges=remapped_edges)
        self.assertEqual(canonical_signature(left, exact_limit=8), canonical_signature(right, exact_limit=8))

    def test_invalid_graph_references_and_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            LabeledGraph.from_parts(("a",), edges=[("a", "b", "e")])
        graph = LabeledGraph.from_parts(("a",))
        with self.assertRaises(ValueError):
            canonical_signature(graph, exact_limit=-1)

    def test_wl_bucket_is_not_advertised_as_collision_safe_cache_key(self):
        nodes = tuple(f"n{i}" for i in range(9))
        edges = [(nodes[i], nodes[i + 1], "e") for i in range(8)]
        graph = LabeledGraph.from_parts(nodes, edges=edges)
        self.assertIsNone(collision_safe_cache_key(graph, exact_limit=8))

    def test_exact_signature_is_available_as_collision_safe_cache_key(self):
        graph = LabeledGraph.from_parts(("a", "b"), edges=[("a", "b", "e")])
        self.assertEqual(
            collision_safe_cache_key(graph), canonical_signature(graph)
        )
