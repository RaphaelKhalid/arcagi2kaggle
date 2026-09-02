"""Permutation-invariant scene-graph signatures for ARC role search.

Object IDs are implementation artifacts.  Quotienting a scene graph by node
renaming prevents a correspondence search from spending probability mass on
the same explanation under many arbitrary labels.  For small graphs an exact
canonical label is affordable; larger graphs use Weisfeiler--Leman refinement,
which is an invariant but not a complete graph-isomorphism test.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import permutations
from typing import Iterable


@dataclass(frozen=True)
class LabeledGraph:
    nodes: tuple[str, ...]
    node_labels: tuple[tuple[str, str], ...]
    edges: tuple[tuple[str, str, str], ...]

    @classmethod
    def from_parts(
        cls,
        nodes: Iterable[str],
        node_labels: dict[str, str] | None = None,
        edges: Iterable[tuple[str, str, str]] = (),
    ) -> "LabeledGraph":
        nodes = tuple(nodes)
        if len(set(nodes)) != len(nodes):
            raise ValueError("node IDs must be unique")
        known = set(nodes)
        labels = node_labels or {}
        if set(labels) - known:
            raise ValueError("node label references an unknown node")
        normalized_edges = tuple(sorted(tuple(edge) for edge in edges))
        if any(len(edge) != 3 or edge[0] not in known or edge[1] not in known
               for edge in normalized_edges):
            raise ValueError("edge references an unknown node")
        return cls(
            nodes=nodes,
            node_labels=tuple(sorted((node, labels.get(node, "")) for node in nodes)),
            edges=normalized_edges,
        )


def _labels(graph: LabeledGraph) -> dict[str, str]:
    return dict(graph.node_labels)


def _edge_map(graph: LabeledGraph) -> dict[tuple[str, str], tuple[str, ...]]:
    result: dict[tuple[str, str], list[str]] = {}
    for left, right, label in graph.edges:
        key = tuple(sorted((left, right)))
        result.setdefault(key, []).append(label)
    return {key: tuple(sorted(value)) for key, value in result.items()}


def _exact_signature(graph: LabeledGraph) -> tuple:
    labels = _labels(graph)
    edges = _edge_map(graph)
    best = None
    for order in permutations(graph.nodes):
        node_part = tuple(labels[node] for node in order)
        edge_part = tuple(
            edges.get(tuple(sorted((order[i], order[j]))), ())
            for i in range(len(order))
            for j in range(i + 1, len(order))
        )
        code = (node_part, edge_part)
        if best is None or code < best:
            best = code
    return best or ((), ())


def _wl_signature(graph: LabeledGraph, rounds: int) -> tuple:
    labels = _labels(graph)
    edges = _edge_map(graph)
    colors = {node: labels[node] for node in graph.nodes}
    for _ in range(rounds):
        refined = {}
        for node in graph.nodes:
            neighborhood = []
            for (left, right), edge_labels in edges.items():
                if node not in (left, right):
                    continue
                other = right if left == node else left
                neighborhood.extend((edge_label, colors[other])
                                    for edge_label in edge_labels)
            payload = repr((colors[node], tuple(sorted(neighborhood)))).encode("utf-8")
            refined[node] = sha256(payload).hexdigest()
        colors = refined
    edge_part = tuple(sorted(
        (colors[left], colors[right], edge_labels)
        for (left, right), edge_labels in edges.items()
    ))
    return (tuple(sorted(colors.values())), edge_part)


def canonical_signature(
    graph: LabeledGraph,
    *,
    exact_limit: int = 8,
    rounds: int = 3,
) -> tuple:
    """Return an ID-invariant signature, exact below ``exact_limit`` nodes."""

    if exact_limit < 0 or rounds < 0:
        raise ValueError("canonicalization bounds must be non-negative")
    if len(graph.nodes) <= exact_limit:
        return ("exact", _exact_signature(graph))
    return ("wl", _wl_signature(graph, rounds))


def collision_safe_cache_key(
    graph: LabeledGraph,
    *,
    exact_limit: int = 8,
    rounds: int = 3,
) -> tuple | None:
    """Return a merge-safe key, or ``None`` when only a WL bucket is known.

    Weisfeiler--Leman collisions are possible for non-isomorphic graphs.  A
    caller may use :func:`canonical_signature` to route a search into a cheap
    bucket, but may only merge cached scene parses when an exact canonical
    label is available (or after a separate exact isomorphism check).
    """

    signature = canonical_signature(
        graph, exact_limit=exact_limit, rounds=rounds
    )
    return signature if signature[0] == "exact" else None


if __name__ == "__main__":
    graph = LabeledGraph.from_parts(("a", "b"), {"a": "x", "b": "y"}, [("a", "b", "near")])
    assert canonical_signature(graph) == canonical_signature(
        LabeledGraph.from_parts(("b", "a"), {"a": "x", "b": "y"}, [("b", "a", "near")])
    )
    print("graph_canonicalization selftest: PASS")
