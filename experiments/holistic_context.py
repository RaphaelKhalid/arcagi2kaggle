"""Deterministic context allocation for a bounded holistic ARC judge.

The judge should see enough raw reasoning to distinguish competing rules, but
naively packing traces lets a large correlated generation batch crowd out a
rare, potentially correct output class.  This module treats exact output
classes as coverage constraints and uses remaining space for provenance
diversity.  It does not decide correctness and never creates a grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


@dataclass(frozen=True)
class TraceEvidence:
    """One bounded reasoning trace attached to an existing output class."""

    candidate_id: str
    output_hash: str
    family: str
    trace: str
    lineage: str | None = None
    priority: float = 0.0


@dataclass(frozen=True)
class TraceSelection:
    """Selected traces and accounting needed by a prompt builder."""

    selected: tuple[TraceEvidence, ...]
    covered_output_hashes: tuple[str, ...]
    used_chars: int


def render_trace_evidence(item: TraceEvidence, *, trace_chars: int) -> str:
    """Render one trace with an explicit, deterministic character cap."""

    if trace_chars <= 0:
        raise ValueError("trace_chars must be positive")
    trace = str(item.trace).strip()[:trace_chars]
    lineage = "?" if item.lineage is None else item.lineage
    return (
        f"TRACE candidate={item.candidate_id} output={item.output_hash} "
        f"family={item.family} lineage={lineage}\n{trace}\n"
    )


def select_trace_evidence(
    evidence: Iterable[TraceEvidence],
    *,
    max_chars: int,
    per_trace_chars: int = 2_000,
    required_output_hashes: Iterable[str] | None = None,
) -> TraceSelection:
    """Select traces while guaranteeing one witness per output class.

    The first phase chooses the highest-priority canonical witness for every
    output hash.  If those witnesses cannot fit, the caller must use a
    cards-only prompt rather than silently losing a minority class.  The
    remaining budget is filled greedily by marginal family/lineage coverage,
    then priority and canonical identifiers.  This is a context policy, not a
    semantic correctness score.
    """

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if per_trace_chars <= 0:
        raise ValueError("per_trace_chars must be positive")

    required = tuple(sorted(set(required_output_hashes or ())))
    items = [item for item in evidence if str(item.trace).strip()]
    if not items:
        if required:
            raise ValueError(
                "trace evidence is missing required output classes: "
                + ", ".join(required)
            )
        return TraceSelection((), (), 0)

    def canonical(item: TraceEvidence) -> tuple[float, str, str, str]:
        return (
            -float(item.priority),
            item.family,
            "" if item.lineage is None else item.lineage,
            item.candidate_id,
        )

    groups: dict[str, list[TraceEvidence]] = {}
    for item in items:
        if not item.output_hash or not item.candidate_id or not item.family:
            raise ValueError("trace evidence identifiers must be non-empty")
        groups.setdefault(item.output_hash, []).append(item)
    missing = [output_hash for output_hash in required if output_hash not in groups]
    if missing:
        raise ValueError(
            "trace evidence is missing required output classes: "
            + ", ".join(missing)
        )
    for group in groups.values():
        group.sort(key=canonical)

    def block(item: TraceEvidence) -> str:
        return render_trace_evidence(item, trace_chars=per_trace_chars)

    # Exact output classes are the hard coverage constraint.  Do this before
    # looking at priority so a frequent correlated lineage cannot crowd out a
    # structurally different minority hypothesis.
    mandatory = [groups[key][0] for key in sorted(groups)]
    mandatory_chars = sum(len(block(item)) for item in mandatory)
    if mandatory_chars > max_chars:
        raise ValueError(
            "trace budget cannot cover one witness for every output class"
        )

    selected = list(mandatory)
    selected_keys = {(item.output_hash, item.candidate_id) for item in selected}
    used_chars = mandatory_chars
    families = {item.family for item in selected}
    lineages = {item.lineage for item in selected if item.lineage is not None}
    output_families = {(item.output_hash, item.family) for item in selected}
    remaining = [
        item for item in items
        if (item.output_hash, item.candidate_id) not in selected_keys
    ]

    while remaining:
        feasible = [item for item in remaining if used_chars + len(block(item)) <= max_chars]
        if not feasible:
            break

        def marginal_key(item: TraceEvidence) -> tuple[float, float, str, str, str]:
            new_family = float(item.family not in families)
            new_lineage = float(
                item.lineage is not None and item.lineage not in lineages
            )
            new_pair = float((item.output_hash, item.family) not in output_families)
            utility = 2.0 * new_family + new_lineage + new_pair
            return (
                -utility,
                -float(item.priority),
                item.output_hash,
                item.family,
                item.candidate_id,
            )

        chosen = min(feasible, key=marginal_key)
        selected.append(chosen)
        remaining.remove(chosen)
        used_chars += len(block(chosen))
        families.add(chosen.family)
        if chosen.lineage is not None:
            lineages.add(chosen.lineage)
        output_families.add((chosen.output_hash, chosen.family))

    ordered = tuple(sorted(selected, key=lambda item: (item.output_hash, item.family, item.candidate_id)))
    return TraceSelection(
        selected=ordered,
        covered_output_hashes=tuple(sorted({item.output_hash for item in ordered})),
        used_chars=used_chars,
    )


if __name__ == "__main__":
    print("holistic_context selftest: PASS")
