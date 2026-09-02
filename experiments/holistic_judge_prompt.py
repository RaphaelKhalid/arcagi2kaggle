"""Bounded offline prompt/parse contract for holistic ARC candidate judging.

This is a selection seam, not a model or a verifier.  It presents already
deduplicated candidate cards and their exact output grids to a judge, then
accepts only two existing output-class hashes.  A future implementation may
attach model reasoning traces, but the final parser never accepts an invented
grid from the judge.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence

from experiments.candidate_cards import CandidateCard
from experiments.candidate_records import grid_hash, normalize_grid
from experiments.holistic_context import (
    TraceEvidence,
    render_trace_evidence,
    select_trace_evidence,
)
from experiments.repair_prompt import render_grid


Grid = Sequence[Sequence[int]]


def build_holistic_judge_prompt(
    cards: Iterable[CandidateCard],
    outputs: Mapping[str, Grid],
    *,
    max_chars: int = 12_000,
    trace_evidence: Iterable[TraceEvidence] | None = None,
    per_trace_chars: int = 2_000,
) -> str:
    """Build a deterministic compact judge bundle within a context budget."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    ordered = tuple(sorted(cards, key=lambda card: card.output_hash))
    if not ordered:
        raise ValueError("at least one candidate card is required")
    lines = [
        "Select the two most likely correct existing ARC output classes.",
        "Use the provenance and structural evidence, but do not invent a new grid.",
        'Return only JSON: {"top_two": [full_hash_1, full_hash_2]}.',
        "The two hashes must be distinct and must appear below.",
        "",
    ]
    for card in ordered:
        if card.output_hash not in outputs:
            raise ValueError(f"missing output for card {card.output_hash}")
        grid = normalize_grid(outputs[card.output_hash])
        if grid_hash(grid) != card.output_hash:
            raise ValueError(f"output hash does not match card {card.output_hash}")
        grid_text = render_grid(grid)
        lines.extend([
            f"CANDIDATE {card.output_hash}",
            card.prompt_line(),
            "exact_output:",
            grid_text,
            "",
        ])
    prompt = "\n".join(lines)
    if trace_evidence is not None:
        trace_budget = max_chars - len(prompt) - len("\nTRACE EVIDENCE\n")
        selection = select_trace_evidence(
            trace_evidence,
            max_chars=trace_budget,
            per_trace_chars=per_trace_chars,
            required_output_hashes=(card.output_hash for card in ordered),
        )
        if selection.selected:
            prompt += "\nTRACE EVIDENCE\n"
            prompt += "".join(
                render_trace_evidence(item, trace_chars=per_trace_chars)
                for item in selection.selected
            )
    if len(prompt) > max_chars:
        raise ValueError("judge bundle exceeds context budget")
    return prompt


def parse_judge_choice(raw: str, known_hashes: Iterable[str]) -> tuple[str, str]:
    """Parse and validate a judge's exact two-class JSON response."""

    known = set(known_hashes)
    if len(known) < 2:
        raise ValueError("at least two known output classes are required")
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("judge response must be JSON") from exc
    choice = payload.get("top_two") if isinstance(payload, dict) else None
    if not isinstance(choice, list) or len(choice) != 2:
        raise ValueError("judge must return exactly two classes")
    if any(not isinstance(value, str) or value not in known for value in choice):
        raise ValueError("judge returned an unknown output class")
    if choice[0] == choice[1]:
        raise ValueError("judge classes must be distinct")
    return choice[0], choice[1]


if __name__ == "__main__":
    print("holistic_judge_prompt selftest: PASS")
