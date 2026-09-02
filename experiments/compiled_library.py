"""Verified, invariant-indexed transform-library prototype.

This module is intentionally model-agnostic.  It indexes executable program
records by a compact, role-normalized scene signature and returns only records
that have already reproduced every demonstration.  It does not contain ARC
answers and it does not infer a program from a raw task by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Signature = tuple[int, ...]


@dataclass(frozen=True)
class LibraryEntry:
    """A compiled transform plus its offline verification certificate."""

    entry_id: str
    signature: Signature
    family: str
    program_hash: str
    mdl_length: float
    demo_exact: bool = False


def signature_distance(left: Signature, right: Signature) -> int:
    """Hamming distance with ``-1`` as an unknown/wildcard feature."""

    distance = abs(len(left) - len(right))
    for left_value, right_value in zip(left, right):
        if left_value != -1 and right_value != -1 and left_value != right_value:
            distance += 1
    return distance


def retrieve_verified(
    entries: Iterable[LibraryEntry],
    query_signature: Signature,
    *,
    limit: int = 8,
    max_distance: int | None = None,
) -> tuple[LibraryEntry, ...]:
    """Retrieve compact verified programs for a role-normalized task.

    The verification bit is a hard gate.  Signature similarity ranks already
    safe candidates; it never turns a near match into a prediction.  Ties are
    resolved by shorter description length and then stable ID, making replay
    deterministic.
    """

    if limit < 0:
        raise ValueError("limit must be non-negative")
    ranked: list[tuple[int, LibraryEntry]] = []
    for entry in entries:
        if not entry.demo_exact:
            continue
        distance = signature_distance(entry.signature, query_signature)
        if max_distance is not None and distance > max_distance:
            continue
        ranked.append((distance, entry))
    ranked.sort(key=lambda item: (item[0], item[1].mdl_length, item[1].entry_id))
    return tuple(entry for _, entry in ranked[:limit])


if __name__ == "__main__":
    query = (3, 2, 1, 0)
    result = retrieve_verified(
        [
            LibraryEntry("unverified-exact", query, "program", "bad", 1, False),
            LibraryEntry("verified-near", (3, 2, 1, 1), "program", "near", 2, True),
            LibraryEntry("verified-exact", query, "program", "exact", 4, True),
        ],
        query,
    )
    assert [entry.entry_id for entry in result] == ["verified-exact", "verified-near"]
    print("compiled_library selftest: PASS", [entry.entry_id for entry in result])
