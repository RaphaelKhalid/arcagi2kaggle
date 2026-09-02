"""Guards against accidentally joining labeled outputs to hidden challenges."""

from __future__ import annotations

from typing import Mapping


def assert_label_free_hidden(
    hidden_challenges: Mapping[str, object],
    solutions: Mapping[str, object] | None = None,
) -> None:
    """Reject any solution mapping that can label a hidden task by ID."""

    if solutions is None:
        return
    overlap = sorted(set(hidden_challenges) & set(solutions))
    if overlap:
        raise ValueError(
            "hidden analysis received overlapping solution labels: "
            + ", ".join(overlap[:3])
            + ("..." if len(overlap) > 3 else "")
        )


if __name__ == "__main__":
    assert_label_free_hidden({"hidden": {}})
    print("label_guard selftest: PASS")
