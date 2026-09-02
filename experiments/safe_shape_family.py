"""Empirical release allowlist for high-precision size hypotheses.

The names are deliberately an allowlist.  A rule that fits demonstrations is
not automatically safe to use as a hard decoder mask; families with observed
extrapolation failures remain outside this module until re-audited.
"""

from __future__ import annotations

from typing import Iterable, Sequence


RELEASE_SHAPE_RULE_ALLOWLIST = frozenset(
    {
        "same_as_input",
        "transpose",
        "largest_obj_4",
        "largest_obj_8",
        "smallest_obj_4",
        "smallest_obj_8",
    }
)


def filter_size_candidates(
    candidates: Iterable[tuple[str, tuple[int, int]]],
    *,
    allowed_rules: frozenset[str] = RELEASE_SHAPE_RULE_ALLOWLIST,
) -> tuple[tuple[str, tuple[int, int]], ...]:
    """Keep only candidates from the empirically gated rule family."""

    return tuple(candidate for candidate in candidates if candidate[0] in allowed_rules)


def agreed_size(
    candidates: Sequence[tuple[str, tuple[int, int]]],
) -> tuple[int, int] | None:
    """Return a size only when all retained rules agree on one size."""

    sizes = {size for _, size in candidates}
    if len(sizes) != 1:
        return None
    return next(iter(sizes))


if __name__ == "__main__":
    retained = filter_size_candidates(
        (("same_as_input", (3, 4)), ("constant", (2, 2)))
    )
    assert retained == (("same_as_input", (3, 4)),)
    assert agreed_size(retained) == (3, 4)
    assert agreed_size((("transpose", (4, 3)), ("largest_obj_4", (4, 3)))) == (4, 3)
    assert agreed_size((("transpose", (4, 3)), ("same_as_input", (3, 4)))) is None
    print("safe_shape_family selftest: PASS")
