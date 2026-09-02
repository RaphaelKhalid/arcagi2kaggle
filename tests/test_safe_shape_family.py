from __future__ import annotations

from experiments.safe_shape_family import (
    agreed_size,
    filter_size_candidates,
)


def test_unsafe_extrapolation_families_are_filtered() -> None:
    candidates = filter_size_candidates(
        (("same_as_input", (3, 4)), ("constant", (2, 2)), ("ratio", (6, 8)))
    )
    assert candidates == (("same_as_input", (3, 4)),)


def test_agreed_size_requires_one_distinct_size() -> None:
    assert agreed_size((("transpose", (4, 3)), ("largest_obj_4", (4, 3)))) == (4, 3)
    assert agreed_size((("transpose", (4, 3)), ("same_as_input", (3, 4)))) is None
    assert agreed_size(()) is None


def test_custom_allowlist_can_be_narrower() -> None:
    assert filter_size_candidates(
        (("same_as_input", (3, 4)), ("transpose", (4, 3))),
        allowed_rules=frozenset({"transpose"}),
    ) == (("transpose", (4, 3)),)
