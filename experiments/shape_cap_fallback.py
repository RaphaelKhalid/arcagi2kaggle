"""Monotone fallback policy for two independently audited shape predictors."""

from __future__ import annotations


def _validate_size(size: tuple[int, int]) -> None:
    if len(size) != 2 or not all(isinstance(value, int) for value in size):
        raise ValueError("size must be an integer (height, width) pair")
    if not all(1 <= value <= 30 for value in size):
        raise ValueError("size dimensions must lie in [1, 30]")


def primary_or_fallback(
    primary: tuple[int, int] | None,
    fallback: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Keep the primary prediction and consult fallback only on abstention."""

    if primary is not None:
        _validate_size(primary)
        return primary
    if fallback is not None:
        _validate_size(fallback)
    return fallback


def token_cap(size: tuple[int, int], slack: int = 2) -> int:
    """Return the notebook-compatible cell/newline/EOS cap with slack."""

    _validate_size(size)
    if slack < 0:
        raise ValueError("slack must be non-negative")
    height, width = size
    return height * width + height + slack


if __name__ == "__main__":
    assert primary_or_fallback((3, 4), (9, 9)) == (3, 4)
    assert primary_or_fallback(None, (9, 9)) == (9, 9)
    assert primary_or_fallback(None, None) is None
    assert token_cap((2, 3)) == 10
    print("shape_cap_fallback selftest: PASS")
