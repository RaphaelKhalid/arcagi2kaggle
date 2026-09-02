"""Per-token weights for demo-balanced test-time training episodes."""

from __future__ import annotations

from math import fsum


def demo_loss_weights(
    answer_lengths: tuple[int, ...],
    *,
    demo_balance: float = 1.0,
) -> tuple[tuple[float, ...], ...]:
    """Return normalized token weights for each supervised demo span.

    ``demo_balance=0`` is ordinary token balancing. ``demo_balance=1`` gives
    every demonstration equal total weight. Intermediate values linearly mix
    the two objectives while preserving a total weight of one.
    """

    if not answer_lengths or any(length <= 0 for length in answer_lengths):
        raise ValueError("answer_lengths must contain positive lengths")
    if not 0.0 <= demo_balance <= 1.0:
        raise ValueError("demo_balance must lie in [0, 1]")
    total_tokens = sum(answer_lengths)
    demos = len(answer_lengths)
    result = tuple(
        tuple(
            demo_balance / (demos * length)
            + (1.0 - demo_balance) / total_tokens
            for _ in range(length)
        )
        for length in answer_lengths
    )
    if abs(fsum(weight for span in result for weight in span) - 1.0) > 1e-12:
        raise ArithmeticError("loss weights failed to normalize")
    return result


if __name__ == "__main__":
    weights = demo_loss_weights((2, 4))
    assert abs(fsum(map(fsum, weights)) - 1.0) < 1e-12
    print("demo_loss_weights selftest: PASS")
