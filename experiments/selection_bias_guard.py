"""Uniform selection-bias correction for pass@2 calibration."""

from __future__ import annotations

from math import comb, log, sqrt


def two_attempt_action_count(n_classes: int) -> int:
    """Count one- or two-distinct-class actions from ``n_classes`` classes."""

    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    return n_classes + comb(n_classes, 2)


def uniform_pass2_lower_bound(
    successes: int,
    trials: int,
    n_classes: int,
    *,
    delta: float = 0.05,
) -> float:
    """Subtract a Hoeffding/union-bound penalty for searching all pairs.

    The guarantee is conditional on a fixed candidate class set and
    exchangeable calibration positions.  It is deliberately not a claim that
    ARC tasks are iid; callers must use it as a stress-test or proof gate.
    """

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes must lie between zero and positive trials")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    actions = two_attempt_action_count(n_classes)
    penalty = sqrt(log(actions / delta) / (2.0 * trials))
    return max(0.0, successes / trials - penalty)


if __name__ == "__main__":
    assert uniform_pass2_lower_bound(9, 10, 2) < 0.9
    print("selection_bias_guard selftest: PASS")
