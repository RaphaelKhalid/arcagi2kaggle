"""Finite-sample budget for Monte Carlo orbit projection."""

from __future__ import annotations

from math import ceil, log, sqrt


def projection_error_bound(
    sample_count: int,
    coordinate_count: int,
    delta: float,
) -> float:
    """Hoeffding-union bound for [0,1]-valued projected coordinates."""

    if sample_count < 1 or coordinate_count < 1:
        raise ValueError("sample_count and coordinate_count must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return sqrt(log(2.0 * coordinate_count / delta) / (2.0 * sample_count))


def required_sample_count(
    coordinate_count: int,
    epsilon: float,
    delta: float,
) -> int:
    """Smallest sample count with the union-bound error <= epsilon."""

    if coordinate_count < 1:
        raise ValueError("coordinate_count must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return ceil(log(2.0 * coordinate_count / delta) / (2.0 * epsilon ** 2))


if __name__ == "__main__":
    count = required_sample_count(100, 0.1, 0.05)
    assert projection_error_bound(count, 100, 0.05) <= 0.1
    assert projection_error_bound(count - 1, 100, 0.05) > 0.1
    print("orbit_sample_budget selftest: PASS", count)
