"""Leakage-safe calibration of unknown test-output serialization cost."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import fsum
from typing import Iterable


def input_area_bucket(area: int) -> int:
    """Coarse input-area bucket shared by calibration and target routing."""

    if area < 1:
        raise ValueError("area must be positive")
    return 0 if area <= 25 else 1 if area <= 100 else 2 if area <= 225 else 3


def demo_count_bucket(demo_count: int) -> int:
    if demo_count < 1:
        raise ValueError("demo_count must be positive")
    return 0 if demo_count <= 2 else 1 if demo_count <= 4 else 2


@dataclass(frozen=True)
class OutputCostCalibration:
    global_mean: float
    bucket_means: dict[tuple[int, int], float]
    min_bucket_samples: int

    def estimate(
        self,
        input_area: int,
        demo_count: int,
        *,
        known_shape_cap: float | None = None,
    ) -> float:
        """Estimate extra output cost, preferring an audited shape cap."""

        if known_shape_cap is not None:
            if known_shape_cap <= 0.0:
                raise ValueError("known_shape_cap must be positive")
            return float(known_shape_cap)
        key = (input_area_bucket(input_area), demo_count_bucket(demo_count))
        return self.bucket_means.get(key, self.global_mean)


def fit_output_cost_calibration(
    samples: Iterable[tuple[int, int, float]],
    *,
    min_bucket_samples: int = 20,
) -> OutputCostCalibration:
    """Fit bucket means from labeled training outputs only.

    Each sample is `(test_input_area, demo_count, true_output_token_cost)`.
    Small buckets fall back to the pooled mean, avoiding high-variance routing
    estimates for sparse structural groups.
    """

    if min_bucket_samples < 1:
        raise ValueError("min_bucket_samples must be positive")
    rows = tuple(samples)
    if not rows:
        raise ValueError("at least one calibration sample is required")
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    costs = []
    for area, demos, cost in rows:
        key = (input_area_bucket(int(area)), demo_count_bucket(int(demos)))
        cost = float(cost)
        if cost <= 0.0:
            raise ValueError("output costs must be positive")
        grouped[key].append(cost)
        costs.append(cost)
    means = {
        key: fsum(values) / len(values)
        for key, values in grouped.items()
        if len(values) >= min_bucket_samples
    }
    return OutputCostCalibration(
        global_mean=fsum(costs) / len(costs),
        bucket_means=means,
        min_bucket_samples=min_bucket_samples,
    )


if __name__ == "__main__":
    calibration = fit_output_cost_calibration(
        [(10, 2, 5.0)] * 20 + [(200, 4, 20.0)] * 20
    )
    assert calibration.estimate(10, 2) == 5.0
    assert calibration.estimate(10, 2, known_shape_cap=7.0) == 7.0
    print("output_cost_calibration selftest: PASS")
