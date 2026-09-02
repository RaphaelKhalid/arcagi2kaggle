"""CPU reference for Reynolds averaging of aligned soft predictions."""

from __future__ import annotations

from math import fsum, isfinite
from typing import Iterable, Mapping, Sequence


def _validate(
    view_vectors: Mapping[str, Sequence[float]],
    permutations: Mapping[str, Sequence[int]],
) -> tuple[tuple[str, ...], int]:
    if not view_vectors or set(view_vectors) != set(permutations):
        raise ValueError("every view needs an alignment permutation")
    names = tuple(sorted(view_vectors))
    lengths = {len(view_vectors[name]) for name in names}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("all view vectors must have the same non-empty length")
    size = next(iter(lengths))
    for name in names:
        permutation = tuple(permutations[name])
        if sorted(permutation) != list(range(size)):
            raise ValueError("alignments must be permutations of vector indices")
        if any(not isfinite(float(value)) for value in view_vectors[name]):
            raise ValueError("view vectors must contain finite values")
    return names, size


def reynolds_projection(
    view_vectors: Mapping[str, Sequence[float]],
    permutations: Mapping[str, Sequence[int]],
) -> tuple[float, ...]:
    """Average inverse-aligned view vectors in canonical coordinates."""

    names, size = _validate(view_vectors, permutations)
    return tuple(
        fsum(
            float(view_vectors[name][permutations[name][coordinate]])
            for name in names
        ) / len(names)
        for coordinate in range(size)
    )


def orbit_residual(
    view_vectors: Mapping[str, Sequence[float]],
    permutations: Mapping[str, Sequence[int]],
) -> float:
    """Return the largest aligned deviation from the Reynolds projection."""

    names, size = _validate(view_vectors, permutations)
    projection = reynolds_projection(view_vectors, permutations)
    return max(
        abs(float(view_vectors[name][permutations[name][coordinate]])
            - projection[coordinate])
        for name in names
        for coordinate in range(size)
    )


if __name__ == "__main__":
    result = reynolds_projection(
        {"identity": (1.0, 3.0), "swap": (3.0, 1.0)},
        {"identity": (0, 1), "swap": (1, 0)},
    )
    assert result == (1.0, 3.0)
    assert orbit_residual(
        {"identity": (1.0, 3.0), "swap": (3.0, 1.0)},
        {"identity": (0, 1), "swap": (1, 0)},
    ) == 0.0
    print("reynolds_projection selftest: PASS")
