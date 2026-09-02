"""Label-free behavioral quotienting for verified ARC program hypotheses."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

try:
    from experiments.cegis_version_space import freeze_output
except ModuleNotFoundError:  # direct ``python experiments/behavioral_partition.py``
    from cegis_version_space import freeze_output


Pair = tuple[str, str]


def _key(value: Any) -> Any:
    frozen = freeze_output(value)
    try:
        hash(frozen)
    except TypeError:
        return repr(frozen)
    return frozen


def _validate(probe_outputs: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    if not probe_outputs:
        raise ValueError("at least one probe is required")
    program_sets = {frozenset(outputs) for outputs in probe_outputs.values()}
    if len(program_sets) != 1:
        raise ValueError("every probe must evaluate every program")
    programs = tuple(sorted(next(iter(program_sets))))
    if not programs:
        raise ValueError("at least one program is required")
    return programs


def separated_pairs(
    probe_outputs: Mapping[str, Mapping[str, Any]],
    selected_probes: tuple[str, ...] | list[str] | None = None,
) -> frozenset[Pair]:
    """Return program pairs separated by at least one selected probe."""

    programs = _validate(probe_outputs)
    probes = tuple(probe_outputs) if selected_probes is None else tuple(selected_probes)
    if any(probe not in probe_outputs for probe in probes):
        raise ValueError("selected probe is unknown")
    separated: set[Pair] = set()
    for probe in probes:
        values = probe_outputs[probe]
        for left, right in combinations(programs, 2):
            if _key(values[left]) != _key(values[right]):
                separated.add((left, right))
    return frozenset(separated)


@dataclass(frozen=True)
class BehavioralClass:
    signature: tuple[Any, ...]
    members: tuple[str, ...]


def behavioral_partition(
    probe_outputs: Mapping[str, Mapping[str, Any]],
    selected_probes: tuple[str, ...] | list[str] | None = None,
) -> tuple[BehavioralClass, ...]:
    """Partition programs by their exact outputs on the selected probes."""

    programs = _validate(probe_outputs)
    probes = tuple(probe_outputs) if selected_probes is None else tuple(selected_probes)
    if any(probe not in probe_outputs for probe in probes):
        raise ValueError("selected probe is unknown")
    buckets: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for program in programs:
        signature = tuple(_key(probe_outputs[probe][program]) for probe in probes)
        buckets[signature].append(program)
    return tuple(
        BehavioralClass(signature, tuple(members))
        for signature, members in sorted(buckets.items(), key=lambda item: repr(item[0]))
    )


def greedy_probe_selection(
    probe_outputs: Mapping[str, Mapping[str, Any]],
    max_probes: int,
) -> tuple[str, ...]:
    """Greedily maximize newly separated program pairs.

    Pair separation is a coverage function, hence monotone submodular.  The
    greedy order has the standard ``1 - 1/e`` approximation guarantee for a
    fixed-size probe budget when the goal is maximum separated-pair coverage.
    """

    _validate(probe_outputs)
    if max_probes < 0:
        raise ValueError("max_probes must be non-negative")
    remaining = set(probe_outputs)
    selected: list[str] = []
    covered: frozenset[Pair] = frozenset()
    for _ in range(min(max_probes, len(remaining))):
        best: tuple[int, str, frozenset[Pair]] | None = None
        for probe in sorted(remaining):
            next_covered = separated_pairs(probe_outputs, selected + [probe])
            gain = len(next_covered - covered)
            candidate = (gain, probe, next_covered)
            if (best is None or gain > best[0] or
                    (gain == best[0] and probe < best[1])):
                best = candidate
        assert best is not None
        if best[0] == 0:
            break
        _, probe, covered = best
        selected.append(probe)
        remaining.remove(probe)
    return tuple(selected)


if __name__ == "__main__":
    probes = {
        "p1": {"a": 0, "b": 0, "c": 0},
        "p2": {"a": 0, "b": 1, "c": 1},
    }
    assert greedy_probe_selection(probes, 1) == ("p2",)
    assert len(behavioral_partition(probes)) == 2
    print("behavioral_partition selftest: PASS")
