"""Finite, typed anchor-equation terms for relational effect induction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


Anchor = tuple[int, int]


@dataclass(frozen=True)
class AnchorObservation:
    """One demo's role anchors and the desired destination anchor."""

    source_role: str
    target_anchor: Anchor
    anchors: Mapping[str, Anchor]


class EffectEquation:
    name: str
    description_length: int

    def evaluate(self, observation: AnchorObservation) -> Anchor:
        raise NotImplementedError


@dataclass(frozen=True)
class ConstantOffset(EffectEquation):
    offset: Anchor
    name: str = "constant_offset"
    description_length: int = 2

    def evaluate(self, observation: AnchorObservation) -> Anchor:
        source = observation.anchors[observation.source_role]
        return source[0] + self.offset[0], source[1] + self.offset[1]


@dataclass(frozen=True)
class RelativeRoleOffset(EffectEquation):
    positive_role: str
    negative_role: str
    name: str = "source_plus_role_delta"
    description_length: int = 4

    def evaluate(self, observation: AnchorObservation) -> Anchor:
        source = observation.anchors[observation.source_role]
        positive = observation.anchors[self.positive_role]
        negative = observation.anchors[self.negative_role]
        return (
            source[0] + positive[0] - negative[0],
            source[1] + positive[1] - negative[1],
        )


@dataclass(frozen=True)
class RolePlusOffset(EffectEquation):
    reference_role: str
    offset: Anchor
    name: str = "reference_plus_offset"
    description_length: int = 3

    def evaluate(self, observation: AnchorObservation) -> Anchor:
        reference = observation.anchors[self.reference_role]
        return reference[0] + self.offset[0], reference[1] + self.offset[1]


def enumerate_equations(
    observations: tuple[AnchorObservation, ...],
    *,
    max_constant_offset: int = 30,
) -> tuple[EffectEquation, ...]:
    """Enumerate a finite candidate set grounded by the first observation."""

    if not observations:
        return ()
    first = observations[0]
    source = first.anchors[first.source_role]
    constant = (
        first.target_anchor[0] - source[0],
        first.target_anchor[1] - source[1],
    )
    if max(abs(constant[0]), abs(constant[1])) > max_constant_offset:
        return ()
    candidates: list[EffectEquation] = [ConstantOffset(constant)]
    roles = tuple(sorted(first.anchors))
    for role in roles:
        if role == first.source_role:
            continue
        reference = first.anchors[role]
        offset = (
            first.target_anchor[0] - reference[0],
            first.target_anchor[1] - reference[1],
        )
        if max(abs(offset[0]), abs(offset[1])) <= max_constant_offset:
            candidates.append(RolePlusOffset(role, offset))
    for positive in roles:
        for negative in roles:
            if positive == negative or positive == first.source_role or negative == first.source_role:
                continue
            candidates.append(RelativeRoleOffset(positive, negative))
    return tuple(sorted(set(candidates), key=lambda item: (
        item.description_length, repr(item)
    )))


def fit_equations(
    observations: tuple[AnchorObservation, ...],
    *,
    max_constant_offset: int = 30,
) -> tuple[EffectEquation, ...]:
    """Return equations that exactly reproduce every observed destination."""

    return tuple(
        equation for equation in enumerate_equations(
            observations, max_constant_offset=max_constant_offset
        )
        if all(equation.evaluate(observation) == observation.target_anchor
               for observation in observations)
    )


if __name__ == "__main__":
    observations = (
        AnchorObservation("A", (2, 2), {"A": (0, 0), "B": (1, 1)}),
        AnchorObservation("A", (4, 3), {"A": (2, 1), "B": (3, 2)}),
    )
    assert fit_equations(observations)
    print("effect_equations selftest: PASS")
