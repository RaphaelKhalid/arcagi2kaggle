"""Compile multiple aligned object traces into proof-gated frame programs."""

from __future__ import annotations

from itertools import islice, product
from typing import Any, Iterable, Mapping

from experiments.frame_role_executor import (
    FrameClause,
    FrameProgram,
    LocalObservation,
    _local_observations,
    execute_frame_program,
)
from experiments.graph_lgg import lgg_observations
from experiments.object_correspondence import top_k_correspondences
from experiments.object_deltas import extract_objects, normalize_grid
from experiments.trace_alignment import _full_alignments


def _ground_from_task(
    schema: tuple[Any, ...],
    reference: tuple[LocalObservation, ...],
    source_grid: Any,
    target_grid: Any,
) -> FrameProgram | None:
    if len(schema) != len(reference):
        return None
    source_objects = extract_objects(normalize_grid(source_grid))
    target_objects = extract_objects(normalize_grid(target_grid))
    clauses: list[FrameClause] = []
    for schema_item, local in zip(schema, reference):
        if schema_item.kind != local.observation.kind:
            return None
        if schema_item.kind == "identity":
            clauses.append(FrameClause("identity", schema_item.source_guard))
            continue
        if not schema_item.source_guard or local.source_index is None:
            return None
        if schema_item.kind == "delete":
            clauses.append(FrameClause("delete", schema_item.source_guard))
            continue
        if local.target_index is None or not schema_item.target_guard:
            return None
        left = source_objects[local.source_index]
        right = target_objects[local.target_index]
        if left.shape != right.shape:
            return None
        if schema_item.kind == "move":
            if left.colored_shape != right.colored_shape:
                return None
            clauses.append(FrameClause(
                "move", schema_item.source_guard,
                displacement=(right.anchor[0] - left.anchor[0],
                              right.anchor[1] - left.anchor[1]),
                source_shape=left.shape,
            ))
        elif schema_item.kind == "recolor":
            if left.anchor != right.anchor:
                return None
            clauses.append(FrameClause(
                "recolor", schema_item.source_guard,
                source_shape=left.shape,
                recolor_target=right.colored_shape,
            ))
        else:
            return None
    return FrameProgram(tuple(clauses))


def compile_aligned_frame_programs(
    task: Mapping[str, Any],
    *,
    k: int = 4,
    max_objects: int = 10,
    max_hypotheses: int = 32,
) -> tuple[FrameProgram, ...]:
    """Enumerate aligned trace hypotheses and retain exact replay programs."""

    if k <= 0 or max_hypotheses <= 0:
        raise ValueError("k and max_hypotheses must be positive")
    pairs = tuple(task.get("train", []))
    if not pairs:
        return ()
    local_options: list[tuple[tuple[LocalObservation, ...], ...]] = []
    try:
        for pair in pairs:
            correspondences = top_k_correspondences(
                pair["input"], pair["output"], k=k, max_objects=max_objects
            )
            if not correspondences:
                return ()
            local_options.append(tuple(
                _local_observations(pair["input"], pair["output"], correspondence)
                for correspondence in correspondences
            ))
    except (IndexError, ValueError):
        return ()

    programs: list[FrameProgram] = []
    seen: set[FrameProgram] = set()
    # The product is the explicit anytime boundary.  First-demo candidates are
    # not privileged: a globally useful trace may be a non-top-1 local parse.
    for choices in islice(product(*local_options), max_hypotheses * k):
        reference = choices[0]
        reference_observations = tuple(item.observation for item in reference)
        aligned_traces: list[tuple[Any, ...]] = [reference_observations]
        valid = True
        for other in choices[1:]:
            alignments = _full_alignments(
                reference_observations,
                tuple(item.observation for item in other),
                k=k,
            )
            if not alignments:
                valid = False
                break
            alignment = alignments[0]
            by_reference = {left: right for left, right in alignment.pairs}
            aligned_traces.append(tuple(
                other[by_reference[index]].observation
                for index in range(len(reference))
            ))
        if not valid:
            continue
        schema = lgg_observations(aligned_traces)
        if schema is None:
            continue
        program = _ground_from_task(
            schema, reference, pairs[0]["input"], pairs[0]["output"]
        )
        if program is None or program in seen:
            continue
        if all(
            execute_frame_program(program, pair["input"])
            == normalize_grid(pair["output"])
            for pair in pairs
        ):
            seen.add(program)
            programs.append(program)
        if len(programs) >= max_hypotheses:
            break
    return tuple(programs)
