"""Nemotron Leg C: verified program induction over the real competition queue.

Same contract as arc_induction_v2.py's output (Leg C merge cell in the base
notebook consumes this unchanged): writes /kaggle/working/induction_results.json
as {task_id: {"verified": bool, "outputs": [{"attempt": grid|None, "alt": grid|None}, ...]}}.

Engine and verification are the tested probe_core.py functions (prompt format,
AST-sandboxed demo-only verification, majority vote); this script points them at
the real test file (or the smoke subset) instead of a diagnostic dev sample, and
generates in cost-ascending, deadline-checked chunks so a budget overrun degrades
to partial coverage instead of a hard failure.  The default is one sampling
batch and legacy raw-majority behavior; ``--lineages`` and ``--lineage-aware``
are explicit shadow-fold controls for correlation-aware sampling.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:  # package import in local tests; flat import in the generated Kaggle notebook
    from .probe_core import (
        CandidateResult,
        build_messages,
        extract_program,
        grid_key,
        program_hash,
        rank_verified_outputs,
        run_candidate,
    )
except ImportError:  # pragma: no cover - exercised inside the Kaggle working dir
    from probe_core import (
        CandidateResult,
        build_messages,
        extract_program,
        grid_key,
        program_hash,
        rank_verified_outputs,
        run_candidate,
    )


@dataclass(frozen=True)
class SamplingLineage:
    """One independently seeded decoding group in a fixed sample budget."""

    name: str
    sample_count: int
    seed: int
    temperature: float
    prompt_style: str = "strict"


def make_sampling_lineages(
    total_samples: int,
    lineage_count: int,
    *,
    seed: int = 260901,
    temperatures: tuple[float, ...] = (1.0,),
    prompt_styles: tuple[str, ...] = ("strict",),
) -> tuple[SamplingLineage, ...]:
    """Return a balanced, deterministic lineage plan.

    ``lineage_count=1`` and the default temperature reproduce the historical
    one-batch contract.  Extra lineages split the same total sample count;
    callers may provide a frozen temperature schedule for a shadow-fold
    ablation.  The runner never infers independence from the names: the
    resulting IDs are carried explicitly to the verifier.
    """

    if total_samples < 1 or lineage_count < 1:
        raise ValueError("sample and lineage counts must be positive")
    if lineage_count > total_samples:
        raise ValueError("lineage_count cannot exceed total_samples")
    if not temperatures or any(temperature <= 0.0 for temperature in temperatures):
        raise ValueError("temperatures must be non-empty and positive")
    if not prompt_styles or any(style not in {"strict", "minimal", "freeform"}
                                for style in prompt_styles):
        raise ValueError("prompt_styles must be non-empty known styles")
    quotient, remainder = divmod(total_samples, lineage_count)
    return tuple(
        SamplingLineage(
            name=f"lineage-{index}",
            sample_count=quotient + (1 if index < remainder else 0),
            seed=seed + 1009 * index,
            temperature=temperatures[index % len(temperatures)],
            prompt_style=prompt_styles[index % len(prompt_styles)],
        )
        for index in range(lineage_count)
    )


def find_test_path() -> Path:
    root = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
    name = "arc-agi_test_challenges.json" if os.getenv("KAGGLE_IS_COMPETITION_RERUN") \
        else "arc-agi_evaluation_challenges.json"
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"competition challenges file missing: {path}")
    return path


def find_model(override: str | None) -> str:
    if override and Path(override).exists():
        return override
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        for config in root.rglob("config.json"):
            low = str(config.parent).lower()
            if "nemotron" in low and "lightning" in low:
                return str(config.parent)
    raise FileNotFoundError("Nemotron Lightning model not attached; pass --model-path")


def prompt_cost(unit: dict[str, Any]) -> int:
    grids = [grid for pair in unit["train"] for grid in (pair["input"], pair["output"])]
    grids.append(unit["test_input"])
    return sum(len(grid) * len(grid[0]) for grid in grids)


def load_units(challenges: dict[str, Any]) -> list[dict[str, Any]]:
    units = []
    for task_id, task in challenges.items():
        for test_index, item in enumerate(task["test"]):
            units.append({
                "task_id": task_id,
                "test_index": test_index,
                "train": task["train"],
                "test_input": item["input"],
            })
    units.sort(key=lambda unit: (prompt_cost(unit), unit["task_id"], unit["test_index"]))
    return units


def load_engine(model_path: str, max_model_len: int, gpu_memory: float):
    import torch
    from vllm import LLM

    print(f"[induction/env] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"[induction/env] devices={torch.cuda.device_count()}", flush=True)
    kwargs = dict(
        model=model_path,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        quantization="modelopt_fp4",
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    print("[induction/engine] START", json.dumps(kwargs, default=str), flush=True)
    started = time.perf_counter()
    engine = LLM(**kwargs)
    print(f"[induction/engine] READY boot_seconds={time.perf_counter() - started:.1f}", flush=True)
    return engine


def verify_unit(
    unit: dict[str, Any],
    texts: list[str],
    timeout_seconds: float,
    *,
    lineage_texts: tuple[tuple[str, list[str]], ...] | None = None,
    collapse_correlated: bool = False,
    include_diagnostics: bool = False,
    diagnostic_limit: int = 8,
    diagnostic_trace_chars: int = 6_000,
) -> dict[str, Any]:
    """Verify one unit and rank outputs, optionally by decoding lineage.

    The legacy ``texts`` path remains raw-majority compatible.  The opt-in
    lineage path keeps one normalized output distribution per lineage and then
    ranks semantic output classes through ``probe_core``.
    """

    if diagnostic_limit < 0:
        raise ValueError("diagnostic_limit must be non-negative")
    if diagnostic_trace_chars < 0:
        raise ValueError("diagnostic_trace_chars must be non-negative")
    diagnostics: list[dict[str, Any]] = []

    def finish(value: dict[str, Any]) -> dict[str, Any]:
        if include_diagnostics:
            value["diagnostics"] = diagnostics[:diagnostic_limit]
        return value

    def record(
        lineage: str,
        source: str,
        raw_text: str,
        result: CandidateResult,
    ) -> None:
        if not include_diagnostics or result.status == "verified":
            return
        if len(diagnostics) >= diagnostic_limit:
            return
        diagnostics.append({
            "lineage": lineage,
            "source": source,
            "result": asdict(result),
            "trace": raw_text[:diagnostic_trace_chars],
        })

    if lineage_texts is None:
        lineage_texts = (("lineage-0", texts),)
    if not lineage_texts:
        return finish({"attempt": None, "alt": None})

    if not collapse_correlated:
        flattened = [text for _, group_texts in lineage_texts for text in group_texts]
        programs: dict[str, tuple[str, str]] = {}
        for text in flattened:
            source = extract_program(text)
            if source is not None:
                programs.setdefault(program_hash(source), (source, text))
        predictions = []
        for source, raw_text in programs.values():
            result = run_candidate(source, unit["train"], unit["test_input"], timeout_seconds)
            record("lineage-0", source, raw_text, result)
            if result.status == "verified" and result.prediction is not None:
                predictions.append(result.prediction)
        if not predictions:
            return finish({"attempt": None, "alt": None})
        votes = Counter(grid_key(g) for g in predictions)
        ranked = [json.loads(key) for key, _ in votes.most_common()]
        return finish({"attempt": ranked[0], "alt": ranked[1] if len(ranked) > 1 else None})

    verified: list[CandidateResult] = []
    correlation_groups: dict[str, str] = {}
    for lineage, group_texts in lineage_texts:
        programs: dict[str, tuple[str, str]] = {}
        for text in group_texts:
            source = extract_program(text)
            if source is not None:
                programs.setdefault(program_hash(source), (source, text))
        for source, raw_text in programs.values():
            result = run_candidate(source, unit["train"], unit["test_input"], timeout_seconds)
            record(lineage, source, raw_text, result)
            if result.status == "verified" and result.prediction is not None:
                unique_id = f"{result.program_hash}@{lineage}"
                verified.append(CandidateResult(
                    program_hash=unique_id,
                    status=result.status,
                    pairs_passed=result.pairs_passed,
                    n_pairs=result.n_pairs,
                    prediction=result.prediction,
                    error=result.error,
                ))
                correlation_groups[unique_id] = lineage

    ranked = rank_verified_outputs(
        verified,
        correlation_groups=correlation_groups,
        collapse_correlated=True,
    )
    if not ranked:
        return finish({"attempt": None, "alt": None})
    outputs = [json.loads(key) for key, _, _ in ranked]
    return finish({"attempt": outputs[0], "alt": outputs[1] if len(outputs) > 1 else None})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=260901)
    parser.add_argument("--lineages", type=int, default=1,
                        help="balanced independent decoding groups (default: 1)")
    parser.add_argument("--lineage-temperatures", default="1.0",
                        help="comma-separated frozen temperatures for lineages")
    parser.add_argument("--prompt-styles", default="strict",
                        help="comma-separated prompt styles for lineages")
    parser.add_argument("--lineage-aware", action="store_true",
                        help="collapse verified outputs by decoding lineage")
    parser.add_argument("--diagnostics", action="store_true",
                        help="persist bounded failed-candidate repair telemetry")
    parser.add_argument("--diagnostic-limit", type=int, default=8,
                        help="maximum failed candidates recorded per output")
    parser.add_argument("--diagnostic-trace-chars", type=int, default=6000,
                        help="maximum raw trace characters recorded per candidate")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory", type=float, default=0.88)
    parser.add_argument("--program-timeout", type=float, default=8.0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--budget-h", type=float, default=1.0)
    parser.add_argument("--end-ts", type=float, default=0.0)
    parser.add_argument("--tasks", default="", help="comma-separated task-id allowlist (smoke mode)")
    parser.add_argument("--output", default="/kaggle/working/induction_results.json")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        from probe_core import _main as probe_selftest  # noqa: F401
        import subprocess
        import sys
        return subprocess.run([sys.executable, "probe_core.py", "selftest"], check=False).returncode

    deadline = args.end_ts if args.end_ts else (time.time() + args.budget_h * 3600)

    challenges = json.loads(find_test_path().read_text(encoding="utf-8"))
    if args.tasks:
        allow = {t for t in args.tasks.split(",") if t}
        challenges = {k: v for k, v in challenges.items() if k in allow}
    units = load_units(challenges)
    print(f"[induction] tasks={len(challenges)} outputs={len(units)} k={args.k} "
          f"deadline_in_min={max(0.0, deadline - time.time()) / 60:.1f}", flush=True)

    results: dict[str, dict[str, Any]] = {}
    for task_id, task in challenges.items():
        results[task_id] = {"verified": False, "outputs": [{"attempt": None, "alt": None}
                                                             for _ in task["test"]]}

    if time.time() > deadline:
        print("[induction] no budget remaining before engine boot; writing empty results", flush=True)
        Path(args.output).write_text(json.dumps(results), encoding="utf-8")
        return 0

    model_path = find_model(args.model_path)
    engine = load_engine(model_path, args.max_model_len, args.gpu_memory)
    from vllm import SamplingParams

    tokenizer = engine.get_tokenizer()
    temperatures = tuple(float(value) for value in args.lineage_temperatures.split(",") if value.strip())
    prompt_styles = tuple(value.strip() for value in args.prompt_styles.split(",") if value.strip())
    lineages = make_sampling_lineages(
        args.k,
        args.lineages,
        seed=args.seed,
        temperatures=temperatures,
        prompt_styles=prompt_styles,
    )
    print("[induction] lineages:", json.dumps([lineage.__dict__ for lineage in lineages]), flush=True)

    processed = verified_count = 0
    for start in range(0, len(units), args.chunk_size):
        if time.time() > deadline:
            print(f"[induction] budget exhausted after {processed}/{len(units)} outputs", flush=True)
            break
        chunk = units[start:start + args.chunk_size]
        lineage_outputs: list[list[tuple[str, list[str]]]] = [[] for _ in chunk]
        started = time.perf_counter()
        for lineage in lineages:
            prompts = [
                tokenizer.apply_chat_template(
                    build_messages(u["train"], u["test_input"],
                                   prompt_style=lineage.prompt_style),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for u in chunk
            ]
            sampling = SamplingParams(
                n=lineage.sample_count,
                temperature=lineage.temperature,
                top_p=0.95,
                max_tokens=args.max_new_tokens,
                seed=lineage.seed,
            )
            outputs = engine.generate(prompts, sampling, use_tqdm=False)
            for index, output in enumerate(outputs):
                lineage_outputs[index].append(
                    (lineage.name, [sample.text for sample in output.outputs])
                )
        elapsed = time.perf_counter() - started
        print(f"[induction] chunk {start}-{start + len(chunk)} generated in {elapsed:.1f}s", flush=True)

        for unit, groups in zip(chunk, lineage_outputs):
            entry = verify_unit(
                unit,
                [text for _, texts in groups for text in texts],
                args.program_timeout,
                lineage_texts=tuple(groups),
                collapse_correlated=args.lineage_aware,
                include_diagnostics=args.diagnostics,
                diagnostic_limit=args.diagnostic_limit,
                diagnostic_trace_chars=args.diagnostic_trace_chars,
            )
            rec = results[unit["task_id"]]
            rec["outputs"][unit["test_index"]] = entry
            if entry["attempt"] is not None:
                rec["verified"] = True
                verified_count += 1
            processed += 1
            print(f"[induction/result] {unit['task_id']}#{unit['test_index']} "
                  f"verified={entry['attempt'] is not None}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results), encoding="utf-8")
    n_tasks_verified = sum(1 for r in results.values() if r["verified"])
    print(f"[induction] wrote {args.output}: {n_tasks_verified}/{len(results)} tasks, "
          f"{verified_count}/{processed} outputs verified ({processed}/{len(units)} attempted)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
