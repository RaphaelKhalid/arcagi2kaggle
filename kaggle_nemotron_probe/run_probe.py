"""Generate Nemotron Lightning ARC programs and measure verified pass@k."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

try:  # package import in local tests; flat import in the generated Kaggle notebook
    from .probe_core import aggregate_records, build_messages, evaluate_responses
except ImportError:  # pragma: no cover - exercised inside the Kaggle working dir
    from probe_core import aggregate_records, build_messages, evaluate_responses


DATA_DIRS = (
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
    "/kaggle/input/arc-prize-2026-arc-agi-2",
    "../data/raw",
    "data/raw",
)


def find_data_dir(override: str | None) -> Path:
    for raw in ([override] if override else []) + list(DATA_DIRS):
        if raw:
            path = Path(raw)
            if (path / "arc-agi_evaluation_challenges.json").exists():
                return path
    raise FileNotFoundError("ARC evaluation files not found")


def find_model(override: str | None) -> str:
    if override and Path(override).exists():
        return override
    roots = [Path("/kaggle/input"), Path("/kaggle/working")]
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for config in root.rglob("config.json"):
            low = str(config.parent).lower()
            if "nemotron" in low and "lightning" in low:
                matches.append(config.parent)
    if not matches:
        raise FileNotFoundError(
            "Nemotron Lightning model not attached; pass --model-path explicitly"
        )
    return str(sorted(matches, key=lambda path: len(str(path)))[0])


def compute_folds(task_ids: list[str], n_folds: int = 5) -> dict[str, int]:
    """Mirror ``arc.folds.compute_folds`` without requiring the repo package."""
    ordered = sorted(
        set(task_ids),
        key=lambda task_id: (hashlib.sha256(task_id.encode()).hexdigest(), task_id),
    )
    return {task_id: index % n_folds for index, task_id in enumerate(ordered)}


def prompt_cost(task: dict[str, Any], test_index: int) -> int:
    pairs = task.get("train", [])
    grids = [grid for pair in pairs for grid in (pair["input"], pair["output"])]
    grids.append(task["test"][test_index]["input"])
    return sum(len(grid) * len(grid[0]) for grid in grids)


def load_units(data_dir: Path, folds: set[int], max_outputs: int) -> list[dict[str, Any]]:
    challenges = json.loads(
        (data_dir / "arc-agi_evaluation_challenges.json").read_text(encoding="utf-8")
    )
    solutions = json.loads(
        (data_dir / "arc-agi_evaluation_solutions.json").read_text(encoding="utf-8")
    )
    assignment = compute_folds(list(challenges))
    units = []
    for task_id, task in challenges.items():
        fold = assignment[task_id]
        if fold not in folds:
            continue
        for test_index, item in enumerate(task["test"]):
            units.append({
                "task_id": task_id,
                "test_index": test_index,
                "fold": fold,
                "train": task["train"],
                "test_input": item["input"],
                "truth": solutions[task_id][test_index],
                "cost": prompt_cost(task, test_index),
            })
    # Deterministic, complexity-spread sample: interleave prompt-cost quartiles.
    units.sort(key=lambda unit: (unit["cost"], unit["task_id"], unit["test_index"]))
    if max_outputs and len(units) > max_outputs:
        stride = len(units) / max_outputs
        units = [units[min(int((index + 0.5) * stride), len(units) - 1)]
                 for index in range(max_outputs)]
    return units


def load_engine(model_path: str, max_model_len: int, gpu_memory: float):
    import torch
    import vllm
    from vllm import LLM

    print(f"[env] torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
    print(f"[env] vllm={vllm.__version__} devices={torch.cuda.device_count()}", flush=True)
    for device in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device)
        print(
            f"[env] gpu{device}={props.name} vram={props.total_memory / 2**30:.1f}GiB",
            flush=True,
        )
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
    print("[engine] START loading and sharding model", flush=True)
    print("[engine]", json.dumps(kwargs, default=str), flush=True)
    started = time.perf_counter()
    engine = LLM(**kwargs)
    print(f"[engine] READY boot_seconds={time.perf_counter() - started:.1f}", flush=True)
    return engine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--data-dir")
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--max-outputs", type=int, default=24)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=260901)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory", type=float, default=0.88)
    parser.add_argument("--program-timeout", type=float, default=8.0)
    parser.add_argument("--output", default="/kaggle/working/nemotron_probe.json")
    args = parser.parse_args()

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        raise RuntimeError("This public-evaluation probe must never run in a competition rerun")

    data_dir = find_data_dir(args.data_dir)
    model_path = find_model(args.model_path)
    folds = {int(item) for item in args.folds.split(",") if item.strip()}
    if 4 in folds:
        raise ValueError("fold 4 is the untouched shadow; remove it from --folds")
    units = load_units(data_dir, folds, args.max_outputs)
    print(f"[probe] model={model_path}", flush=True)
    print(
        f"[probe] data={data_dir} outputs={len(units)} k={args.k} folds={sorted(folds)}",
        flush=True,
    )

    engine = load_engine(model_path, args.max_model_len, args.gpu_memory)
    from vllm import SamplingParams

    prompts = [
        engine.get_tokenizer().apply_chat_template(
            build_messages(unit["train"], unit["test_input"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for unit in units
    ]
    sampling = SamplingParams(
        n=args.k,
        temperature=1.0,
        top_p=0.95,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    print(
        f"[generate] START {len(units)} outputs x {args.k} candidates; "
        "vLLM progress bar follows",
        flush=True,
    )
    started = time.perf_counter()
    outputs = engine.generate(prompts, sampling, use_tqdm=True)
    generation_seconds = time.perf_counter() - started
    generated_tokens = sum(len(sample.token_ids) for output in outputs for sample in output.outputs)
    print(
        f"[generate] seconds={generation_seconds:.1f} tokens={generated_tokens} "
        f"tokens_per_second={generated_tokens / max(generation_seconds, 1e-9):.1f}",
        flush=True,
    )

    from tqdm.auto import tqdm

    print("[verify] START sandbox execution and exact scoring", flush=True)
    records = []
    verify_bar = tqdm(
        zip(units, outputs),
        total=len(units),
        desc="ARC outputs verified",
        unit="output",
        dynamic_ncols=True,
    )
    for unit, output in verify_bar:
        texts = [sample.text for sample in output.outputs]
        record = evaluate_responses(
            texts,
            unit["train"],
            unit["test_input"],
            unit["truth"],
            timeout_seconds=args.program_timeout,
        )
        record.update({key: unit[key] for key in ("task_id", "test_index", "fold", "cost")})
        records.append(record)
        verify_bar.set_postfix(
            oracle=sum(bool(item["oracle_correct"]) for item in records),
            top2=sum(bool(item["top2_correct"]) for item in records),
            refresh=True,
        )
        print(
            f"[result] {unit['task_id']}#{unit['test_index']} "
            f"parsed={record['n_parsed']}/{args.k} verified={record['n_verified_programs']} "
            f"oracle={record['oracle_correct']} top2={record['top2_correct']}",
            flush=True,
        )

    summary = aggregate_records(records)
    summary.update({
        "k": args.k,
        "generation_seconds": generation_seconds,
        "generated_tokens": generated_tokens,
        "tokens_per_second": generated_tokens / max(generation_seconds, 1e-9),
        "model_path": model_path,
    })
    payload = {"summary": summary, "records": records}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("[summary]", json.dumps(summary, indent=2))
    print(f"[probe] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
