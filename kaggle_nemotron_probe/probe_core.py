"""Evaluation core for the isolated Nemotron Lightning ARC probe.

The model writes a Python ``transform(grid)`` function.  Candidate programs run
in a short-lived subprocess, are checked on every demonstration, and only
demo-verified programs are allowed to predict the held-out test grid.

This is an evaluation instrument, not a competition submission solver.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from math import exp, fsum, log
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


Grid = list[list[int]]

SYSTEM_PROMPT = """You are an expert at solving ARC-AGI (Abstraction and Reasoning Corpus) puzzles by writing Python code.
Your goal is to analyze input-output examples and create a `transform` function that correctly transforms any input grid into the corresponding output grid.

Analyze grid dimensions, colors, objects, symmetries, spatial operations, and patterns. Find the simplest single rule that works for ALL examples.

Code requirements:
- Function signature: `def transform(grid: list[list[int]]) -> list[list[int]]:`
- Input is a rectangular 2D list of integers 0-9.
- Return a rectangular 2D list of integers 0-9.
- Wrap code in ```python ... ```.
- Available imports: numpy, scipy, itertools, collections.
- The function must generalize to new valid grid sizes.
- Do not read files, use the network, start processes, or include an `if __name__` block.

Before the code, briefly explain the rule and why it fits every example."""

MINIMAL_SYSTEM_PROMPT = """Solve the ARC-AGI task from the demonstrations.
Infer a general transformation and return a Python function with signature
`def transform(grid: list[list[int]]) -> list[list[int]]:`. Explain your rule
briefly, then put the function in a ```python``` block. The input and output are
rectangular grids with integer colors 0-9. Do not use files, network access,
hidden-test data, or an `if __name__` block."""

FREEFORM_SYSTEM_PROMPT = """Work out the ARC-AGI transformation in the way you find most reliable.
Explore any relevant spatial, object, relational, color, or compositional
interpretation, and then provide a general Python `transform(grid)` function.
Return a brief explanation followed by the function in a ```python``` block.
The function receives and returns rectangular list-of-list grids with colors
0-9; do not use files, network access, hidden-test data, or an `if __name__`
block."""

PROMPT_STYLES = {
    "strict": SYSTEM_PROMPT,
    "minimal": MINIMAL_SYSTEM_PROMPT,
    "freeform": FREEFORM_SYSTEM_PROMPT,
}


@dataclass(frozen=True)
class CandidateResult:
    program_hash: str
    status: str
    pairs_passed: int = 0
    n_pairs: int = 0
    prediction: Grid | None = None
    error: str | None = None
    first_failed_demo: int | None = None
    observed: Grid | None = None
    expected: Grid | None = None


def render_grid(grid: Grid) -> str:
    """Render a grid in NVIDIA's compact prompt representation."""
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def build_user_prompt(train: list[dict[str, Grid]], test_input: Grid) -> str:
    chunks = ["Please solve this ARC-AGI problem:\n"]
    for index, pair in enumerate(train, 1):
        chunks.append(
            f"Train Example {index}:\n\nInput:\n{render_grid(pair['input'])}"
            f"\n\nOutput:\n{render_grid(pair['output'])}\n"
        )
    chunks.append(f"\nTest Input:\n{render_grid(test_input)}\n")
    return "\n".join(chunks)


def build_messages(
    train: list[dict[str, Grid]],
    test_input: Grid,
    *,
    prompt_style: str = "strict",
) -> list[dict[str, str]]:
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style: {prompt_style}")
    return [
        {"role": "system", "content": PROMPT_STYLES[prompt_style]},
        {"role": "user", "content": build_user_prompt(train, test_input)},
    ]


_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_program(text: str) -> str | None:
    """Extract the first fenced candidate containing ``def transform``."""
    for block in _FENCE_RE.findall(text or ""):
        if re.search(r"\bdef\s+transform\s*\(", block):
            return block.strip()
    match = re.search(r"\bdef\s+transform\s*\(", text or "")
    if match:
        return text[match.start():].strip()
    return None


def program_hash(source: str) -> str:
    normalized = "\n".join(line.rstrip() for line in source.strip().splitlines())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


_ALLOWED_IMPORT_ROOTS = {"numpy", "scipy", "itertools", "collections"}
_BANNED_NODES = (
    ast.AsyncFunctionDef,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
    ast.With,
)
_BANNED_NAMES = {
    "breakpoint", "compile", "eval", "exec", "exit", "getattr", "globals",
    "help", "input", "locals", "memoryview", "open", "quit", "setattr",
    "vars", "__import__",
}
_BANNED_ATTRIBUTES = {
    "ctypes", "dump", "dumps", "fromfile", "load", "loads", "memmap",
    "popen", "save", "savetxt", "system", "tofile",
}
_MAX_SOURCE_CHARS = 32_000
_MAX_AST_NODES = 6_000


def validate_program(source: str) -> str | None:
    if len(source) > _MAX_SOURCE_CHARS:
        return f"source exceeds {_MAX_SOURCE_CHARS} character budget"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        return f"program exceeds {_MAX_AST_NODES} AST-node budget"
    transforms = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "transform"
    ]
    if len(transforms) != 1:
        return "candidate must define exactly one top-level transform function"
    for node in ast.walk(tree):
        if isinstance(node, _BANNED_NODES):
            return f"banned construct: {type(node).__name__}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(module.split(".", 1)[0] not in _ALLOWED_IMPORT_ROOTS for module in modules):
                return f"banned import: {', '.join(modules)}"
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            return f"banned name: {node.id}"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in _BANNED_ATTRIBUTES:
                return f"banned attribute: {node.attr}"
    return None


def _valid_grid(value: Any) -> bool:
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        return False
    if not all(isinstance(row, list) for row in value):
        return False
    width = len(value[0]) if value else 0
    if not 1 <= width <= 30 or any(len(row) != width for row in value):
        return False
    return all(
        isinstance(cell, int) and not isinstance(cell, bool) and 0 <= cell <= 9
        for row in value for cell in row
    )


def _normalize_grid(value: Any) -> Grid:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not _valid_grid(value):
        raise ValueError("transform returned an invalid ARC grid")
    return [[int(cell) for cell in row] for row in value]


def _safe_import(
    name: str,
    globals_: dict[str, Any] | None = None,
    locals_: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    if level or name.split(".", 1)[0] not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import of {name!r} is disabled")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _safe_builtins() -> dict[str, Any]:
    names = (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "frozenset", "int", "isinstance", "len", "list", "map", "max",
        "min", "pow", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError",
    )
    result = {name: getattr(builtins, name) for name in names}
    result["__import__"] = _safe_import
    return result


def _limit_worker(cpu_seconds: int, memory_gib: int) -> None:
    """Apply Linux resource limits when available; subprocess timeout is the backstop."""
    try:
        import resource

        memory = memory_gib * 1024**3
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    except (ImportError, OSError, ValueError):
        pass


def _worker(request_path: str, response_path: str) -> int:
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    source = request["source"]
    error = validate_program(source)
    if error:
        result = {"status": "rejected", "pairs_passed": 0, "error": error}
    else:
        _limit_worker(int(request["cpu_seconds"]), int(request["memory_gib"]))
        namespace: dict[str, Any] = {"__builtins__": _safe_builtins()}
        try:
            exec(compile(source, "<arc-candidate>", "exec"), namespace)
            transform = namespace["transform"]
            passed = 0
            first_error = None
            first_failed_demo = None
            first_observed = None
            first_expected = None
            for index, pair in enumerate(request["train"]):
                try:
                    got = _normalize_grid(transform([row[:] for row in pair["input"]]))
                    if got == pair["output"]:
                        passed += 1
                    elif first_error is None:
                        first_error = f"demo {index} mismatch"
                        first_failed_demo = index
                        first_observed = got
                        first_expected = pair["output"]
                except Exception as exc:  # candidate failure, contained in worker
                    if first_error is None:
                        first_error = f"demo {index}: {type(exc).__name__}: {exc}"
                        first_failed_demo = index
                        first_expected = pair["output"]
            if passed == len(request["train"]):
                prediction = _normalize_grid(
                    transform([row[:] for row in request["test_input"]])
                )
                result = {
                    "status": "verified",
                    "pairs_passed": passed,
                    "n_pairs": len(request["train"]),
                    "prediction": prediction,
                }
            else:
                result = {
                    "status": "partial",
                    "pairs_passed": passed,
                    "n_pairs": len(request["train"]),
                    "error": first_error,
                    "first_failed_demo": first_failed_demo,
                    "observed": first_observed,
                    "expected": first_expected,
                }
        except BaseException as exc:  # worker is disposable
            result = {
                "status": "error",
                "pairs_passed": 0,
                "n_pairs": len(request["train"]),
                "error": f"{type(exc).__name__}: {exc}"[:400],
            }
    Path(response_path).write_text(json.dumps(result), encoding="utf-8")
    return 0


def run_candidate(
    source: str,
    train: list[dict[str, Grid]],
    test_input: Grid,
    timeout_seconds: float = 8.0,
    memory_gib: int = 4,
) -> CandidateResult:
    digest = program_hash(source)
    static_error = validate_program(source)
    if static_error:
        return CandidateResult(digest, "rejected", error=static_error)
    request = {
        "source": source,
        "train": train,
        "test_input": test_input,
        "cpu_seconds": max(1, int(timeout_seconds)),
        "memory_gib": memory_gib,
    }
    with tempfile.TemporaryDirectory(prefix="arc_nemotron_") as temp_dir:
        request_path = Path(temp_dir) / "request.json"
        response_path = Path(temp_dir) / "response.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "_worker",
                 str(request_path), str(response_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 3.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CandidateResult(digest, "timeout", error="subprocess timeout")
        if completed.returncode != 0 or not response_path.exists():
            detail = (completed.stderr or completed.stdout or "worker failed")[-400:]
            return CandidateResult(digest, "error", error=detail)
        raw = json.loads(response_path.read_text(encoding="utf-8"))
    return CandidateResult(
        program_hash=digest,
        status=raw["status"],
        pairs_passed=int(raw.get("pairs_passed", 0)),
        n_pairs=int(raw.get("n_pairs", len(train))),
        prediction=raw.get("prediction"),
        error=raw.get("error"),
        first_failed_demo=raw.get("first_failed_demo"),
        observed=raw.get("observed"),
        expected=raw.get("expected"),
    )


def grid_key(grid: Grid) -> str:
    return json.dumps(grid, separators=(",", ":"))


def rank_verified_outputs(
    results: Iterable[CandidateResult],
    *,
    correlation_groups: dict[str, str] | None = None,
    mdl_lengths: dict[str, float] | None = None,
    collapse_correlated: bool = False,
) -> tuple[tuple[str, float, tuple[str, ...]], ...]:
    """Rank verified output classes with optional lineage correction.

    The default keeps each program hash independent for backwards-compatible
    probe behavior.  In collapse mode, one lineage contributes one normalized
    semantic distribution; within a lineage/output class the shortest known
    program supplies the MDL weight.
    """

    verified = tuple(
        result for result in results
        if result.status == "verified" and result.prediction is not None
    )
    if not verified:
        return ()
    groups: dict[str, list[CandidateResult]] = {}
    for result in verified:
        group = (
            (correlation_groups or {}).get(result.program_hash, result.program_hash)
            if collapse_correlated else result.program_hash
        )
        groups.setdefault(group, []).append(result)
    class_mass: dict[str, float] = {}
    witness_groups: dict[str, set[str]] = {}
    group_weight = 1.0 / len(groups)
    lengths = mdl_lengths or {}
    for group, members in sorted(groups.items()):
        by_output: dict[str, list[CandidateResult]] = {}
        for member in members:
            by_output.setdefault(grid_key(member.prediction), []).append(member)
        scores = {
            output: max(exp(-lengths.get(member.program_hash, 0.0) * log(2.0))
                        for member in output_members)
            for output, output_members in by_output.items()
        }
        total = fsum(scores.values())
        for output, score in scores.items():
            class_mass[output] = class_mass.get(output, 0.0) + group_weight * score / total
            witness_groups.setdefault(output, set()).add(group)
    ranked = sorted(class_mass, key=lambda output: (-class_mass[output], output))
    return tuple(
        (output, class_mass[output], tuple(sorted(witness_groups[output])))
        for output in ranked
    )


def evaluate_responses(
    responses: Iterable[str],
    train: list[dict[str, Grid]],
    test_input: Grid,
    truth: Grid | None = None,
    timeout_seconds: float = 8.0,
    correlation_groups: dict[str, str] | None = None,
    mdl_lengths: dict[str, float] | None = None,
    collapse_correlated: bool = False,
) -> dict[str, Any]:
    responses = list(responses)
    programs: dict[str, str] = {}
    parse_count = 0
    for text in responses:
        source = extract_program(text)
        if source is not None:
            parse_count += 1
            programs.setdefault(program_hash(source), source)

    results = [
        run_candidate(source, train, test_input, timeout_seconds)
        for source in programs.values()
    ]
    verified = [result for result in results if result.status == "verified"]
    votes = Counter(grid_key(result.prediction) for result in verified if result.prediction)
    ranked_classes = rank_verified_outputs(
        results,
        correlation_groups=correlation_groups,
        mdl_lengths=mdl_lengths,
        collapse_correlated=collapse_correlated,
    )
    ranked = [json.loads(key) for key, _, _ in ranked_classes]
    correct_verified = (
        sum(1 for result in verified if result.prediction == truth)
        if truth is not None else None
    )
    return {
        "n_responses": len(responses),
        "n_parsed": parse_count,
        "n_unique_programs": len(programs),
        "n_verified_programs": len(verified),
        "status_counts": dict(Counter(result.status for result in results)),
        "n_unique_verified_outputs": len(votes),
        "top_predictions": ranked[:2],
        "top_vote_counts": [count for _, count in votes.most_common(2)],
        "top_class_masses": [mass for _, mass, _ in ranked_classes[:2]],
        "top_class_witness_groups": [groups for _, _, groups in ranked_classes[:2]],
        "oracle_correct": (
            any(result.prediction == truth for result in verified)
            if truth is not None else None
        ),
        "top1_correct": ranked[0] == truth if truth is not None and ranked else False,
        "top2_correct": (
            any(prediction == truth for prediction in ranked[:2])
            if truth is not None else None
        ),
        "correct_verified_programs": correct_verified,
        "candidates": [asdict(result) for result in results],
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    responses = sum(record["n_responses"] for record in records)
    parsed = sum(record["n_parsed"] for record in records)
    unique = sum(record["n_unique_programs"] for record in records)
    verified = sum(record["n_verified_programs"] for record in records)
    covered = sum(bool(record["n_verified_programs"]) for record in records)
    total = len(records)
    return {
        "outputs": total,
        "responses": responses,
        "parse_rate": parsed / responses if responses else 0.0,
        "unique_program_rate": unique / parsed if parsed else 0.0,
        "demo_verified_program_rate": verified / unique if unique else 0.0,
        "verified_output_coverage": covered / total if total else 0.0,
        "oracle_pass_at_k": sum(bool(r["oracle_correct"]) for r in records) / total if total else 0.0,
        "top1_accuracy": sum(bool(r["top1_correct"]) for r in records) / total if total else 0.0,
        "top2_accuracy": sum(bool(r["top2_correct"]) for r in records) / total if total else 0.0,
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["_worker", "selftest"])
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    if args.mode == "_worker":
        if len(args.paths) != 2:
            parser.error("_worker requires request and response paths")
        return _worker(args.paths[0], args.paths[1])

    train = [
        {"input": [[0, 2], [0, 0]], "output": [[0, 3], [0, 0]]},
        {"input": [[2, 0], [0, 2]], "output": [[3, 0], [0, 3]]},
    ]
    good = """```python
def transform(grid):
    return [[3 if cell == 2 else cell for cell in row] for row in grid]
```"""
    bad = """```python
import os
def transform(grid):
    return grid
```"""
    report = evaluate_responses([good, bad], train, [[2]], [[3]])
    assert report["n_parsed"] == 2
    assert report["n_verified_programs"] == 1
    assert report["top1_correct"] is True
    print(json.dumps(report, indent=2))
    print("Nemotron probe selftest: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
