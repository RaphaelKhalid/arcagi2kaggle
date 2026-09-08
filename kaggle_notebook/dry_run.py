"""CPU-only dry run of the notebook's selection + submission cell.

The GPU pipeline cannot run locally, but everything after it can: this script
fabricates per-view candidate files in the exact format the workers write
(bz2 pickles under ``/kaggle/inference_outputs``), executes the *actual*
selection cell emitted by ``build_notebook.py`` against them (with the Kaggle
paths redirected), and validates and scores the resulting ``submission.json``
with the ``arc`` harness.

Usage (needs local data, see scripts/make_local_data.py):
    python kaggle_notebook/dry_run.py --challenges data/raw/arc-agi_evaluation_challenges.json \
        --solutions data/raw/arc-agi_evaluation_solutions.json --tasks 24

Scenario per fabricated task (deterministic from --seed):
  * solved     - the ground truth appears in several views with good scores plus a decoy;
  * decoy-only - only wrong candidates exist (the task must score 0 but stay schema-valid);
  * unfinished - no candidate files at all (placeholders [[0]] must be written);
  * corrupt    - one truncated candidate file (must not crash the loader).
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import pickle
import random
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_notebook" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

VIEWS = [
    "", ".transpose", ".rot90", ".rot90.rot90", ".rot90.rot90.rot90",
    ".transpose.rot90", ".transpose.rot90.rot90", ".transpose.rot90.rot90.rot90",
]


def install_stubs() -> None:
    """arc_loader imports transformers at module level (type hints only)."""
    if "transformers" not in sys.modules:
        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = object
        sys.modules["transformers"] = stub


def decoy(grid: list[list[int]], rng: random.Random) -> list[list[int]]:
    g = [row[:] for row in grid]
    r, c = rng.randrange(len(g)), rng.randrange(len(g[0]))
    g[r][c] = (g[r][c] + 1 + rng.randrange(9)) % 10
    return g


def fabricate(challenges: dict, solutions: dict, out_dir: Path, task_ids: list[str], seed: int) -> dict[str, str]:
    """Write fake candidate files; return {task_id: scenario}."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = {}
    kinds = ["solved", "solved", "decoy", "unfinished", "corrupt", "solved"]
    for n, tid in enumerate(task_ids):
        kind = kinds[n % len(kinds)]
        scenarios[tid] = kind
        if kind == "unfinished":
            continue
        for i, truth in enumerate(solutions[tid]):
            base = f"{tid}_{i}"
            wrong = decoy(truth, rng)
            for v, view in enumerate(VIEWS):
                perm = "".join(map(str, rng.sample(range(10), 10)))
                subkey = f"{base}{view}.permute{perm}.ex{''.join(str(k) for k in range(len(challenges[tid]['train'])))}"
                records = []
                if kind == "solved" and v % 2 == 0:
                    records.append({"beam_score": 0.3 + 0.1 * rng.random(),
                                    "score_aug": [0.2 + 0.2 * rng.random() for _ in range(8)],
                                    "solution": np.array(truth)})
                if kind in ("solved", "decoy", "corrupt") and v % 3 == 0:
                    records.append({"beam_score": 1.2 + 0.3 * rng.random(),
                                    "score_aug": [1.0 + 0.5 * rng.random() for _ in range(8)],
                                    "solution": np.array(wrong)})
                if not records:
                    continue
                path = out_dir / subkey
                with bz2.BZ2File(path, "w") as f:
                    pickle.dump(records, f)
                if kind == "corrupt" and v == 0:
                    data = path.read_bytes()
                    path.write_bytes(data[: len(data) // 2])
    return scenarios


def selection_cell_source(rerun: bool) -> str:
    sys.path.insert(0, str(ROOT / "kaggle_notebook"))
    import build_notebook  # noqa: E402

    return build_notebook.selection_cell(build_notebook.PROFILE)


def run_cell(source: str, challenges_path: Path, solutions_path: Path | None, outputs_dir: Path, workdir: Path,
             rerun: bool) -> dict:
    """Execute the notebook cell with its hardcoded Kaggle paths redirected."""
    src = source
    src = src.replace("/kaggle/inference_outputs", str(outputs_dir))
    base = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/"
    src = src.replace(base + "arc-agi_test_challenges.json", str(challenges_path))
    src = src.replace(base + "arc-agi_evaluation_challenges.json", str(challenges_path))
    src = src.replace(base + "arc-agi_evaluation_solutions.json", str(solutions_path))
    assert "/kaggle/" not in src, "unredirected Kaggle path in cell"
    env_before = os.environ.get("KAGGLE_IS_COMPETITION_RERUN")
    if rerun:
        os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"
    else:
        os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        namespace: dict = {"__name__": "__notebook__"}
        exec(compile(src, "<selection-cell>", "exec"), namespace)
    finally:
        os.chdir(cwd)
        if env_before is None:
            os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
        else:
            os.environ["KAGGLE_IS_COMPETITION_RERUN"] = env_before
    return json.loads((workdir / "submission.json").read_text(encoding="utf-8"))


def dry_run(challenges: dict, solutions: dict, n_tasks: int, seed: int, rerun: bool = False,
            keep: Path | None = None) -> dict:
    """Run the whole scenario; return a report dict. Raises AssertionError on any contract violation."""
    from arc.score import score_submission
    from arc.validate import validate_submission

    install_stubs()
    task_ids = sorted(challenges)[:n_tasks] if n_tasks else sorted(challenges)
    tmp = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="arc-dry-run-"))
    outputs = tmp / "inference_outputs"
    work = tmp / "working"
    work.mkdir(parents=True, exist_ok=True)
    subset = {tid: challenges[tid] for tid in task_ids}
    sub_solutions = {tid: solutions[tid] for tid in task_ids}
    ch_path, sol_path = tmp / "challenges.json", tmp / "solutions.json"
    ch_path.write_text(json.dumps(subset), encoding="utf-8")
    sol_path.write_text(json.dumps(sub_solutions), encoding="utf-8")

    scenarios = fabricate(subset, sub_solutions, outputs, task_ids, seed)
    submission = run_cell(selection_cell_source(rerun), ch_path, sol_path, outputs, work, rerun)

    issues = validate_submission(submission, subset)
    assert not issues, f"submission schema issues: {issues[:5]}"
    report = score_submission(submission, subset, sub_solutions)
    by_kind: dict[str, list[str]] = {}
    for tid, kind in scenarios.items():
        by_kind.setdefault(kind, []).append(tid)
    # Contracts: every 'solved' task must be recovered; nothing else may score.
    for tid in by_kind.get("solved", []):
        assert report.per_task[tid].solved, f"{tid}: truth was present in candidates but not selected"
    for kind in ("decoy", "unfinished", "corrupt"):
        for tid in by_kind.get(kind, []):
            assert report.per_task[tid].correct == 0, f"{tid} ({kind}) scored without a correct candidate"
    for tid in by_kind.get("unfinished", []):
        for entry in submission[tid]:
            assert entry == {"attempt_1": [[0]], "attempt_2": [[0]]}, f"{tid}: expected placeholders"
    # attempt_2 must differ from attempt_1 whenever a second candidate existed.
    for tid in by_kind.get("solved", []):
        for entry in submission[tid]:
            assert entry["attempt_1"] != entry["attempt_2"], f"{tid}: attempt_2 duplicates attempt_1"
    return {
        "tmp": str(tmp),
        "tasks": len(task_ids),
        "scenarios": {k: len(v) for k, v in by_kind.items()},
        "score": report.score,
        "correct_outputs": report.correct_outputs,
        "total_outputs": report.total_outputs,
        "rerun_mode": rerun,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--challenges", required=True, type=Path)
    parser.add_argument("--solutions", required=True, type=Path)
    parser.add_argument("--tasks", type=int, default=24, help="number of tasks to fabricate (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", type=Path, default=None, help="keep the working directory here")
    args = parser.parse_args(argv)

    challenges = json.loads(args.challenges.read_text(encoding="utf-8"))
    solutions = json.loads(args.solutions.read_text(encoding="utf-8"))
    for rerun in (False, True):
        report = dry_run(challenges, solutions, args.tasks, args.seed, rerun=rerun, keep=args.keep)
        print(json.dumps(report, indent=2))
    print("dry run OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
