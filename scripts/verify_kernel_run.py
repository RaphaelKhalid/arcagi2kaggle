"""Fetch a Kaggle kernel run's output and verify it before submitting.

Usage: python scripts/verify_kernel_run.py [kernel_ref] [--challenges PATH] [--skip-fetch]
Default kernel_ref: the "id" in kaggle_notebook/kernel-metadata.json.

Checks (commit run = eval mode, 4 smoke tasks against the public evaluation file):
  1. kernel status is COMPLETE
  2. output contains submission.json
  3. submission.json passes the arc/ harness validator against the evaluation
     challenges (data/raw/arc-agi_evaluation_challenges.json by default; see
     scripts/make_local_data.py or scripts/secure_kaggle_download.py)
  4. log shows all 4 worker ranks starting, the "submission.json written" line
     and the eval-mode "Reload score:" line, and no Traceback

Exit code 0 = safe to submit; 1 = do not submit.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "artifacts" / "kernel_run"
DEFAULT_CHALLENGES = ROOT / "data" / "raw" / "arc-agi_evaluation_challenges.json"


def default_kernel() -> str:
    meta = json.loads((ROOT / "kaggle_notebook" / "kernel-metadata.json").read_text(encoding="utf-8"))
    return meta["id"]


def kaggle(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "kaggle_cli.py"), *args],
        capture_output=True, text=True,
    )
    return result.stdout + result.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernel", nargs="?", default=None)
    parser.add_argument("--challenges", type=Path, default=DEFAULT_CHALLENGES)
    parser.add_argument("--skip-fetch", action="store_true", help="verify an already downloaded artifacts/kernel_run")
    args = parser.parse_args(argv)
    kernel = args.kernel or default_kernel()

    failures: list[str] = []

    if not args.skip_fetch:
        status = kaggle("kernels", "status", kernel)
        print(f"status: {status.strip()}")
        if "complete" not in status.lower():
            print("NOT COMPLETE - do not submit yet.")
            return 1
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", kernel, "-p", str(OUT_DIR)))

    sub_path = OUT_DIR / "submission.json"
    if not sub_path.exists():
        failures.append("submission.json missing from kernel output")
    elif not args.challenges.exists():
        print(f"WARNING: {args.challenges} not found; schema not validated (see scripts/make_local_data.py)")
    else:
        from arc.tasks import load_challenges
        from arc.validate import validate_submission

        challenges = load_challenges(args.challenges)
        submission = json.loads(sub_path.read_text(encoding="utf-8"))
        issues = validate_submission(submission, challenges)
        if issues:
            failures.append(f"validator found {len(issues)} issues; first: {issues[0]}")
        else:
            print(f"submission.json: schema VALID ({len(submission)} tasks)")

    logs = list(OUT_DIR.glob("*.log")) + list(OUT_DIR.glob("*.txt"))
    log_text = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in logs)
    if log_text:
        ranks = sorted(set(re.findall(r"\[Rank (\d)\] start!", log_text)))
        if ranks != ["0", "1", "2", "3"]:
            failures.append(f"worker ranks started: {ranks} (expected all 4)")
        else:
            print("all 4 GPU workers started")
        if "submission.json written" not in log_text:
            failures.append("no 'submission.json written' line - selection cell did not complete")
        if "Reload score:" not in log_text:
            failures.append("no 'Reload score:' line - final cell did not complete")
        else:
            print("reload-score line present (final cell completed)")
        if "Traceback" in log_text:
            failures.append("a Traceback appears in the log")
        finished = re.findall(r"\[Rank \d\] finished (\S+) in", log_text)
        if not finished:
            failures.append("no task finished in the smoke run (no '[Rank n] finished' line)")
        else:
            print(f"tasks finished in the smoke run: {sorted(set(finished))}")
        for line in log_text.splitlines():
            if re.search(r"Reload score:|size caps|WARNING|Traceback|submission.json written|finished .* in", line):
                print("  log:", line.strip()[:160])
    else:
        print("WARNING: no log file found in output; check the notebook page manually")

    if failures:
        print("\nDO NOT SUBMIT:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nALL CHECKS PASSED - safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
