"""Fetch a Kaggle kernel run's output and verify it before submitting.

Usage: python scripts/verify_kernel_run.py [kernel_ref]
Default kernel_ref: raphaelkhalid0/arc2026-baseline-fork-v1

Checks (eval-mode smoke run):
  1. kernel status is COMPLETE
  2. output contains submission.json
  3. submission.json passes the arc/ harness validator against the local
     evaluation challenges (the file the notebook reads in eval mode)
  4. log shows all 4 worker ranks starting and the eval-mode reload score line

Exit code 0 = safe to submit; 1 = do not submit.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".tools"))

KERNEL = sys.argv[1] if len(sys.argv) > 1 else "raphaelkhalid0/arc2026-baseline-fork-v1"
OUT_DIR = ROOT / "artifacts" / "kernel_run"


def kaggle(*args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "kaggle_cli.py"), *args],
        capture_output=True, text=True,
    )
    return result.stdout + result.stderr


def main() -> int:
    failures: list[str] = []

    status = kaggle("kernels", "status", KERNEL)
    print(f"status: {status.strip()}")
    if "complete" not in status.lower():
        print("NOT COMPLETE — do not submit yet.")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", KERNEL, "-p", str(OUT_DIR)))

    sub_path = OUT_DIR / "submission.json"
    if not sub_path.exists():
        failures.append("submission.json missing from kernel output")
    else:
        from arc.tasks import load_challenges
        from arc.validate import validate_submission

        challenges = load_challenges(ROOT / "data" / "raw" / "arc-agi_evaluation_challenges.json")
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
        if "Reload score:" not in log_text:
            failures.append("no 'Reload score:' line — final cell did not complete")
        else:
            print("reload-score line present (final cell completed)")
        for line in log_text.splitlines():
            if re.search(r"Reload score:|size caps|LEGC|Traceback", line):
                print("  log:", line.strip()[:160])
    else:
        print("WARNING: no log file found in output; check the notebook page manually")

    if failures:
        print("\nDO NOT SUBMIT:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nALL CHECKS PASSED — safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
