"""Build Kaggle-format local data files from the public ARC-AGI-2 repository.

The competition bundle is not redistributed here.  If you cannot run the
secure Kaggle downloader, the public task files at
https://github.com/arcprize/ARC-AGI-2 (Apache 2.0, ARC Prize Foundation) carry
the same training (1000) and evaluation (120) tasks and can be converted into
the ``arc-agi_<split>_challenges.json`` / ``arc-agi_<split>_solutions.json``
layout the notebook and the local harness read.

Usage:
    git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 /tmp/ARC-AGI-2
    python scripts/make_local_data.py /tmp/ARC-AGI-2

Output: data/raw/arc-agi_{training,evaluation}_{challenges,solutions}.json
(``data/`` is gitignored).  Note: the Kaggle bundle may lag or lead the
GitHub repository slightly (test-output counts have differed by a few), so use
these files for dry runs and audits, not as the last word on task ids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw"


def convert(repo: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("training", "evaluation"):
        src = repo / "data" / split
        if not src.is_dir():
            raise SystemExit(f"missing {src}; is {repo} a clone of arcprize/ARC-AGI-2?")
        challenges, solutions = {}, {}
        for path in sorted(src.glob("*.json")):
            task = json.loads(path.read_text(encoding="utf-8"))
            tid = path.stem
            challenges[tid] = {
                "train": task["train"],
                "test": [{"input": pair["input"]} for pair in task["test"]],
            }
            solutions[tid] = [pair["output"] for pair in task["test"]]
        (out_dir / f"arc-agi_{split}_challenges.json").write_text(json.dumps(challenges), encoding="utf-8")
        (out_dir / f"arc-agi_{split}_solutions.json").write_text(json.dumps(solutions), encoding="utf-8")
        n_out = sum(len(v) for v in solutions.values())
        print(f"{split}: {len(challenges)} tasks, {n_out} test outputs -> {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", type=Path, help="path to a clone of https://github.com/arcprize/ARC-AGI-2")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    convert(args.repo, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
