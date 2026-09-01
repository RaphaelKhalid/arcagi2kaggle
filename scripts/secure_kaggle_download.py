"""Authenticate in memory, verify ARC-AGI-2 access, and download official data.

The API token is read with terminal echo disabled, passed to Kaggle only through
the child-process environment, and removed before this process exits.
"""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


COMPETITION = "arc-prize-2026-arc-agi-2"
ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = ROOT / "data" / "raw"


def kaggle_command(*args: str, env: dict[str, str]) -> None:
    command = [
        sys.executable,
        "-c",
        "from kaggle.cli import main; main()",
        *args,
    ]
    completed = subprocess.run(command, env=env, cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
        bundle.extractall(destination)


def validate_json_files(directory: Path) -> None:
    files = sorted(directory.rglob("*.json"))
    if not files:
        raise RuntimeError("Download completed but contained no JSON files.")
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            json.load(handle)
    print(f"Validated {len(files)} JSON files in {directory}")


def main() -> None:
    print("The token will not be displayed or written to disk.")
    token = getpass.getpass("Kaggle API token (hidden): ").strip()
    if not token:
        raise SystemExit("No token supplied.")

    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    token = ""
    try:
        print("\nChecking competition access...")
        kaggle_command("competitions", "files", "-c", COMPETITION, env=env)
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        print("\nDownloading official competition bundle...")
        kaggle_command(
            "competitions",
            "download",
            "-c",
            COMPETITION,
            "-p",
            str(DOWNLOAD_DIR),
            "--force",
            env=env,
        )
        archives = sorted(DOWNLOAD_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if not archives:
            raise RuntimeError("Kaggle reported success but no ZIP archive was found.")
        safe_extract(archives[-1], DOWNLOAD_DIR)
        validate_json_files(DOWNLOAD_DIR)
        print("\nKAGGLE_ACCESS_OK")
    finally:
        env.pop("KAGGLE_API_TOKEN", None)


if __name__ == "__main__":
    main()
