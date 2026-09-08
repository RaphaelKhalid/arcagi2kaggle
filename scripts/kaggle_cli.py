"""Kaggle CLI wrapper.

Usage: python scripts/kaggle_cli.py <kaggle-cli-args...>

Uses the vendored ``.tools`` packages when present, otherwise an installed
``kaggle`` package (``pip install kaggle``).  Auth: the Kaggle CLI reads
``KAGGLE_API_TOKEN`` from the environment (or ``~/.kaggle/access_token``).
No credentials live in this repository; never write a kaggle.json.
"""

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / ".tools"
if TOOLS.is_dir():
    sys.path.insert(0, str(TOOLS))

try:
    from kaggle.cli import main  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment dependent
    sys.exit(
        f"kaggle package not found ({exc}). Either vendor it under .tools/ or run: pip install kaggle"
    )

if __name__ == "__main__":
    sys.argv[0] = "kaggle"
    main()
