"""Kaggle CLI wrapper: bootstraps the vendored .tools packages onto sys.path.

Usage: python scripts/kaggle_cli.py <kaggle-cli-args...>
Auth: the Kaggle CLI reads ~/.kaggle/access_token natively (or KAGGLE_API_TOKEN
from the environment). No credentials live in this repository.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".tools"))

from kaggle.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.argv[0] = "kaggle"
    main()
