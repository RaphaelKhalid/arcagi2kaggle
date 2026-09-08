"""Unit tests for the ARC Prize 2026 harness. Run: python -m tests.run_all (or pytest).

The notebook modules under kaggle_notebook/src are importable as flat modules here,
exactly as they are on Kaggle after the %%writefile cells run.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "kaggle_notebook" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
