from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LegCBudgetContractTests(unittest.TestCase):
    def test_launch_enforces_declared_budget_and_base_reserve(self):
        source = (ROOT / "kaggle_nemotron_probe" / "build_notebook.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("time.time() + LEGC_BUDGET_H * 3600", source)
        self.assertIn("global_end_time - BASE_RESERVE_H * 3600", source)
        self.assertIn("legc_end_ts = min(", source)
