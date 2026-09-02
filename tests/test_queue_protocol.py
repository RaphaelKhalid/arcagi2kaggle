from pathlib import Path
import unittest


class QueueProtocolTests(unittest.TestCase):
    def test_solver_uses_sentinel_get_not_empty_polling(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "kaggle_nemotron_probe"
            / "arc_solver.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("while not queue.empty()", source)
        self.assertIn("while True:", source)
        self.assertIn("key = queue.get()", source)
        self.assertIn("if key is None:", source)


if __name__ == "__main__":
    unittest.main()
