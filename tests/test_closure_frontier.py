import unittest

from experiments.closure_frontier import inspect_dataset, inspect_task


class ClosureFrontierTests(unittest.TestCase):
    def test_supported_recolor_is_verified(self) -> None:
        result = inspect_task(
            "t", {"train": [{"input": [[0, 1]], "output": [[0, 2]]}]}
        )
        self.assertEqual(result.status, "closed_verified")
        self.assertTrue(result.language_closed)
        self.assertTrue(result.verified)
        self.assertEqual(result.unsupported_labels, frozenset())

    def test_grid_resize_is_a_language_gap(self) -> None:
        result = inspect_task(
            "t", {"train": [{"input": [[0, 1]], "output": [[0, 1, 1]]}]}
        )
        self.assertEqual(result.status, "language_gap")
        self.assertIn("grid_resize", result.unsupported_labels)
        self.assertFalse(result.language_closed)

    def test_empty_and_dataset_order_are_explicit(self) -> None:
        challenges = {
            "empty": {"train": []},
            "recolor": {"train": [{"input": [[0, 1]], "output": [[0, 2]]}]},
        }
        results = inspect_dataset(challenges)
        self.assertEqual([item.task_id for item in results], ["empty", "recolor"])
        self.assertEqual(results[0].status, "empty")
        self.assertEqual(results[1].status, "closed_verified")


if __name__ == "__main__":
    unittest.main()
