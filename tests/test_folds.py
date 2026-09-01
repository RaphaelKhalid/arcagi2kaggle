"""Tests for arc.folds: determinism, balance, and the shadow fold."""

import subprocess
import sys
import unittest

from arc.folds import (
    NUM_FOLDS,
    SHADOW_FOLD,
    compute_folds,
    dev_task_ids,
    fold_members,
    shadow_task_ids,
)

IDS = [f"task{i:04d}" for i in range(100)]


class FoldTests(unittest.TestCase):
    def test_deterministic_same_process(self) -> None:
        self.assertEqual(compute_folds(IDS), compute_folds(IDS))

    def test_order_and_duplicates_do_not_matter(self) -> None:
        self.assertEqual(compute_folds(IDS), compute_folds(list(reversed(IDS))))
        self.assertEqual(compute_folds(IDS), compute_folds(IDS + IDS[:10]))

    def test_deterministic_across_processes(self) -> None:
        # Guards against PYTHONHASHSEED / dict-order leakage: run the fold
        # computation in two fresh interpreters and compare the output.
        code = (
            "from arc.folds import compute_folds;"
            "ids=[f'task{i:04d}' for i in range(100)];"
            "print(sorted(compute_folds(ids).items()))"
        )
        outs = [
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, check=True,
            ).stdout
            for _ in range(2)
        ]
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(outs[0], f"{sorted(compute_folds(IDS).items())}\n")

    def test_balanced_folds(self) -> None:
        assignment = compute_folds(IDS)
        members = fold_members(assignment)
        self.assertEqual(set(members), set(range(NUM_FOLDS)))
        sizes = [len(members[f]) for f in range(NUM_FOLDS)]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        self.assertEqual(sum(sizes), len(IDS))

    def test_all_folds_in_range(self) -> None:
        assignment = compute_folds(IDS)
        self.assertTrue(all(0 <= f < NUM_FOLDS for f in assignment.values()))

    def test_shadow_and_dev_partition(self) -> None:
        assignment = compute_folds(IDS)
        shadow = shadow_task_ids(assignment)
        dev = dev_task_ids(assignment)
        self.assertEqual(sorted(shadow + dev), sorted(IDS))
        self.assertTrue(all(assignment[t] == SHADOW_FOLD for t in shadow))
        self.assertTrue(all(assignment[t] != SHADOW_FOLD for t in dev))

    def test_known_pin(self) -> None:
        # Pin a tiny assignment so any change to the fold algorithm is loud.
        self.assertEqual(
            compute_folds(["a", "b", "c", "d", "e"]),
            {"d": 0, "c": 1, "b": 2, "e": 3, "a": 4},
        )


if __name__ == "__main__":
    unittest.main()
