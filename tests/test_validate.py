"""Tests for arc.validate: every malformation must be caught."""

import copy
import unittest

from arc.validate import validate_submission

CHALLENGES = {
    "task_a": {"train": [], "test": [{"input": [[1]]}]},
    "task_b": {"train": [], "test": [{"input": [[1]]}, {"input": [[2]]}]},
}


def good_submission() -> dict:
    return {
        "task_a": [{"attempt_1": [[0]], "attempt_2": [[1]]}],
        "task_b": [
            {"attempt_1": [[2]], "attempt_2": [[3]]},
            {"attempt_1": [[4]], "attempt_2": [[5]]},
        ],
    }


def codes(issues) -> set[str]:
    return {i.code for i in issues}


class ValidateTests(unittest.TestCase):
    def test_valid_submission_passes(self) -> None:
        self.assertEqual(validate_submission(good_submission(), CHALLENGES), [])

    def test_not_an_object(self) -> None:
        self.assertEqual(codes(validate_submission([], CHALLENGES)), {"not_object"})
        self.assertEqual(codes(validate_submission(None, CHALLENGES)), {"not_object"})

    def test_missing_task(self) -> None:
        sub = good_submission()
        del sub["task_b"]
        issues = validate_submission(sub, CHALLENGES)
        self.assertEqual(codes(issues), {"missing_task"})
        self.assertEqual(issues[0].task_id, "task_b")

    def test_unknown_task(self) -> None:
        sub = good_submission()
        sub["task_zzz"] = []
        self.assertIn("unknown_task", codes(validate_submission(sub, CHALLENGES)))

    def test_wrong_test_count(self) -> None:
        sub = good_submission()
        sub["task_b"] = sub["task_b"][:1]  # too few
        issues = validate_submission(sub, CHALLENGES)
        self.assertEqual(codes(issues), {"wrong_test_count"})
        sub = good_submission()
        sub["task_a"].append({"attempt_1": [[0]], "attempt_2": [[0]]})  # too many
        self.assertIn("wrong_test_count", codes(validate_submission(sub, CHALLENGES)))

    def test_task_value_not_a_list(self) -> None:
        sub = good_submission()
        sub["task_a"] = {"attempt_1": [[0]], "attempt_2": [[0]]}
        self.assertIn("bad_task_type", codes(validate_submission(sub, CHALLENGES)))

    def test_entry_not_a_dict(self) -> None:
        sub = good_submission()
        sub["task_a"] = [[[0]]]
        self.assertIn("bad_entry_type", codes(validate_submission(sub, CHALLENGES)))

    def test_missing_attempt_2(self) -> None:
        sub = good_submission()
        del sub["task_a"][0]["attempt_2"]
        issues = validate_submission(sub, CHALLENGES)
        self.assertEqual(codes(issues), {"missing_attempt"})
        self.assertEqual(issues[0].attempt, "attempt_2")

    def test_missing_attempt_1(self) -> None:
        sub = good_submission()
        del sub["task_b"][1]["attempt_1"]
        issues = validate_submission(sub, CHALLENGES)
        self.assertEqual(codes(issues), {"missing_attempt"})
        self.assertEqual((issues[0].task_id, issues[0].index), ("task_b", 1))

    def test_extra_key(self) -> None:
        sub = good_submission()
        sub["task_a"][0]["attempt_3"] = [[0]]
        self.assertIn("extra_key", codes(validate_submission(sub, CHALLENGES)))

    def test_non_int_cells(self) -> None:
        for bad_cell in ["3", 3.0, None, True]:
            sub = good_submission()
            sub["task_a"][0]["attempt_1"] = [[bad_cell]]
            issues = validate_submission(sub, CHALLENGES)
            self.assertEqual(codes(issues), {"bad_grid"}, msg=repr(bad_cell))

    def test_out_of_range_colors(self) -> None:
        for bad in ([[10]], [[-1]]):
            sub = good_submission()
            sub["task_b"][0]["attempt_2"] = bad
            issues = validate_submission(sub, CHALLENGES)
            self.assertEqual(codes(issues), {"bad_grid"}, msg=repr(bad))
            self.assertEqual(issues[0].attempt, "attempt_2")

    def test_ragged_grid(self) -> None:
        sub = good_submission()
        sub["task_a"][0]["attempt_1"] = [[1, 2], [3]]
        issues = validate_submission(sub, CHALLENGES)
        self.assertEqual(codes(issues), {"bad_grid"})
        self.assertIn("ragged", issues[0].message)

    def test_empty_and_oversized_grids(self) -> None:
        for bad in ([], [[]], [[0]] * 31, [[0] * 31]):
            sub = good_submission()
            sub["task_a"][0]["attempt_1"] = copy.deepcopy(bad)
            self.assertEqual(
                codes(validate_submission(sub, CHALLENGES)), {"bad_grid"},
                msg=repr(bad),
            )

    def test_multiple_issues_all_reported(self) -> None:
        sub = good_submission()
        del sub["task_a"]
        del sub["task_b"][0]["attempt_2"]
        sub["task_b"][1]["attempt_1"] = [[42]]
        issues = validate_submission(sub, CHALLENGES)
        self.assertEqual(
            codes(issues), {"missing_task", "missing_attempt", "bad_grid"}
        )
        self.assertEqual(len(issues), 3)


if __name__ == "__main__":
    unittest.main()
