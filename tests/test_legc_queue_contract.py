import ast
from pathlib import Path
import unittest


class LegCQueueContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (
            Path(__file__).resolve().parents[1]
            / "kaggle_nemotron_probe"
            / "starter.py"
        ).read_text(encoding="utf-8")

    def test_starter_defines_full_task_skip_helper(self) -> None:
        tree = ast.parse(self.source)
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("fully_verified_task_ids", functions)

    def test_queue_exclusion_uses_full_task_helper(self) -> None:
        self.assertIn("legc_skip = fully_verified_task_ids(json.load(f))", self.source)
        self.assertNotIn("if isinstance(r, dict) and r.get(\"verified\")", self.source)

    def test_partial_task_is_not_skipped_but_complete_task_is(self) -> None:
        tree = ast.parse(self.source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "fully_verified_task_ids"
        )
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        namespace = {}
        exec(compile(module, "<starter-helper>", "exec"), namespace)
        helper = namespace["fully_verified_task_ids"]
        ids = helper({
            "partial": {
                "verified": True,
                "outputs": [{"attempt": [[1]]}, {"attempt": None}],
            },
            "complete": {
                "verified": True,
                "outputs": [{"attempt": [[1]]}, {"attempt": [[2]]}],
            },
            "false": {"verified": False, "outputs": [{"attempt": [[1]]}]},
        })
        self.assertEqual(ids, {"complete"})


if __name__ == "__main__":
    unittest.main()
