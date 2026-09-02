"""CPU-only tests for the Nemotron Lightning ARC probe."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from kaggle_nemotron_probe.probe_core import (
    aggregate_records,
    build_messages,
    build_user_prompt,
    evaluate_responses,
    extract_program,
    rank_verified_outputs,
    program_hash,
    run_candidate,
    validate_program,
    CandidateResult,
)
from kaggle_nemotron_probe.nemotron_induction import (
    SamplingLineage,
    make_sampling_lineages,
    verify_unit,
)
import kaggle_nemotron_probe.nemotron_induction as induction
from kaggle_nemotron_probe.run_probe import compute_folds
from arc.folds import compute_folds as harness_compute_folds


class NemotronProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = [
            {"input": [[0, 2]], "output": [[0, 3]]},
            {"input": [[2], [0]], "output": [[3], [0]]},
        ]
        self.good = """Reasoning.\n```python
def transform(grid):
    return [[3 if x == 2 else x for x in row] for row in grid]
```"""

    def test_nvidia_style_prompt(self) -> None:
        prompt = build_user_prompt(self.train, [[2]])
        self.assertIn("Train Example 1:", prompt)
        self.assertIn("Input:\n02", prompt)
        self.assertIn("Test Input:\n2", prompt)

    def test_prompt_styles_are_opt_in_and_deterministic(self) -> None:
        strict = build_messages(self.train, [[2]])[0]["content"]
        minimal = build_messages(self.train, [[2]], prompt_style="minimal")[0]["content"]
        freeform = build_messages(self.train, [[2]], prompt_style="freeform")[0]["content"]
        self.assertNotEqual(strict, minimal)
        self.assertNotEqual(minimal, freeform)
        self.assertEqual(strict, build_messages(self.train, [[2]])[0]["content"])
        with self.assertRaises(ValueError):
            build_messages(self.train, [[2]], prompt_style="unknown")

    def test_extract_and_validate(self) -> None:
        program = extract_program(self.good)
        self.assertIsNotNone(program)
        self.assertIsNone(validate_program(program or ""))

    def test_rejects_unsafe_import(self) -> None:
        source = "import os\ndef transform(grid):\n    return grid\n"
        self.assertIn("banned import", validate_program(source) or "")

    def test_rejects_source_and_ast_resource_exhaustion(self) -> None:
        oversized_source = "def transform(grid):\n    return grid\n" + ("#" * 32_001)
        self.assertIn("character budget", validate_program(oversized_source) or "")
        oversized_ast = "def transform(grid):\n" + "    x = 0\n" * 2_000 + "    return grid\n"
        self.assertIn("AST-node budget", validate_program(oversized_ast) or "")

    def test_partial_candidate_preserves_bounded_counterexample(self) -> None:
        wrong = "def transform(grid):\n    return [[0]]\n"
        result = run_candidate(
            wrong,
            [{"input": [[1]], "output": [[1]]}],
            [[1]],
            timeout_seconds=3,
        )
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.first_failed_demo, 0)
        self.assertEqual(result.observed, [[0]])
        self.assertEqual(result.expected, [[1]])

    def test_verified_oracle_and_selection_metrics(self) -> None:
        wrong = """```python
def transform(grid):
    return grid
```"""
        report = evaluate_responses(
            [self.good, self.good, wrong], self.train, [[2]], [[3]], timeout_seconds=3
        )
        self.assertEqual(report["n_parsed"], 3)
        self.assertEqual(report["n_unique_programs"], 2)
        self.assertEqual(report["n_verified_programs"], 1)
        self.assertTrue(report["oracle_correct"])
        self.assertTrue(report["top1_correct"])

    def test_aggregate(self) -> None:
        records = [
            {"n_responses": 2, "n_parsed": 2, "n_unique_programs": 2,
             "n_verified_programs": 1, "oracle_correct": True,
             "top1_correct": False, "top2_correct": True},
            {"n_responses": 2, "n_parsed": 1, "n_unique_programs": 1,
             "n_verified_programs": 0, "oracle_correct": False,
             "top1_correct": False, "top2_correct": False},
        ]
        summary = aggregate_records(records)
        self.assertEqual(summary["parse_rate"], 0.75)
        self.assertEqual(summary["oracle_pass_at_k"], 0.5)
        self.assertEqual(summary["top2_accuracy"], 0.5)

    def test_lineage_corrected_verified_output_masses(self) -> None:
        results = [
            CandidateResult("a", "verified", prediction=[[1]]),
            CandidateResult("b", "verified", prediction=[[1]]),
            CandidateResult("c", "verified", prediction=[[2]]),
        ]
        ranked = rank_verified_outputs(
            results,
            correlation_groups={"a": "batch-a", "b": "batch-a", "c": "batch-b"},
            collapse_correlated=True,
        )
        self.assertEqual([json_key for json_key, _, _ in ranked], ["[[1]]", "[[2]]"])
        self.assertAlmostEqual(ranked[0][1], 0.5)
        self.assertEqual(ranked[0][2], ("batch-a",))

    def test_probe_fold_assignment_matches_harness(self) -> None:
        task_ids = ["00576224", "9110e3c5", "c8b7cc0f", "0934a4d8", "aa4ec2a5"]
        self.assertEqual(compute_folds(task_ids), harness_compute_folds(task_ids))

    def test_lineage_plan_is_balanced_and_seeded(self) -> None:
        lineages = make_sampling_lineages(
            8, 4, seed=10, temperatures=(0.8, 1.0)
        )
        self.assertEqual([item.sample_count for item in lineages], [2, 2, 2, 2])
        self.assertEqual([item.seed for item in lineages], [10, 1019, 2028, 3037])
        self.assertEqual([item.temperature for item in lineages], [0.8, 1.0, 0.8, 1.0])

    def test_lineage_plan_can_stratify_prompt_styles(self) -> None:
        lineages = make_sampling_lineages(
            4, 4, prompt_styles=("strict", "minimal", "freeform")
        )
        self.assertEqual(
            [item.prompt_style for item in lineages],
            ["strict", "minimal", "freeform", "strict"],
        )

    def test_default_lineage_plan_preserves_single_batch_contract(self) -> None:
        lineages = make_sampling_lineages(6, 1)
        self.assertEqual(
            lineages,
            (SamplingLineage("lineage-0", 6, 260901, 1.0),),
        )
        self.assertEqual(lineages[0].sample_count, 6)
        self.assertEqual(lineages[0].seed, 260901)
        self.assertEqual(lineages[0].temperature, 1.0)

    def test_lineage_plan_rejects_overpartitioning(self) -> None:
        with self.assertRaises(ValueError):
            make_sampling_lineages(3, 4)

    def test_lineage_aware_unit_ranking_resists_one_batch_flood(self) -> None:
        wrong = [
            f"def transform(grid):\n    return [[0]]\n# variant-{index}"
            for index in range(10)
        ]
        right = "def transform(grid):\n    return [[1]]"

        def fake_run(source, train, test_input, timeout_seconds):
            prediction = [[1]] if "return [[1]]" in source else [[0]]
            return CandidateResult(
                program_hash(source), "verified", prediction=prediction,
                pairs_passed=len(train), n_pairs=len(train),
            )

        with patch.object(induction, "run_candidate", side_effect=fake_run):
            entry = verify_unit(
                {"train": self.train, "test_input": [[2]]},
                [],
                1.0,
                lineage_texts=(
                    ("flooded-batch", wrong),
                    ("independent-a", [right]),
                    ("independent-b", [right]),
                ),
                collapse_correlated=True,
            )
        self.assertEqual(entry["attempt"], [[1]])
        self.assertEqual(entry["alt"], [[0]])

    def test_shadow_diagnostics_persist_bounded_failed_candidate(self) -> None:
        wrong = "def transform(grid):\n    return [[0]]\n"
        entry = verify_unit(
            {"train": self.train, "test_input": [[2]]},
            [wrong],
            3.0,
            include_diagnostics=True,
            diagnostic_limit=1,
        )
        self.assertIsNone(entry["attempt"])
        self.assertEqual(len(entry["diagnostics"]), 1)
        diagnostic = entry["diagnostics"][0]
        self.assertEqual(diagnostic["lineage"], "lineage-0")
        self.assertEqual(diagnostic["result"]["first_failed_demo"], 0)
        self.assertEqual(diagnostic["result"]["observed"], [[0]])
        self.assertEqual(diagnostic["result"]["expected"], [[0, 3]])
        self.assertIn("def transform", diagnostic["trace"])


if __name__ == "__main__":
    unittest.main()
