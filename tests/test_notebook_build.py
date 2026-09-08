"""Contracts for the generated submission notebook and its CPU-only dry run."""

from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "kaggle_notebook"
sys.path.insert(0, str(NB_DIR))

import build_notebook  # noqa: E402
import dry_run  # noqa: E402


def cells(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"]]


class NotebookBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.notebook = build_notebook.build()
        self.sources = cells(self.notebook)

    def test_checked_in_notebook_matches_fresh_build(self) -> None:
        current = (NB_DIR / "notebook.ipynb").read_text(encoding="utf-8")
        self.assertEqual(current, build_notebook.render(self.notebook),
                         "notebook.ipynb is stale: run python kaggle_notebook/build_notebook.py")

    def test_every_writefile_module_compiles_and_matches_src(self) -> None:
        seen = []
        for src in self.sources:
            m = re.match(r"%%writefile (\S+)\n", src)
            if not m:
                continue
            name = m.group(1)
            seen.append(name)
            body = src[m.end():]
            ast.parse(body, filename=name)
            self.assertEqual(body, build_notebook.module_source(name, build_notebook.PROFILE))
        expected = list(build_notebook.BASE_MODULES)
        if build_notebook.PROFILE["LEGC_ENABLED"]:
            expected.append(build_notebook.LEGC_MODULE)
        self.assertEqual(seen, expected)

    def test_profile_flags_are_reflected_in_cells(self) -> None:
        joined = "\n".join(self.sources)
        for flag in ("CHEAP_FIRST_ORDER", "SIZE_CAP_TOKENS", "DIVERSE_ATTEMPT_2"):
            self.assertIn(f"{flag} = {build_notebook.PROFILE[flag]}", joined)
        legc_cells = [s for s in self.sources if "arc_induction_v2.py" in s or "[LegC merge]" in s]
        if build_notebook.PROFILE["LEGC_ENABLED"]:
            self.assertEqual(len(legc_cells), 3)
        else:
            self.assertEqual(legc_cells, [])
            self.assertIn("LEGC_ENABLED = False", joined)

    def test_notebook_cell_order(self) -> None:
        self.assertEqual(self.notebook["cells"][0]["cell_type"], "markdown")
        code = [s for s in self.sources[1:]]
        self.assertIn("global_end_time = time.time() + 12 * 3600 - 600", code[0])
        self.assertIn("pip uninstall -y tensorflow", code[1])
        launch = [i for i, s in enumerate(code) if "python starter.py --end-time {global_end_time}" in s]
        select = [i for i, s in enumerate(code) if "submission.json" in s and "get_submission" in s]
        self.assertEqual(len(launch), 1)
        self.assertEqual(len(select), 1)
        self.assertLess(launch[0], select[0])
        writefiles = [i for i, s in enumerate(code) if s.startswith("%%writefile")]
        self.assertLess(max(writefiles), launch[0])
        # the launcher is wrapped in coreutils timeout so a hung worker cannot block the selection cell
        self.assertIn("timeout -k 30 {launcher_timeout}", code[launch[0]])

    def test_kernel_metadata_matches_profile(self) -> None:
        self.assertEqual(build_notebook.check_metadata(build_notebook.PROFILE), [])

    def test_kernel_metadata_rejects_missing_coder_model_when_legc_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = json.loads((NB_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))
            meta["model_sources"] = [build_notebook.BASE_MODEL]
            path = Path(tmp) / "kernel-metadata.json"
            path.write_text(json.dumps(meta), encoding="utf-8")
            profile = dict(build_notebook.PROFILE, LEGC_ENABLED=True)
            problems = build_notebook.check_metadata(profile, path)
            self.assertTrue(any(build_notebook.CODER_MODEL in p for p in problems))

    def test_selection_cell_writes_submission_before_diagnostics(self) -> None:
        cell = build_notebook.selection_cell(build_notebook.PROFILE)
        self.assertLess(cell.index('json.dump(submission, f)'), cell.index("benchmark_selection_algos"))
        self.assertIn("try:", cell)

    def test_src_defaults_match_profile(self) -> None:
        for name, flags in build_notebook.MODULE_FLAGS.items():
            text = (build_notebook.SRC / name).read_text(encoding="utf-8")
            for flag in flags:
                self.assertIn(f"\n{flag} = {build_notebook.PROFILE[flag]}\n", text)


def synthetic_tasks(n: int) -> tuple[dict, dict]:
    """Tiny deterministic tasks so the dry run needs no competition data."""
    challenges, solutions = {}, {}
    for k in range(n):
        tid = f"{k:08x}"
        size = 2 + (k % 3)
        grid = [[(r * size + c + k) % 10 for c in range(size)] for r in range(size)]
        out = [[(v + 1) % 10 for v in row] for row in grid]
        n_test = 2 if k % 4 == 3 else 1
        challenges[tid] = {
            "train": [{"input": grid, "output": out}, {"input": out, "output": grid}],
            "test": [{"input": grid} for _ in range(n_test)],
        }
        solutions[tid] = [out for _ in range(n_test)]
    return challenges, solutions


class DryRunTests(unittest.TestCase):
    def test_selection_cell_end_to_end_eval_and_rerun_modes(self) -> None:
        challenges, solutions = synthetic_tasks(12)
        for rerun in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                report = dry_run.dry_run(challenges, solutions, 12, seed=1, rerun=rerun, keep=Path(tmp))
                self.assertEqual(report["tasks"], 12)
                self.assertGreater(report["correct_outputs"], 0)
                self.assertLess(report["correct_outputs"], report["total_outputs"])

    def test_no_candidates_at_all_still_writes_valid_placeholders(self) -> None:
        challenges, solutions = synthetic_tasks(3)
        dry_run.install_stubs()
        from arc.validate import validate_submission

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "working"
            work.mkdir()
            ch = Path(tmp) / "c.json"
            sol = Path(tmp) / "s.json"
            ch.write_text(json.dumps(challenges), encoding="utf-8")
            sol.write_text(json.dumps(solutions), encoding="utf-8")
            # missing directory: workers never started
            sub = dry_run.run_cell(dry_run.selection_cell_source(False), ch, sol, Path(tmp) / "missing", work, False)
            self.assertEqual(validate_submission(sub, challenges), [])
            self.assertTrue(all(e == {"attempt_1": [[0]], "attempt_2": [[0]]} for v in sub.values() for e in v))


if __name__ == "__main__":
    unittest.main()
