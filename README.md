# arcagi2kaggle

Open-source campaign for the [ARC Prize 2026 — ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
Kaggle competition. Public research brief: [ARC/RESEARCH](https://raphaelkhalid.github.io/arcagi2kaggle/).

## Status

| Run | Notebook | Public LB |
|-----|----------|-----------|
| Upstream baseline `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` | NVARC Qwen3-4B test-time-training pipeline, no flags | 33.89 |
| Submission #1 "Launch B" (2026-08-31) | four flags on, including the Leg C program-induction pre-pass | 29.31 |
| Submission #2 (ready to push) | Leg C off, three low-risk flags on, worker robustness fixes | pending |

Submission #2 is the current `kaggle_notebook/notebook.ipynb`. It is a controlled step: the only
behavioural difference from submission #1 is that Leg C no longer takes ~1 h of the 12 h budget
and no longer removes tasks from the base solver's queue, so the result measures Leg C's effect
directly. Launch steps: [`kaggle_notebook/RUNBOOK.md`](kaggle_notebook/RUNBOOK.md).

## Approach

The NVARC-lineage solver: per-task LoRA test-time training of the public `sorokin/qwen3_4b_grids15_sft139`
checkpoint on augmented demonstrations, depth-first decoding over the 12 grid tokens, augmentation-consistency
scoring, top-2 selection. On top of it, each behind its own switch in `kaggle_notebook/build_notebook.py`:

- `CHEAP_FIRST_ORDER` — task queue ordered by serialized cost ascending (coverage is the binding constraint;
  unfinished tasks score 0).
- `SIZE_CAP_TOKENS` — a falsification-based output-size predictor caps DFS decoding at the predicted grid's
  token count. It fires only when a rule explains every demonstration; 100% precise on every fired training
  and evaluation output (882/882 and 107/107 on the public data).
- `DIVERSE_ATTEMPT_2` — attempt_2 is the best `score_full_probmul_3` grid that differs from attempt_1 (a second
  ranking family instead of rank 2 of the same ranking).
- `LEGC_ENABLED` (off) — verified program induction with Qwen2.5-Coder-7B as a pre-pass.

## Layout

- `kaggle_notebook/` — **the deliverable.** `src/` holds the notebook modules (single source of truth),
  `build_notebook.py` assembles `notebook.ipynb` from them, `kernel-metadata.json` is the Kaggle kernel
  spec, `dry_run.py` executes the real selection cell on fabricated candidates, `RUNBOOK.md` is the launch
  procedure and `BASELINE_ANALYSIS.md` documents the pipeline. Never edit `notebook.ipynb` by hand.
- `arc/` — evaluation harness: task loading, strict submission validation, official-metric scoring,
  deterministic 5-fold split (`python -m arc.cli --help`).
- `symbolic/` — the size/palette predictors with their measurement runner (`python -m symbolic.measure`).
  `kaggle_notebook/src/symbolic_size.py` is the dependency-free copy embedded in the notebook; the test
  suite and `scripts/` audits keep them equivalent.
- `scripts/` — `kaggle_cli.py` (CLI wrapper, token from env only), `verify_kernel_run.py` (checks a commit
  run before you submit), `secure_kaggle_download.py` (official data, interactive), `make_local_data.py`
  (public ARC-AGI-2 task files → Kaggle layout for local dry runs).
- `tests/` — `python -m tests.run_all` (or `pytest`). Covers the harness, the notebook build, the dry run,
  the worker-queue contracts and the research modules.
- `experiments/`, `docs/AUTORESEARCH_*.md`, `experiments/LEDGER.md` — CPU-only research modules and the
  experiment ledger from the September autoresearch loop. Nothing under `experiments/` is wired into the
  notebook; promote a module only through the gates in `docs/KAGGLE_NOTEBOOK_HANDOFF_TODO.md`.
- `kaggle_nemotron_probe/` — experimental Nemotron-30B induction fork. It reuses `kaggle_notebook/src/`
  for the base pipeline and needs a private model attachment; not a tonight candidate.
- `data/`, `artifacts/`, `.tools/` — gitignored: competition data, kernel outputs, vendored packages.

## Quick start

```bash
pip install -r requirements-dev.txt
python -m tests.run_all                                   # 448 tests, CPU only
python kaggle_notebook/build_notebook.py --check          # notebook.ipynb matches src/ + PROFILE
# optional local data for dry runs / audits (public task files, Apache 2.0):
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2 /tmp/ARC-AGI-2
python scripts/make_local_data.py /tmp/ARC-AGI-2
python kaggle_notebook/dry_run.py --challenges data/raw/arc-agi_evaluation_challenges.json \
    --solutions data/raw/arc-agi_evaluation_solutions.json
```

## Conventions

- Credentials never touch disk: no `kaggle.json`, no tokens in files. Kaggle auth is `KAGGLE_API_TOKEN` in
  the process environment only.
- One scored submission per day; every one is a controlled measurement and gets a row in
  `experiments/LEDGER.md`. Change one thing at a time versus the previous scored run.
- All switches live in `PROFILE` in `build_notebook.py`; the builder refuses to build if the module defaults
  drift, and the test suite refuses to pass if the checked-in notebook is stale.

## License

MIT-0 (see `LICENSE`). Competition data is Apache 2.0, © ARC Prize Foundation, and is not included here.
