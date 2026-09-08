# Launch Runbook — `raphaelkhalid0/arc2026-baseline-fork-v1`

Everything runs from the repo root. The Kaggle token lives only in the environment
(`KAGGLE_API_TOKEN`, or `~/.kaggle/access_token` written by `kaggle` itself); never in a file
here. `python scripts/kaggle_cli.py` uses the vendored `.tools` packages if present, otherwise
`pip install kaggle`. PowerShell users: `$env:KAGGLE_API_TOKEN = "<token>"`; bash:
`export KAGGLE_API_TOKEN=<token>`.

Timeline for an overnight run: push (1 min) → commit run (~30–40 min on 4xL4) → verify (2 min)
→ submit → scoring rerun (up to 12 h, on Kaggle's quota) → score on the leaderboard page.

## 0. Preconditions

- GPU quota: the commit run burns ~1–1.5 h of the ~30 h weekly L4x4 quota (2× multiplier); the
  scoring rerun is free. `python scripts/kaggle_cli.py kernels status raphaelkhalid0/arc2026-baseline-fork-v1`
  must not show a run in progress.
- No submission today yet (1/day):
  `python scripts/kaggle_cli.py competitions submissions -c arc-prize-2026-arc-agi-2`
- The notebook is built from the sources and the tests pass:

```bash
python kaggle_notebook/build_notebook.py --check
python -m tests.run_all
```

## 1. Push (starts the commit run)

```bash
python scripts/kaggle_cli.py kernels push -p kaggle_notebook
```

`kernel-metadata.json` is validated by the builder. Do not modernize it:

- `machine_shape: "NvidiaL4"` selects the 4×L4 machine.
- `docker_image` sha `...320043e...` is the pinned Python 3.11 image; the attached unsloth/flash-attn
  wheels (`kernel_sources: sorokin/pip-install-unsloth-flash-patch`) are cp311 and break on newer images.
- `model_sources: sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1` mounts at the path hard-coded
  in `arc_solver.py`. The Qwen2.5-Coder model is detached because Leg C is off.
- `enable_internet: false` is mandatory for submission eligibility.

## 2. Monitor the commit run

```bash
python scripts/kaggle_cli.py kernels status raphaelkhalid0/arc2026-baseline-fork-v1
```

The commit run executes eval mode (no `KAGGLE_IS_COMPETITION_RERUN`): 4 smoke tasks against the public
evaluation file, ~30 min. On `error`, fetch the log:

```bash
python scripts/kaggle_cli.py kernels output raphaelkhalid0/arc2026-baseline-fork-v1 -p artifacts/kernel_run
```

## 3. Verify BEFORE submitting

```bash
python scripts/verify_kernel_run.py
```

It downloads the output and fails unless: status complete, `submission.json` present and schema-valid
against the evaluation challenges (needs `data/raw/`; see `scripts/make_local_data.py`), all four
`[Rank n] start!` lines, at least one `[Rank n] finished <task>` line, the `submission.json written` and
`Reload score:` lines, and no `Traceback`. In eval mode only the 4 smoke tasks have real predictions; the
rest are `[[0]]` placeholders — expected. Also glance at the log for `size caps for <task>` lines
(the size predictor fired) and the `acc:` lines of the two selection algorithms.

## 4. Submit for scoring (uses the 1/day slot)

Web UI: open the notebook version on Kaggle → "Submit to Competition". CLI alternative (code competition
= submit a kernel version; check `python scripts/kaggle_cli.py competitions submit -h` for the flags):

```bash
python scripts/kaggle_cli.py competitions submit -c arc-prize-2026-arc-agi-2 \
    -k raphaelkhalid0/arc2026-baseline-fork-v1 -v <version> -m "v2: Leg C off, cheap-first + size cap + diverse attempt_2"
```

The scoring rerun takes up to 12 h; the score appears on the My Submissions page.

## 5. Record

Add the row to `experiments/LEDGER.md` (date, kernel version, LB score, delta). Public LB is ~120 tasks,
so ±0.8 is one task; do not over-read small deltas.

## Switches (all in `PROFILE`, `kaggle_notebook/build_notebook.py`)

| switch | current | effect | evidence |
|--------|---------|--------|----------|
| `CHEAP_FIRST_ORDER` | on | queue cost-ascending (coverage ↑) | deterministic, complete, monotone on training/evaluation files |
| `SIZE_CAP_TOKENS` | on | DFS capped at predicted grid tokens per augmented view | 882/882 + 107/107 correct fires on public data; view parity verified |
| `DIVERSE_ATTEMPT_2` | on | attempt_2 from probmul_3 when it disagrees with kgmon | unit tests; dry run |
| `LEGC_ENABLED` | **off** | Qwen2.5-Coder-7B verified induction pre-pass (1 h budget) | the run with it on scored 29.31 vs 33.89 baseline |

Always-on robustness (no effect on results): sentinel worker loop, deadline-aware test-time training,
per-task crash guard with a two-failure circuit breaker, `timeout`-wrapped launcher, unreadable candidate
files skipped, crash-proof eval-mode diagnostics, smoke-task fallback.

To change a switch: edit `PROFILE` **and** the matching default in `src/` (the builder tells you if they
disagree), rebuild, run the tests, push. All switches off = byte-equivalent baseline behaviour. Turning
Leg C back on also requires re-attaching `qwen-lm/qwen2.5-coder/Transformers/7b-instruct/1` in
`kernel-metadata.json` (the builder checks this).

## Known risks

- **Environment rot**: the pinned image + cp311 wheels is what the 33.89 lineage runs; if Kaggle retires the
  image, switch to `konstantinboyko/unsloth-2026-7-2-torch-2-10-0-cu128-patched` (see
  `docs/recon-2026-08-31.md` §5) and expect ABI work.
- **Coverage variance**: the run processes tasks until time expires; timing noise moves the completed-task
  frontier and the score can wobble ±0.5 with no code change.
- **Kaggle is unreachable from Claude Code sandboxes** (proxy denies kaggle.com), so pushing, monitoring and
  submitting must be done from your own machine.
