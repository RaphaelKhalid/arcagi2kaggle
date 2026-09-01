# Launch Runbook — baseline fork v1 (`raphaelkhalid0/arc2026-baseline-fork-v1`)

Everything below assumes PowerShell in the repo root with the token in env only
(never in files):

```powershell
$env:PYTHONPATH = (Resolve-Path .tools).Path
$env:KAGGLE_API_TOKEN = "<token>"   # or already set in the session
function kaggle { python -c "from kaggle.cli import main; main()" @args }
```

## 0. Preconditions (once, before first push)

- **USER APPROVAL REQUIRED** — pushing the kernel starts a run and burns GPU quota.
- Check quota headroom: `kaggle kernels quota` (name may vary; see `kaggle kernels -h`).
  A 12 h L4x4 run burns quota at **2× = 24 h of the ~30 h weekly budget**; the scored
  rerun after "Submit" is on Kaggle's dime, but the *commit* run is on ours.
  The commit run executes **eval mode** (no `KAGGLE_IS_COMPETITION_RERUN`): only 4
  hardcoded smoke tasks, ~26 min on L4x4 (~52 min quota).
- Confirm no submission was already made today (max 1/day):
  `kaggle competitions submissions -c arc-prize-2026-arc-agi-2`

## 1. Push (starts the commit run)

```powershell
kaggle kernels push -p kaggle_notebook
```

Metadata is `kernel-metadata.json` — copied field-for-field from the real captured
LB33.89 metadata (see scratchpad recon report §5). Do not "modernize" any of it:

- `machine_shape: "NvidiaL4"` → this string selects the **4×L4 (L4x4)** machine.
- `docker_image` sha `...320043e...` → pinned **Python 3.11** image. sorokin's
  attached unsloth/flash-attn wheels are cp311; the current default image is
  Py 3.12/Torch 2.10 and **will break them**. Keep the sha verbatim.
- `kernel_sources: ["sorokin/pip-install-unsloth-flash-patch"]` → the pinned wheel
  bundle the solver imports; the baseline does not work without it.
- `model_sources: ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]`
  (framework capitalized) → mounts at
  `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1`,
  exactly the path hard-coded in `arc_solver.py`.
- `enable_internet: false` is mandatory for submission eligibility on L4s.

## 2. Monitor the commit run

```powershell
kaggle kernels status raphaelkhalid0/arc2026-baseline-fork-v1
```

Poll until `complete` (eval mode ≈ 26 min). On `error`, fetch the log:

```powershell
kaggle kernels output raphaelkhalid0/arc2026-baseline-fork-v1 -p artifacts\kernel_run
```

## 3. Verify the commit output BEFORE submitting

From `artifacts\kernel_run`:

1. The log's tail must show `benchmark_selection_algos` output and a
   `*** Reload score: ...` line (eval mode ran end-to-end and re-parsed its own
   submission.json).
2. `submission.json` must exist. Validate schema and coverage locally:
   `python -m arc.cli validate artifacts\kernel_run\submission.json`
   (harness being built by another agent; until it lands, minimally check: valid JSON,
   all 120 eval task_ids present, every entry has both `attempt_1` and `attempt_2`).
   Note: in eval mode only the 4 smoke tasks have real predictions; the rest are
   `[[0]]` placeholders — that is expected and correct.
3. Log should show all 4 ranks starting (`[Rank 0..3] start!`) — confirms the L4x4
   machine and the 4-worker spawn.

## 4. Submit for scoring (needs user approval; uses the 1/day slot)

Code-competition submission = submitting the kernel version. CLI:

```powershell
kaggle competitions submit -c arc-prize-2026-arc-agi-2 -k raphaelkhalid0/arc2026-baseline-fork-v1 -v 1 -m "baseline fork v1: diverse attempt_2"
```

(`-k/--kernel` + `-v/--version` for code competitions; check `kaggle competitions submit -h`.
If the CLI form misbehaves, the fallback is the notebook page → "Submit to Competition".)
The scored rerun takes up to ~12 h + up to 10 min obfuscation variance; the score appears
on the My Submissions page / `kaggle competitions submissions`.

## 5. After the score lands

Record in the experiment ledger: date, kernel version, LB score, runtime, and the delta
vs 33.89 (attributable to DIVERSE_ATTEMPT_2 only). Public LB = ~50 % of 240 tasks, so
±0.8 is one task — do not over-read small deltas.

## Rollback

Set `DIVERSE_ATTEMPT_2 = False` in the final cell of `notebook.ipynb` and push again —
this restores the baseline's exact selection (top-2 by `score_kgmon`), byte-equivalent
pipeline behavior throughout.

## Known risks

- **Environment rot**: the pinned docker sha + cp311 wheels combination is what the
  33.89 pack still runs; if Kaggle ever retires the image, switch to the modernized
  wheel kernel `konstantinboyko/unsloth-2026-7-2-torch-2-10-0-cu128-patched` (see recon §5)
  and expect ABI patching work.
- **Coverage variance**: the solver processes tasks in sorted order until time expires;
  run-to-run timing noise moves the completed-task frontier a little; scores can wobble
  ~±0.5 without any code change.
- **Selection tweak risk**: diverse attempt_2 replaces kgmon's #2 with probmul_3's #1
  when they differ; the eval-mode benchmark prints both algorithms' accuracy — sanity
  check that in the commit log before submitting.
