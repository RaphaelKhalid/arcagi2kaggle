# Baseline Analysis — `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` (public LB 33.89)

Source: pulled 2026-08-31. 9 cells; cells 3–6 are `%%writefile` modules (`arc_loader.py`,
`arc_decoder.py`, `arc_solver.py`, `starter.py`), cell 7 launches the solver, cell 8 builds
`submission.json`. Lineage: ARChitects 2024 pipeline + NVARC public Qwen3-4B checkpoint +
CPU↔GPU logits perf patch. Original run metadata: accelerator `nvidiaL4` (the 4×L4 machine),
docker image version 31090, internet off.

## Model

- Checkpoint: `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1`
  (Kaggle model source `sorokin/qwen3_4b_grids15_sft139`, framework `transformers`,
  variation `bfloat16`, version `1`) — Qwen3-4B, offline-SFT'd on NVARC synthetic grid data.
- Loaded with unsloth `FastLanguageModel`, bf16 (NOT 4-bit), `max_seq_length = 8192`,
  no gradient checkpointing. All fp32 params cast down to bf16 after PEFT wrap.
- Restricted vocab at decode: `ARC_VOCAB` = digits 0–9 → token ids 0–9, `Ċ` (newline) → 10,
  `<|im_end|>` → 15. Also `USER_TOKEN_ID=11`, `ASSISTANT_TOKEN_ID=12`, `PAD_ID=13`, `EOS_ID=15`.
  (The checkpoint has a 16-token grid tokenizer — "grids15".)

## Per-task flow (`worker()` in arc_solver.py)

Each of 4 worker processes (one per GPU, `CUDA_VISIBLE_DEVICES=rank`) loads the model once,
snapshots the fresh LoRA state (`default_weights`), then loops on a shared task queue:

1. **Reset**: `set_peft_model_state_dict(model, default_weights)` — LoRA re-initialized per task.
2. **TTT (test-time training)**:
   - Data: `puzzle_ds.augment(n=16, shfl_keys=True, seed=1)` → the D8 group (identity+transpose,
     then ×4 rotations with `keep`) × 16 random color permutations = **128 augmented copies** of
     the task, each with shuffled example order (`shuffle_ex`). Text format: all demo pairs as
     `<|im_start|>user\n<grid><|im_end|><|im_start|>assistant\n<grid><|im_end|>` turns, the LAST
     demo pair serving as the challenge (`last_is_challenge=True`; there is no test label).
     `cut_to_len(max_len=8192)` drops earliest demo pairs from oversized samples.
   - Collator masks everything except assistant grid spans (`j % 2 == 1` spans; labels −100 elsewhere).
   - LoRA: `r=256, lora_alpha=32, rslora=True, dropout=0`, targets = all attention+MLP projections
     **plus `embed_tokens` and `lm_head`**, seed 42.
   - Training args: 1 epoch, batch 1, grad-accum 1, lr `5e-5` cosine, `warmup_ratio=0.1`,
     `max_grad_norm=1.0`, adamw_torch, bf16 → ≤128 optimizer steps per task. **No per-task
     training time cap** (only the global end_time gates the next queue pull).
3. **Decode** (`torch.inference_mode`):
   - `split_multi_replies()` → one sub-key per test input (`{taskid}_{i}`).
   - Views: `augment(n=2, seed=2)` → 8 D8 transforms × 2 color permutations = **16 augmented
     views per test input**, decoded in **4 batches of 4 views** (batches deliberately mix
     rotation families; transposed views run in the last 2 batches).
   - Time caps: per-task decode loop breaks when `spend_time > 1200` s (20 min, checked between
     batches) or past global end_time; inside `turbo_dfs` a batch is additionally capped at
     **540 s** from DFS start plus the global end_time check.
   - `inference_turbo_dfs`: prefix = demos + test input (per view). Depth-first enumeration over
     the 12 ARC tokens keeping every completion with cumulative NLL `< max_score = −ln(0.2) ≈ 1.609`
     (i.e. all outputs the model assigns > 20% sequence probability, full-vocab softmax).
     `max_new_tokens` = tokenized 30×30 reply + 1 ≈ 932. The perf patch keeps full-vocab
     normalization on GPU and moves only the 12 ARC-token NLLs to CPU.
4. **Candidate scoring** (`calc_scores`): each decoded grid is inverted back through its view's
   transform (`invert_mod`), deduplicated by content, and teacher-forced-scored under
   `augment()` **8 augmented views** (D8 × 1 color perm, seed `hash(bk) % 1024**2`) in 2 batches
   of 4 → `score_aug` = list of 8 NLLs. Cached per `(task, grid)` in `known_scores`.
5. **Persist**: per view, `[{beam_score, score_aug, solution}, ...]` → bz2 pickle in
   `/kaggle/inference_outputs/{subkey}`.

## Selection (arc_decoder.py + cell 8)

`ArcDecoder.load_decoded_results` pools candidates across all 16 views per base key.
`score_sum` groups identical grids, scores each group, sorts descending; top-2 → attempts.

- `score_kgmon(group)` = `len(guesses)` (number of views that produced this grid) −
  `mean(mean(score_aug))` (average augmented NLL). **This is the DEFAULT** —
  cell 8 calls `decoder.run_selection_algo()` whose default arg is `score_kgmon`.
- `score_full_probmul_3(group)` = `Σ(3 − beam_score)` + `mean(Σ(3 − score_aug))` — a
  probability-mass style score. Computed in benchmarks but NOT used for the submission.

`get_submission` pre-fills **every** task/test-output with `[[0]]` placeholders (schema-valid
even for skipped tasks), then `fill_submission` overwrites attempts 1–2 from the ranked list.

## Time budget & coverage (the binding constraint)

- Cell 1: `global_end_time = t0 + 12 h − 600 s`; cell 7 passes it to starter.py.
- 240 tasks ÷ 4 workers = 60 tasks/GPU in ≈ 710 min → ~11.8 min/task average available.
  Worst-case per task ≈ TTT (uncapped, ~2–6 min) + up to 20 min decode. Tasks are pulled from a
  queue in **sorted key order** — no cost-aware scheduling; when time expires, remaining tasks
  keep `[[0]]` placeholders (guaranteed zeros). Coverage — not per-task accuracy — is what the
  perf patch bought; it is also our cheapest future lever.
- Eval mode (no `KAGGLE_IS_COMPETITION_RERUN`): starter.py runs **only 4 hardcoded tasks**
  (`0934a4d8, 36a08778, 981571dc, aa4ec2a5`) against the 120-task evaluation file — a ~26-min
  smoke test used for the "Save Version" commit; the full 12 h happens only in the scoring rerun.
- Data paths use the `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/...` layout.

## Memory

bf16 4B ≈ 8 GB weights + r=256 LoRA on all modules incl. embeddings (~1.6 GB) + optimizer
state + 8192-ctx activations; the notebook prints per-phase `max_memory_allocated` — fits a
22.5 GB L4 without checkpointing. One model instance per GPU; no cross-GPU sharding.

## Hard-coded constants worth knowing

| Constant | Value | Where |
|---|---|---|
| Global budget | 12 h − 600 s | cell 1 |
| TTT augmentations | 128 (8 D8 × 16 perms), seed 1 | `worker` |
| LoRA | r=256, α=32 rslora, all-modules+embed+head | `peft_params` |
| LR / schedule | 5e-5 cosine, warmup 0.1, 1 epoch, bs 1 | `train_args` |
| Context | 8192 | `max_seq_length` |
| Decode views | 16 (8 D8 × 2 perms), seed 2, batches of 4 | `worker` |
| DFS threshold | `−ln(0.2) ≈ 1.609` | `max_score` |
| DFS batch cap | 540 s | `turbo_dfs` |
| Per-task decode cap | 1200 s | `worker` |
| Scoring views | 8 (D8 × 1 perm), seed `hash(bk)%1024²` | `worker` |
| Selection default | `score_kgmon` | cell 8 / `run_selection_algo` |
| Eval-mode smoke tasks | 4 hardcoded | starter.py |

## Injection seams for planned upgrades

1. **Symbolic size/palette constraints → DFS** (`arc_solver.py`):
   compute constraints from `puzzle_ds` demos at the top of the task loop in `worker()`; pass
   them into `inference_turbo_dfs` → `turbo_dfs`. Two hooks: (a) the candidate loop
   `for token_idx, t in enumerate(ARC_TOKENS)` — mask colors outside the predicted palette,
   force `Ċ` at a predicted row width, force EOS at predicted height; (b) the `max_new_tokens`
   argument — tightening it from ~932 to the predicted output size is the single biggest DFS
   pruning lever. CAUTION: constraints must be transformed per view (height/width swap under
   rot90/transpose; palette must be mapped through the view's color permutation — parse the
   ops from the subkey suffix and reuse `forward_mod` semantics).
2. **Alternative selection algorithms** (`arc_decoder.py` / cell 8): add functions beside
   `score_kgmon` in `selection_algorithms`; candidates expose `beam_score`, 8 `score_aug`
   values, and the raw `solution` grid, and cell 8 has the original task (demos included) in
   `data` — symbolic consistency features (palette algebra, size rules, object counts) can be
   scored post-hoc without touching the solver. `benchmark_selection_algos` already reports
   candidate recall (`subkeys: solved/total`) vs ranking accuracy per algorithm — use it on
   local eval runs.
3. **Program-induction second solver**: safest seam is cell 8 — run induction over the same
   test JSON after the neural results load, execute programs against demos, and let any
   fully-verified program's output claim `attempt_1` (demote neural picks) before
   `get_submission`. Zero risk to the known-good solver; pure post-hoc override. A deeper
   alternative (injecting verified outputs as high-score candidates into
   `/kaggle/inference_outputs`) is available later.

## Fork delta (v1) — see notebook.ipynb final cell

`DIVERSE_ATTEMPT_2 = True`: attempt_1 stays the **score_kgmon top pick** (exact known-good
baseline attempt_1); attempt_2 becomes the top `score_full_probmul_3` pick when it differs;
when the two algorithms agree (or one has nothing), falls back to the exact baseline top-2 of
`score_kgmon`. `False` restores byte-identical baseline selection. NOTE: the parent plan
assumed `score_full_probmul_3` was the baseline default; deep-read shows the default is
`score_kgmon` — the fork therefore keeps kgmon as attempt_1 to avoid perturbing the known-good
behavior, and uses probmul_3 for diversity only.
