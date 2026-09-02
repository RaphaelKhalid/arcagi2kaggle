# V2 Notebook Design — Three-Lineage Combination

> Status note (2026-09-02): the `>34.44` target is a legacy milestone, not the
> current top-eight objective. The active target is the 70s band described in
> `AUTORESEARCH_LOG_2026-09-01.md` Iteration 191.

Target: >34.44 (top-8 cutoff, rising). Ship only after V1 (baseline reproduction) returns a
clean score. Every element below exists publicly and is proven individually; the combination
is not (docs/recon-2026-08-31.md §4, §7).

## Time & GPU budget (12h − 10min safety)

| phase | GPUs | budget | what |
|-------|------|--------|------|
| Leg C: verified induction pre-pass | 0–3 | ~1.0h | Qwen2.5-Coder-7B samples DSL programs per task; AST-whitelisted sandbox executes against all demo pairs; fully-verified programs lock their task's attempt_1 and remove the task from the base queue |
| NVARC base (perfpatch) | 0–2 | ~9.8h | per-task LoRA TTT + turbo-DFS + augmentation rescoring, on the reduced queue |
| TRM leg | 3 | ~run-long | TRM TTT from `cpmpml/arc-prize-trm-031` (public lineage params: 4000 epochs, batch 112, lr 8.75e-5, 128 augs); GPU 3 joins the NVARC pool when done |
| merge + write | CPU | last 10min | monotonic merge: verified-induction > NVARC/TRM agreement > probmul ranking; diverse attempt_2; schema-validate before writing |

## Merge policy (monotonic, falsification-first)

1. A program that exactly reproduces every demo pair is near-proof — its output is attempt_1,
   unconditionally.
2. Where NVARC and TRM agree exactly, that output outranks either alone.
3. attempt_2 must differ from attempt_1: next-best from a *different* ranking family
   (probmul vs. vote-count vs. TRM), falling back to same-family #2.
4. Symbolic gate (our `symbolic/` predictors): candidates violating a fired size prediction
   are demoted, never promoted (precision-first: predictors only act when a rule is
   consistent with every demo).

## Required attachments (all public, verified in recon)

- model `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`
- model `qwen-lm/qwen2.5-coder/Transformers/7b-instruct` (handle TBV before push)
- kernel `sorokin/pip-install-unsloth-flash-patch` (cp311 wheels — keep pinned docker sha)
- datasets `cpmpml/arc-prize-trm-031`, `christopherdaleman/arc-proof-search-trm-2026-source`

## Risks

- Memory: Coder-7B + 4B base can't co-reside on one L4 (24 GB) at bf16 — phases must be
  strictly sequential per GPU with full unload between (del + empty_cache + gc).
- The TRM hybrid's ~40% claim is a projection, not an LB receipt; treat the TRM leg as an
  experiment whose merge is gated on local-eval evidence.
- Leg C sandbox: per-program timeouts and AST whitelist are inherited from the public
  implementation; review before trusting (arbitrary generated code execution).
- Each added attachment increases load time; measure startup in the V1 log before assuming
  the 12h budget holds.

## Beyond V2 (V3+ tracks)

- Evolution-style induction (Imbue-published mechanism; reimplemented, not vendored —
  AGPL): mid-size open coder model, population/mutation/fitness loop, demo-verified.
- Continued SFT of the 4B (or 2B sft141) on targeted synthetic data (docs/SYNTHETIC_DATA.md).
- Symbolic size/palette constraints injected directly into turbo_dfs (prune at decode time,
  not just post-hoc) once V2 establishes the seams.
