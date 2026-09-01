# Campaign Plan — ARC Prize 2026 (ARC-AGI-2)

Deadline: final submissions Nov 2, 2026 (23:59 UTC); writeup artifacts within 7 days after.
Prize targets: top-8 progress prize (cutoff 34.44 and rising as of Aug 31) and the $275k
Grand Prize writeup (judged on accuracy, universality, progress, theory, completeness, novelty).

## Strategy

Stand on the best public baseline (NVARC-lineage TTT pipeline, public LB 33.89), then add
measured improvements, prioritizing neurosymbolic components that are both point-scoring and
writeup-novel:

1. **Selection surgery** — diverse attempt_2 (different ranking families), symbolic
   consistency re-ranking of candidates.
2. **Symbolic constraint injection** — falsification-based size/palette predictors prune the
   DFS decode; freed runtime goes into more TTT/candidates.
3. **Time-budget reallocation** — solve-rate-aware task scheduling across the 4 GPUs.
4. **Three-lineage combination (recon finding: no public notebook does this)** — the
   perfpatch GPU optimizations + the "Leg C" verified program induction pre-pass
   (Qwen2.5-Coder-7B samples programs, sandbox-verified against demos, from the
   `failed-in-aimo` head) + the TRM test-time-training ensemble (CPMP's public
   `arc-prize-trm-031` checkpoints). Near-disjoint solve sets ⇒ additive points.
5. **Evolution-style program induction** (the likely 70%-regime mechanism, per Imbue's
   published code-evolution results: Kimi K2.5 12→34%, Gemini 3.1 Pro 88→95% on public
   eval) — population of Python programs per task, LLM mutations, fitness = demo
   correctness + simplicity, executed/verified locally. Reimplement ourselves (Imbue's
   framework is AGPL v3 — license-incompatible with the CC-BY 4.0 winner obligation; the
   loop is simple). Run on-Kaggle with a mid-size open coder model.
6. **Synthetic-data continued training** — the full NVARC synthetic corpus is PUBLIC on
   Kaggle (sorokin's datasets: synthetic 338 MB, augmented 1.3 GB, artifacts 42 GB).
   Train on it directly; the $50 OpenAI budget goes to NEW consensus-verified tasks
   targeting our measured failure categories instead (see SYNTHETIC_DATA.md).

## Operating rules

- Every change is measured on the 120-task evaluation set locally before it ships in a
  scored submission. Fold 4 of the training split is a shadow fold: untouched between
  major milestones.
- One scored submission per day is a measurement; never submit two uncontrolled changes
  at once. Log every submission in `experiments/LEDGER.md`.
- Submission #1 is the reproduced baseline: it validates schema, runtime, model loading,
  and the hidden-rerun path end-to-end before any cleverness ships.
- No credentials on disk, ever. Competition data never gets committed or redistributed.

## Budget notes

- Hidden set: 240 tasks, ~260 test outputs, 12h wall clock on L4x4 (96 GB) ⇒ ≈12
  GPU-minutes per task average. Serialized tasks are ≤~8.4k tokens (p99 ≈6-8k).
- Kaggle GPU quota ~30h/week; L4x4 burns at 2× ⇒ one full submission run ≈ 24h quota.
  Debug runs must be short or CPU-only.
- OpenAI API budget for synthetic data: $50 hard cap, ledger in SYNTHETIC_DATA.md.
