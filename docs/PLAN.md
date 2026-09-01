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
4. **Program induction second barrel** — model writes Python transformations, executed and
   verified against demos; verified programs override ranking (near-zero false positives).
5. **Synthetic-data continued training** — NVARC-style consensus-verified task generation
   (see SYNTHETIC_DATA.md), continued fine-tuning of the public checkpoint.

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
