# ARC-AGI-2 Autoresearch Checkpoint

Checkpoint date: 2026-09-02  
Goal runtime: 9h 29m 02s (34,142 seconds)  
Goal token usage: 6,708,326 tokens  
Research state: paused after the user's stop request; safe to resume from this file.

## Objective

Find a mathematically defensible path toward the current ARC-AGI-2 top-eight
frontier (roughly the low 70s), under the Kaggle constraints of four L4 GPUs,
one submission per day, a 12-hour run, up to two final submissions, and the
winner licensing requirement. The loop was theoretical and CPU-only: it did not
fine-tune a model, run GPUs, submit to Kaggle, or use hidden labels.

## What was explored

The campaign produced 205 ledger entries and reached numbered Iteration 217.
There are 190 explicit `## Iteration` sections because some investigations were
bundled, renumbered, or recorded as queue/design notes. The complete chronology
is in `AUTORESEARCH_LOG_2026-09-01.md`; `experiments/LEDGER.md` is the compact
decision ledger.

The work covered:

1. **The score mechanism.** ARC scoring is output-position based, so reaching
   72 requires candidate-class recall, calibrated selection, and non-correlated
   second attempts—not merely a better task-level average. A two-final portfolio
   is a separate optimization problem: each final must be a complete valid
   submission, and the portfolio should hedge complementary private-half risk.

2. **Proof-carrying synthesis.** The central abstraction became a finite
   version space of programs/transformations consistent with all demonstrations.
   Exact demo fit is evidence, not proof of target correctness; safe promotion
   needs typed effects, counterexamples, complexity/MDL control, and disjoint-fold
   calibration. Monotone merge rules preserve incumbent recall when a new lane
   abstains or is malformed.

3. **Symbolic program families.** Object correspondence, graph anti-unification,
   role/effect equations, scene graphs, cellular rules, row/column transducers,
   motif panels, and bounded evolutionary/CEGIS search were formalized. Several
   recovered small proposal families, but repeated audits showed that a fixed
   DSL or renderer has a hard coverage ceiling. These are specialist proposal
   lanes, not a universal replacement for the neural baseline.

4. **Neural/recursive search as proposal generation.** TTT, TRM-style recursive
   refinement, masked/soft-masked whole-grid correction, augmentation views,
   stochastic basin exploration, and holistic judging were treated as candidate
   generators. The main theoretical requirement is to preserve provenance and
   correlation metadata, collapse duplicate semantic classes, and judge complete
   output vectors rather than correlated raw beams.

5. **Equivariance and quotienting.** D4/group views, Reynolds projection, task-
   local query equivariance, color-role quotienting, object-ID quotienting, and
   behavioral partitioning were derived as variance-reduction tools. They are
   useful only when the transformation commutes with the task family; symmetry
   averaging can erase the correct asymmetric hypothesis.

6. **Constraints and decoding.** A shape constraint is the strongest audited
   structural gate. The embedded paranoid size predictor fired on 882/882
   training cases and 109/109 evaluation cases when it fired. An allowlisted
   shape family reached 727/727 training and 119/119 evaluation containment,
   but its additional runtime savings were small. Exact palette masking failed
   cross-fold soundness, so palette support must remain soft or branch-based.
   Shape-only prefix grammar is syntactically safe for a validated shape and
   tokenizer, while palette-aware hard masks are not.

7. **Long-grid probability theory.** An absolute `-log(.2)` path threshold is
   catastrophically strict for long outputs: it requires mean token probability
   about 0.995984 at length 400 and 0.998213 at length 900. The safer design is
   an incumbent-preserving union frontier: accept the absolute path or a
   calibrated length-normalized path, with conformal finite-sample thresholds,
   shape/view buckets, and a fallback to the baseline beam.

8. **Compute allocation.** The 12-hour/four-L4 run should maximize expected
   unique correct output classes per second. The derived controller allocates
   by marginal unresolved output value divided by calibrated cost, accounts for
   multi-test tasks position-wise, keeps a deadline reserve, and avoids spending
   all budget on correlated replicas. Pure cheap-first was value-blind; pure
   value-first was unstable in the geometry proxy.

9. **Cost calibration.** A training-only area/demo-count calibration of unknown
   test-output serialization cost improved public-evaluation task-cost MAE
   625.43→81.61 and Pearson rank correlation 0.9916→0.9977 versus the input-only
   proxy. The corrected hidden-geometry stress replay (240 tasks) completed
   192/207/219/229/237 tasks at 55/65/75/85/95% calibrated capacity versus
   179/193/206/217/226 for raw visible-cost ordering. This is a scheduling
   signal, not a score result; real GPU wall-clock and candidate-recall gates
   are still required.

10. **Competition/release gates.** The current target was re-anchored to the
    live official rules/leaderboard rather than the stale historical 34.44
    cutoff. Nemotron/OpenMDW licensing remained unresolved and is not a safe
    winner-path dependency. The current notebook is a 14-cell hybrid with size
    cap, cheap-first, Leg-C, and diverse attempt-2 active; grammar, provenance,
    length-frontier, and calibrated-value scheduling remain shadows.

## Current release recommendation

Keep the current notebook as the control. Do not promote a new lane based only
on training fit, fired-count, token-slot savings, or a public-fold score proxy.
The safest next release candidate is:

`baseline neural search + monotone Leg-C merge + audited paranoid size cap +
shape-only grammar at validated shapes + provenance-aware class deduplication +
calibrated output-value queue + incumbent-preserving normalized length frontier`

Every added component must pass task-disjoint folds, frozen-cache unique-class
recall, pass@2 non-decrease, exact tokenizer/schema validation, and a real
four-worker wall-clock replay. Keep a complete baseline fallback for every
abstention, conflict, unknown shape, malformed program, and licensing concern.

## Resume protocol

1. Read the end of `docs/AUTORESEARCH_LOG_2026-09-01.md` and the last ledger rows.
2. Obtain or build a frozen candidate cache; the workspace currently contains
   no checked-in neural logits/beam cache.
3. Run one controlled ablation at a time against the actual 14-cell hybrid:
   shape-only grammar → normalized length frontier → provenance/class quotient →
   calibrated value/cost scheduler.
4. Measure unique correct output classes, official output-position score,
   pass@2 coverage, p90 wall-clock, and private-shift robustness. Do not infer a
   top-eight claim from proxy counts.
5. Only after a release gate passes should the notebook be edited or a Kaggle
   submission be considered. Preserve the two-final portfolio as a distinct
   final-stage decision.

## Canonical artifacts

- `docs/AUTORESEARCH_LOG_2026-09-01.md` — full research chronology and proofs.
- `experiments/LEDGER.md` — compact experiment/result/decision ledger.
- `kaggle_notebook/BASELINE_ANALYSIS.md` — current notebook inventory and drift.
- `kaggle_notebook/notebook.ipynb` — scored artifact; unchanged by this research
  loop.
- `experiments/` and `tests/` — CPU-checkable hypotheses and contracts; synced
  `sources/` remains read-only.
