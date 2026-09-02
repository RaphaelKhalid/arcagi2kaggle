# ARC-AGI-2 Kaggle Notebook Handoff TODO

Purpose: give another orchestrator a release-oriented implementation sequence
for the overnight autoresearch findings. The control notebook is
`kaggle_notebook/notebook.ipynb`; reference material under `sources/` is read-only.

## Non-negotiable operating constraints

- Four Nvidia L4 GPUs, one 12-hour competition run, one scored submission per day.
- Up to two final submissions are allowed; each must be a complete valid file.
- Keep a 10-minute deadline reserve for merge, validation, and writing.
- No hidden labels, external APIs, credentials on disk, or license-ineligible
  dependencies.
- Every change is measured against the current 14-cell hybrid control, not the
  historical nine-cell description.

## Phase 0 — Freeze and instrument the control

- [ ] Copy the current notebook to a versioned working artifact without changing
  `SIZE_CAP_TOKENS`, `CHEAP_FIRST_ORDER`, `LEGC_ENABLED`, or diverse attempt-2.
- [ ] Run a schema-only dry run on the official local evaluation fold.
- [ ] Record per-task/test-position: candidate count, exact output hashes, chosen
  attempt, lineage, view, wall-clock, GPU, and failure/abstention reason.
- [ ] Add deterministic seeds and a run manifest. Never compare runs without the
  same seed manifest and candidate budget.
- [ ] Confirm `sources/` remains untouched and no credentials are serialized.

## Phase 1 — Build the frozen candidate cache (the current blocker)

- [ ] Run the control once on the local evaluation fold and persist only
  candidate grids plus non-sensitive provenance and scores; do not persist model
  credentials or private data.
- [ ] Store one record per task/test position with: output grid, canonical hash,
  raw score, semantic-class key, generator family, model/lineage ID, augmentation
  view, shape source, proof status, and elapsed time.
- [ ] Verify cache replay reproduces the control's exact selected outputs and
  official output-position score.
- [ ] Add a cache integrity test: duplicate raw beams from one lineage must not
  count as independent discoveries.

## Phase 2 — Shape-only grammar, fail closed

- [ ] At the existing size-cap seam, derive `H×W` only from the audited paranoid
  predictor or monotone primary/fallback contract.
- [ ] Apply grammar constraints only to syntax: digit tokens at cell positions,
  newline at row boundaries, and EOS after the final cell.
- [ ] Never hard-mask colors from the palette predictor; retain all digit support.
- [ ] Transform the shape through every D4/view operation before decoding.
- [ ] Compare cap-only vs shape-only grammar on the frozen cache with identical
  seeds and candidates.
- [ ] Promotion gate: no loss in incumbent recall, no increase in hard-invalid
  outputs, and positive unique-class gain per second. Otherwise retain as shadow.

## Phase 3 — Long-grid normalized likelihood frontier

- [ ] Preserve the current absolute likelihood branch as the incumbent.
- [ ] Add a length-normalized branch using mean legal-token NLL within the
  validated grammar; do not compare conditional grammar scores across unknown
  shapes without a calibrated shape prior.
- [ ] Fit thresholds task-disjointly, stratified by output area, view family, and
  shape source. Use the finite-sample conformal order statistic; underfilled
  buckets fall back to the absolute branch.
- [ ] Test `absolute OR normalized` against absolute-only on long-grid buckets.
- [ ] Promotion gate: long-grid recall and unique correct classes must improve at
  equal wall-clock, with no short-grid regression.

## Phase 4 — Provenance-aware semantic quotient

- [ ] Canonicalize exact output grids into semantic classes before ranking.
- [ ] Preserve generator, seed, augmentation, model, and refinement provenance.
- [ ] Collapse duplicate candidates from the same lineage for confidence counts.
- [ ] Calibrate class-level success on held-out folds; do not invent priors from
  the target evaluation labels.
- [ ] Add a correlation penalty or effective-sample correction for repeated views.
- [ ] Promotion gate: pass@2 and unique correct-class recovery must be no worse
  than the control; raw beam count is not an acceptance metric.

## Phase 5 — Calibrated output-value scheduling

- [ ] Fit unknown test-output serialization cost from training challenge/solution
  pairs only, using area/demo-count buckets and pooled fallback.
- [ ] Override the estimate only when a validated shape cap exists.
- [ ] Estimate each test position's unresolved mass, novelty rate, selector
  recovery, and cost. Use marginal value/cost as the queue key.
- [ ] Schedule position work across four workers with deterministic tie-breaking,
  a deadline reserve, and no task-level skip when another test position remains
  unresolved.
- [ ] Compare raw cheap-first, calibrated cheap-first, and value/cost ordering in
  a real wall-clock replay. Report p50/p90 task time, completed positions, GPU
  utilization, candidate classes, and pass@2.
- [ ] Promotion gate: positive score recovery per second and no deadline/schema
  failures. The CPU geometry result is evidence, not a release decision.

## Phase 6 — Merge, two-final portfolio, and release validation

- [ ] Keep Leg-C verified programs monotonic: a malformed, conflicting, or
  position-incomplete record must return that position to the base solver.
- [ ] Merge in this order: verified proof, high-confidence independent agreement,
  calibrated class score, then baseline fallback.
- [ ] Construct attempt 2 from a different ranking family and ensure it differs
  from attempt 1; never delete baseline coverage on abstention.
- [ ] Produce two complete candidate submissions, not a per-position hybrid.
- [ ] Validate exact task/test keys, grid dimensions, colors, JSON schema, and
  deterministic reproducibility.
- [ ] Confirm all attached models, code, and datasets satisfy the competition
  winner license obligation before any scored run.
- [ ] Spend at most one official submission on the fully controlled release and
  log it in `experiments/LEDGER.md`.

## Required report for every ablation

Record: hypothesis; exact code seam; fold split; seed; candidate budget; output
positions; exact score; pass@1/pass@2; unique output classes; candidate recall;
hard-invalid count; p50/p90 wall-clock; GPU-hours; memory peak; deadline reserve;
license status; and a promote/hold/reject decision. A proxy improvement without
these fields is not evidence of progress toward the low-70s target.

## Current recommendation

The highest-confidence eventual stack is:

`baseline neural search + monotone Leg-C merge + audited paranoid size cap +
shape-only grammar + provenance-aware semantic deduplication + calibrated
output-value queue + incumbent-preserving length frontier`

Do not edit the scored notebook until Phases 1–5 pass their gates on frozen
candidates and a real four-worker replay.
