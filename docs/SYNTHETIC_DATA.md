# Synthetic Data Pipeline (design)

Goal: attack the model gap (public checkpoint vs. private leaders) with consensus-verified
synthetic tasks, NVARC-style, within a $50 OpenAI batch budget.

**Update (Aug 31 recon):** the full NVARC synthetic corpus is already public on Kaggle
(sorokin's datasets: synthetic 338 MB, augmented 1.3 GB, artifacts 42 GB) — and the public
checkpoint was trained on it, so retraining on the same data buys little. The $50 budget
therefore targets *marginal* data only: new consensus-verified tasks in the failure
categories our eval-set analysis identifies (especially compositional and symbol-semantics
families where ARC-AGI-2 concentrates), plus program-induction traces (task → verified
Python solution) to fine-tune a coder model for the evolution loop.

## Recipe (adapted from NVARC 2025, published method)

1. **Concept inventory** — extract elementary transformation descriptions from public
   sources (BARC/Human-ARC seed descriptions) plus our own failure-category analysis of
   the 120 evaluation tasks.
2. **Composition** — combine 2–3 elementary descriptions into composite puzzle specs
   (ARC-AGI-2 tasks are compositional; this is the distribution that matters).
3. **Two-stage generation** (cheap batch model, e.g. gpt-5-mini via Batch API):
   a. Generate a grid-generator program per spec (must produce ≥30 valid, diverse grids).
   b. Generate N=8–20 independent implementations of the transformation.
4. **Consensus verification (the key falsification step, costs $0)** — execute all
   implementations locally; keep the task only when ≥40% of implementations agree exactly
   on all outputs. Agreement = the spec is unambiguous and implementable ⇒ a valid task.
5. **Augment + train** — D8 transforms × color permutations; continued fine-tuning of the
   public Qwen3-4B checkpoint (LoRA on RTX 4060 for pilots, Kaggle/Colab GPU for full runs).

## Cost model

Programs are ~1–2k output tokens. At batch pricing, $50 ≈ tens of thousands of generated
programs ⇒ target ~5–15k verified tasks (NVARC's acceptance rate was ~40–50%). Pilot with
$5 before committing the rest.

## Spend ledger

| date | purpose | model | est. tokens | cost | cumulative |
|------|---------|-------|-------------|------|------------|
| — | (nothing spent yet) | | | $0.00 | $0.00 |

Hard cap: $50. No API calls inside the submitted solver (offline-only rule and no internet
at inference anyway).
