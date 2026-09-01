# Hidden/Unconventional Research Sweep — ARC Prize 2026 (ARC-AGI-2)

Date: 2026-08-31. Context: we run the LB 33.89 Qwen3-4B grid-tokenizer + per-task LoRA TTT + DFS decoding + augmentation-consistency lineage. Prize cutoff (#8) = 34.44. This report catalogs what we had NOT yet cataloged.

Evidence labels: [VERIFIED] = leaderboard/paper receipts; [PARTIAL] = credible writeup, not independently reproduced; [MARKETING] = self-reported, treat skeptically.

---

## 1. Actionable-now techniques (shippable in days)

### 1.1 PTRM — Probabilistic Tiny Recursive Model (inference-only TRM boost)
- Paper: https://arxiv.org/html/2605.19943v1 (May 2026)
- What: **No retraining.** Inject Gaussian noise (scaled sigma) into TRM latent state each recursion step -> K parallel stochastic rollouts; reuse TRM's existing Q/halt head to select the best candidate.
- Numbers [PARTIAL, paper]: ARC-AGI-2 pass@1 8.47% vs 7.36% deterministic; pass@2 9.72%; pass@100 15.97% vs 14.31%.
- Fit: applies directly to our cpmpml/arc-prize-trm-031 TRM ensemble leg. Effort: ~1-2 days (inference loop change + candidate selection). Expected: +1 to +1.5 pts on the TRM leg's contribution; more if combined with our augmentation-consistency scorer as an external verifier.
- No code released — but the change is ~50 lines on the TRM inference path.

### 1.2 NVARC's post-competition TRM finding: pass@128 is where TRM's ceiling actually is
- Source: Trelis interview writeup https://trelis.substack.com/p/nvarc-2025-arc-prize-winners [PARTIAL]
- Extended TRM training moved 10% -> 18% pass@1, and **~30% pass@128** on ARC-AGI-2. Meaning: TRM checkpoints already *generate* correct answers for ~30% of tasks; the bottleneck is selection/reranking, not generation.
- Actionable: sample TRM widely (cheap — 7M params, trivial on L4s), then rerank candidates with our existing Qwen3-4B TTT model's log-likelihood / augmentation-consistency scorer (cross-model verification). NVARC themselves suggest "reordering generations through RL could prove valuable."
- Effort: 2-4 days. Expected: this is plausibly the single largest untapped ensemble gain — the DFS-LLM leg and TRM leg solve partially disjoint task sets.

### 1.3 NVARC full open repo (their exact 24.03% recipe, same hardware constraints as ours)
- Code: https://github.com/1ytic/NVARC (Ivan Sorokin + Jean-Francois Puget, NVIDIA KGMoN). [VERIFIED — 1st place, 24.03% private, $0.20/task]
- Contains: ARChitects-style Qwen3-4B fine-tune w/ **Unsloth Flash LoRA** hyperparameters, improved TRM training scripts, submission notebooks, and the **SDG synthetic-data-generation pipeline (scripts + prompts)** — the generator, not just the data.
- Recipe details (Trelis writeup): 16-token reduced tokenizer + patched embedding tables; augmentation levels tuned per source (256x for existing datasets, 24-32x for synthetic); ~700 seed puzzles (Human-ARC + BARC) -> Claude/GPT-4o structured 5-part descriptions -> GPT-OSS mixes description pairs -> 260k candidates -> two-stage code-based verification (>=30 valid input grids by generated generator code; 20 independent transformation implementations, keep if >=8/20 agree) -> ~103k verified puzzles.
- Official submission notebook: https://www.kaggle.com/code/gregkamradt/arc2-qwen3-unsloth-flash-lora-batch8-queue-trm2/
- Actionable: (a) diff their Unsloth Flash LoRA + queueing config against our perfpatch setup for L4 throughput wins; (b) their TRM2 queue integration is a working template for our TRM ensemble; (c) rerun their SDG with a current open model (see 2.1).
- Effort: days (a/b) to weeks (c).

### 1.4 SOAR — drop-in upgrade for the "Leg C" program-induction leg
- Paper: https://arxiv.org/abs/2507.14172 (ICML 2025, ARC Prize 2025 paper award #2). Code: https://github.com/flowersteam/SOAR (MIT). [VERIFIED paper; models/dataset public]
- Released fine-tuned models (Apache-2.0): julien31/Soar-qwen-7b (**base = Qwen2.5-Coder-7B-Instruct — same base as our Leg C**), Soar-qwen-14b/32b/72b, Soar-mistral-123b. https://huggingface.co/collections/julien31/soar-arc-6856d27681fce01d9af4c4a3
- Method: evolutionary search (sample + refine programs with execution feedback) alternating with hindsight-relabeling fine-tuning on its own search traces. 52% ARC-AGI-1 public test with open weights only.
- Actionable: swap Soar-qwen-7b in for stock Qwen2.5-Coder-7B in Leg C — it's already specialized for ARC program sampling AND refinement (repair-from-failed-execution), which our sandbox loop can exploit. The refinement operator (feed back wrong outputs, ask for patch) runs offline.
- Effort: ~2-3 days (model swap + prompt format alignment). Expected: material lift on the induction leg; SOAR showed refinement-tuned models sharply beat base at equal samples. Caveat: trained on ARC-AGI-1 style tasks; ARC-AGI-2 transfer unproven [PARTIAL].

### 1.5 ARChitects 2025 (2nd place) — open code, and three steal-able components
- Tech report: https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/ ; code: https://github.com/LambdaLabsML/ARC2025_Solution_by_the_ARChitects [VERIFIED — 16.53%/21.67% LB era]
- Base: LLaDA-8B masked-diffusion LM; "Golden Gate RoPE" multi-directional 2D positional encoding; grids padded to 32x32 then padding tokens removed with positional IDs preserved.
- Key inference trick: **soft-masking recursive refinement** — token embeddings treated as continuous ("0.5*A + 0.5*B"), add the <mask> embedding to every position each step, iterate 102 steps (2x51 with cold restart) without discretizing. Candidate selection = most-visited-candidate counting across the refinement trajectory.
- Auxiliary: separate **shape-prediction model** (predict output grid dims by mask-token delimiter placement; 85%+-2 accuracy). With known shape: 30.5%+-1 on eval.
- TTT: 128 steps/task, rank-32 LoRA, random augmentation per step (pretrain: 175k steps rank-512 LoRA on ReARC, ARC-GEN-100K, ARC-Heavy, ConceptARC, official sets).
- Actionable now even without adopting diffusion: (a) most-visited-candidate counting as an additional vote signal; (b) a dedicated shape predictor to prune our DFS (shape errors are a known DFS failure mode); (c) their eval-overfit warning (estimated 26% -> scored 21.67%) — maintain a held-out split.
- Full diffusion leg: an 8B masked-diffusion + TTT fit their 2025 Kaggle budget, so 4xL4/12h is feasible but tight; effort weeks. Ensemble diversity is the draw — its solve set is very different from autoregressive DFS.

### 1.6 GPT-OSS-120B as an offline "frontier" leg + Poetiq-style refinement loop
- Poetiq (current ARC-AGI-2 semi-private SOTA ~54% with Gemini3/GPT-5.1 mixes) open-sourced their loop under MIT: https://github.com/poetiq-ai/poetiq-arc-agi-solver ; blog: https://poetiq.ai/posts/arcagi_announcement/ [VERIFIED for API results; offline transfer PARTIAL]
- The loop is simple and model-agnostic: generate solution (often code) -> execute -> feed back failures -> refine; self-auditing early termination (<2 requests/task avg). They explicitly report it generalizes across 12 LLM families **including GPT-OSS-120B** (open weights, Apache-2.0).
- Unconventional play: GPT-OSS-120B is MoE (~5.1B active, MXFP4 ~63GB) — it can fit in 96GB across 4x L4 with vLLM. A Poetiq-style critique/refine loop over it, offline, on the subset of tasks our TTT model is least confident about, is a legitimate third leg nobody on the public LB appears to run.
- Effort: 1-2 weeks (vLLM MoE on L4s + loop + time budgeting). Risk: L4 throughput; must strictly cap per-task time. Expected: unknown but potentially large — frontier-style code-refinement is what separates 30% systems from 50% systems in every 2026 result.

### 1.7 Verantyx neighborhood-rule op — free CPU adjunct
- https://huggingface.co/kofdai/Verantyx-arc-agi2-7.4 ; code https://github.com/Ag3497120/verantyx-arc-agi2 (MIT). [PARTIAL — 7.4% on the *training* split, self-reported]
- Mostly weak, but one op is notable: **neighborhood_rule** — learn an exact 3x3/5x5 neighborhood->output-value lookup table, CEGIS-verified consistent across all training pairs. It alone solved 240/1000 training tasks. ~0.4s/task, pure Python, zero deps.
- Actionable: run as a zero-GPU-cost pre-pass; any task it solves exactly (verified on all train pairs) frees DFS/TTT budget for hard tasks, and it can supply a high-precision candidate for voting. Effort: <1 day.

### 1.8 ABPR — LLM-driven algorithmic debugging over Prolog traces (idea bank)
- Paper: https://arxiv.org/html/2603.20334v1 (2026). [PARTIAL — no code released]
- Prolog meta-interpreter produces declarative execution traces; LLM oracle localizes buggy clauses in the proof tree; iterative abductive refinement. Gemini-3-Flash: 34.03% -> **56.67% pass@2** on (a 120-task slice of) ARC-AGI-2. Ablation: removing declarative traces costs >10 pts.
- Transferable idea for Leg C without Prolog: structured *execution-diff* feedback (which cells wrong, which sub-step diverged) beats conversational retry. Cheap to add to our sandbox loop: report per-example cell diffs + intermediate variable dumps back into the refine prompt. Effort for the lightweight version: 2-3 days.

---

## 2. Training-track resources (offline SFT / pretraining)

| Resource | What | Where | License / notes |
|---|---|---|---|
| NVARC SDG pipeline | The generator code + prompts (not just data): seed->describe->mix->verify | https://github.com/1ytic/NVARC (SDG folder) | Winner-release (open per prize rules). Rerun with GPT-OSS locally to mint more/harder puzzles. |
| NVARC Synthetic Puzzles (103k) + Augmented (3.2M) + Artifacts | Verified synthetic corpora behind the 24.03% winner | Kaggle datasets linked from the NVARC repo | [VERIFIED provenance] |
| soar_arc_train_5M | 5M deduped ARC program solutions (code + hindsight-relabeled synthetic tasks) | https://huggingface.co/datasets/julien31/soar_arc_train_5M | For SFT of induction models; MIT/Apache ecosystem |
| SOAR fine-tuned models | Soar-qwen-7b/14b/32b/72b, Soar-mistral-123b | https://huggingface.co/collections/julien31/soar-arc-6856d27681fce01d9af4c4a3 | Apache-2.0; 7b/14b run on L4s |
| ARC-GEN-100K | Google's mimetic procedural generator for all 400 ARC-1 train tasks + 100k dataset | Paper https://arxiv.org/abs/2511.00162 | Open-source generator; used by ARChitects |
| ARC-Heavy | ~200k BARC-style synthetic tasks | https://huggingface.co/datasets/Slightwind/ARC-Heavy-Tasks ; https://huggingface.co/datasets/CohenQu/ARC-AGI-Transduction-Heavy | BARC lineage |
| ReARC generator | Procedural example generators for all 400 train tasks (Hodel) | Paper https://arxiv.org/abs/2404.07353 | Foundation of every winner's aug stack |
| Trelis TRM assets | TRM-ARC-AGI-II checkpoints (~10.5% after 48h on 4xH200, Apache-2.0) + TTA paper + fork of TRM code + ARC-AGI-2 reasoning SFT'd Qwen3-4B models | https://huggingface.co/Trelis/TRM-ARC-AGI-II ; https://arxiv.org/pdf/2511.02886 ; https://github.com/TrelisResearch/TinyRecursiveModels ; https://hf.co/Trelis/Qwen3-4B_ds-arc-agi-2-reasoning-5-c178 | Extra TRM seeds to diversify the cpmpml ensemble; config in README (note L_cycles=4 accidental deviation) |
| TRM Mamba-2 hybrid | Swap transformer blocks for Mamba-2 hybrid: +2.0 pass@2, +4.75 pass@100 on ARC-AGI-1 (better candidate coverage) | https://arxiv.org/abs/2602.12078 (ICLR26 workshop) | Retrain required; pairs well with 1.2 reranking |
| TRM analysis | TRM saturates after few latent updates; puzzle-ID embedding is load-bearing; 1000-sample voting = +11 pts pass@1 | https://arxiv.org/abs/2512.11847 | Guides where not to spend TRM compute |
| VARC (CVPR 2026) | ARC as image-to-image: canvas + vanilla ViT (~18M) + aggressive TTT; 54.5 (single) / 60.4 (ensemble) ARC-1, **8.3 ARC-2** from scratch | https://arxiv.org/abs/2511.14761 ; community impl https://github.com/kyegomez/VARC | Cheap diversity leg; official code via project page |
| TraceViT | Ordered intermediate-transformation trace supervision for looped visual reasoning on ARC | https://arxiv.org/abs/2607.29586 (Jul 2026) | Data recipe idea: supervise intermediate steps, not just final grids |
| LongT5 125-token encoding report | 59-page ARC-AGI-2 report: compact 125-token task encoding, group-symmetry + grid-traversal + automata-perturbation augmentations, symmetry-aware decoding | https://arxiv.org/abs/2603.06590 | CC BY-NC-ND (no code found); augmentation taxonomy worth mining |
| ARC Prize 2025 tech report | Canonical winners' details | https://arxiv.org/html/2601.10904v1 | — |
| Living survey (updated 2026) | Full 2026 landscape incl. Land ensemble 72.9% ARC-2 ($38.90/task, API-only), refinement-is-intelligence thesis | https://arxiv.org/html/2603.13372v1 | — |
| MindsAI/Tufa writeup | TTFT + AIRV + tokenizer dropout + pretraining tricks (12.64%) | https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution ; https://arxiv.org/abs/2506.14276 ; code notebook https://www.kaggle.com/code/gregkamradt/mindsai-tufa-2025-v4/ | Writeup page is JS-only (couldn't scrape); the notebook is the ground truth for tokenizer dropout implementation |
| dt-lindberg/asp-arc-agi-data | LoRA on nvidia/Nemotron-Cascade-2-30B-A3B tagged answer-set-programming/clingo (Aug 2026) | https://hf.co/dt-lindberg/asp-arc-agi-data | Undocumented; curiosity only — no card, no scores |

---

## 3. Dead ends checked (do not re-research)

1. **"Why Test-Time Training Plateaus" / anvithpothula/arc-agi2-ttt-analysis**: no public writeup, paper, or blog found under that name or author (multiple searches). Kaggle dataset page is JS-rendered and unscrapable here. Nearest substantive equivalents: (a) https://arxiv.org/abs/2512.11847 — TRM/TTT gains concentrate in the first refinement steps and in vote/ensemble selection, not deeper recursion; (b) https://arxiv.org/abs/2507.15877 — TTFT fails on compositional OOD where execution-guided program synthesis composes; (c) 2601.10904's finding that no static transduction system exceeds ~11% without TTT. Consistent story: the plateau is a *selection and compositionality* problem — more TTT steps don't help; verified search and better candidate reranking do. Someone with Kaggle access should still open the dataset (cheap check).
2. **athanor (dastin359) "95.7% public eval"**: MIT, real traces published, but 100% API-dependent (Claude/GPT/Gemini), and the score is on the *public* eval set with contamination risk acknowledged. Not runnable offline. Portable ideas only: artifact-only cross-context review (fresh-context reviewer sees hypothesis+code+predictions, not reasoning), code-as-verification. https://github.com/dastin359/athanor [MARKETING-adjacent]
3. **epang080516/arc_agi (26.0% ARC-2)**: method = LLM program synthesis + persistent cross-task program library (538 programs, wake-sleep, score-weighted softmax sampling of library exemplars); primary model Grok-4, ~10 calls/task, API-only. No license stated. The *library-in-prompt* idea could seed Leg C, but the repo is not offline-usable as-is. Blog: https://ctpang.substack.com/p/arc-agi-2-sota-efficient-evolutionary
4. **poetiq-arc-agi-solver as-is**: MIT but requires GEMINI/OPENAI keys; only the loop design transfers (see 1.6).
5. **mousberg/arc-agi-2**: just a mirror of the official task data. Nothing else.
6. **GitHub topics arc-agi-2 / arc-prize**: swept. Only notable non-cataloged items were athanor (above), verantyx (1.7), arc-explainer (TS visualization tool, no method), PUMA/Latent-Inference-Manifold (tiny, no receipts), aurascoper CPU modules (below). Topic pages are dominated by ARC-AGI-3 agent repos (different competition).
7. **aurascoper/arc-agi-2-cpu-kaggle-modules**: honest negative result — CPU symbolic engine scored **0.00 on hidden test** despite public-eval development wins. Confirms: public-eval-tuned symbolic engines don't transfer; only CEGIS-verified-on-train-pairs candidates (like 1.7) are safe.
8. **ASP/clingo for ARC**: no substantive 2026 paper/repo found beyond the undocumented dt-lindberg adapter.
9. **Kaggle discussion forum scraping**: all kaggle.com pages (discussions, writeups, datasets) render via JS — WebFetch gets titles only. Site-search surfaced no substantive 2026 ARC-AGI-2 technique threads indexed by Google yet. The 2025 writeups + winner notebooks (linked above) are the accessible ground truth.
10. **ARChitects raw README fetch**: raw.githubusercontent redirect returned the Product-of-Experts repo; use the tech-report page (linked in 1.5) instead, which is complete.
11. **Land 72.9% ARC-AGI-2 ensemble** (survey): three frontier APIs + sandbox + evaluator ranking, $38.90/task. Not offline-relevant; likely the archetype of the 72.08 public LB leader's approach class, but no Kaggle-legal path.

---

## Key URLs (flat list)
- https://github.com/1ytic/NVARC — winner code + SDG generator
- https://trelis.substack.com/p/nvarc-2025-arc-prize-winners — NVARC recipe details
- https://github.com/flowersteam/SOAR + https://huggingface.co/julien31/Soar-qwen-7b + https://huggingface.co/datasets/julien31/soar_arc_train_5M
- https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/ + https://github.com/LambdaLabsML/ARC2025_Solution_by_the_ARChitects
- https://arxiv.org/html/2605.19943v1 (PTRM), https://arxiv.org/abs/2602.12078 (Mamba-TRM), https://arxiv.org/abs/2512.11847 (TRM analysis), https://arxiv.org/pdf/2511.02886 (Trelis TRM TTA)
- https://arxiv.org/html/2603.20334v1 (ABPR), https://arxiv.org/abs/2511.14761 (VARC), https://arxiv.org/abs/2511.00162 (ARC-GEN)
- https://arxiv.org/html/2601.10904v1 (2025 report), https://arxiv.org/html/2603.13372v1 (living survey)
- https://github.com/poetiq-ai/poetiq-arc-agi-solver + https://poetiq.ai/posts/arcagi_announcement/
- https://www.kaggle.com/code/gregkamradt/arc2-qwen3-unsloth-flash-lora-batch8-queue-trm2/ (NVARC submission), https://www.kaggle.com/code/gregkamradt/mindsai-tufa-2025-v4/ (MindsAI), https://www.kaggle.com/code/gregkamradt/arc-2025-diffusion/ (ARChitects)
