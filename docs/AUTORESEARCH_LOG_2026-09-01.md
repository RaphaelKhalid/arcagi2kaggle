# ARC-AGI-2 Autoresearch Log — 2026-09-01

This is a theory-first autoresearch record. The objective is to discover a
Kaggle-legal route from the current public NVARC-lineage baseline into the
low-70% regime, without pretending that a conceptual result is an empirical
leaderboard result. No finetuning is being run in this loop; experiments are
CPU-side audits, formal derivations, design reviews, and cheap falsification
tests. `sources/` remains read-only.

## Operating constraints verified

Checked against the official competition page and rules on 2026-09-01:

- one submission per day; up to two final submissions;
- submission through a self-contained notebook, with no internet access;
- GPU notebook runtime no longer than 12 hours;
- exactly two predictions (`attempt_1`, `attempt_2`) for every test output;
- score is pass@2: an output is correct when either attempt exactly matches;
- L4x4 is available for this competition, with 96 GB pooled GPU memory;
- public external data and pretrained models are allowed, but source/code and
  licensing obligations must be checked before a prize-eligible submission;
- all task IDs and both attempts must be present in `submission.json`.

Primary rule page: <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules>
and overview: <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/>.

The repository's own recon (2026-08-31) reports a public leaderboard head of
72.08 and 70.42, with a 33.89 public baseline pack. Those figures are treated
as a campaign snapshot, not as a permanent leaderboard fact.

## Iteration 0 — baseline audit and problem statement

### Known starting point

The current stack is the public NVARC lineage: Qwen3-4B with a custom tiny
grid tokenizer, per-task LoRA test-time training, augmentation, DFS decoding,
and likelihood/vote-based selection. The perfpatch preserves the baseline's
beam ordering while moving vocabulary normalization and target-token gathering
onto the GPU. The repository has not yet recorded a clean scored submission.

### Local falsification measurement

Command: `python -m symbolic.measure --preset strict --failures 12` using the
bundled workspace Python runtime.

| evaluator | public training | public evaluation |
|---|---:|---:|
| exact shape predictor: coverage | 83.8% | 64.5% |
| exact shape predictor: precision | 99.89% | 99.10% |
| exact palette predictor: coverage | 82.5% | 68.0% |
| exact palette predictor: precision | 99.55% | 99.15% |
| palette superset bound: containment | 99.91% | 100.00% |

The evaluation split contains 120 tasks and 172 test outputs. The rule battery
is therefore valuable as a high-precision constraint and candidate validator,
not as a content solver. It has one observed shape false positive and one
observed exact-palette false positive on the public evaluation set.

### Immediate theoretical conclusion

The official metric changes the optimization target. Let `D` be the train
demos, `t` the test input, and `H(D)` the hypotheses consistent with all demos.
Each candidate hypothesis `h` induces an output `z_h(t)`. For two submitted
outputs `a,b`, the Bayes objective is

    P(pass@2 | D,t,a,b) = P(z(t)=a or z(t)=b | D,t).

If the posterior over exact output classes were known and calibrated, the
optimal pair would be the two highest-mass distinct classes: the second slot
maximizes residual posterior mass after the first is chosen. The role of
diversity is earlier and more subtle: independent families help discover
minority classes and make posterior calibration less correlated. Family
diversity should therefore shape proposal generation and evidence correction;
it must not override a genuinely calibrated top-two posterior.

## Iteration 1 — the 72% mechanism, translated to offline Kaggle

The strongest new external evidence is Johan Land's 72.9% semi-private ARC-2
solver. Its claimed mechanism is independent candidate generation across text,
image, and code modalities followed by a single-context holistic judge, with a
reported gain from recovering correct minority hypotheses. This is not directly
Kaggle-runnable because its main models/API are external, but it reveals the
architecture that a 70% system must approximate: many proposals, preserved
traces, and a joint selector rather than a single autoregressive confidence.

Paper: <https://arxiv.org/abs/2606.31543>
Released implementation: <https://github.com/beetree/ARC-AGI>

The offline translation is an **open-weight council**:

1. NVARC/Qwen3-4B remains the dense transduction/TTT family.
2. A verified program-induction family proposes executable transformations.
3. A TRM family produces many cheap recursive candidates.
4. Optional visual/object and neighborhood families add genuinely different
   hypotheses, not copies of the same text prompt.
5. A compact attached judge or deterministic proof scorer ranks the full set of
   candidates jointly, retaining the best two output equivalence classes.

The candidate artifact should be `{program/output, demo proof, invariants,
shape, palette, object trace, confidence}`, not an unstructured explanation.

## Iteration 2 — a stronger mathematical target than “better guessing”

### Finite version-space theorem

Let `H_L` be a finite, executable DSL grammar truncated at description length
`L`, and let

    V_L(D) = {h in H_L : h(x_i) = y_i for all demos i}.

If `V_L(D)` is nonempty and every `h` in `V_L(D)` produces the same output on
the test input `t`, then that output is correct for every ground-truth rule in
`H_L` consistent with the demonstrations. The proof is direct: the unknown
ground-truth rule belongs to `V_L(D)`, and all members agree on `t`.

This gives an actual solver certificate: **DSL completeness is the only
assumption; selection disappears when the version space collapses on the test
input.** When the version space does not collapse, the two-attempt optimum is
the two output equivalence classes with maximum posterior mass. The system
should explicitly report which case it is in, rather than hiding ambiguity
behind a confidence score.

### Exact synthesis engine implied by the theorem

The practical implementation is not blind brute force. It is dynamic-programming
enumeration over typed terms with e-graph-style canonicalization:

1. enumerate short typed terms from object extraction, selectors, relations,
   geometry, recoloring, masking, and composition;
2. execute each term on all demos and memoize its behavior signature;
3. merge terms with identical demo behavior, retaining the shortest proof;
4. expand only signatures that have not yet produced a test-output consensus;
5. stop with a certificate when all surviving behaviors agree on the test;
6. if the version space is ambiguous, pass its output classes plus proofs to the
   learned judge and reserve attempt 2 for the strongest distinct class.

This separates *expressivity* from *search*: a zero valid transition means the
grammar lacks a primitive, while many valid transitions with competing test
outputs means the grammar or prior is underconstrained. The distinction is
diagnostically important.

### DSL coverage is the hard theoretical bottleneck

The structural-descriptor paper reports that its primitive library generated no
valid transition from the input on 229 of 400 evaluation tasks, invariant to
search budget. This is evidence against spending the entire 12-hour window on
deeper search inside a fixed DSL. The correct response is grammar expansion by
typed neural proposals or a new object representation, followed by exact
verification—not an even larger beam over the same operators.

This yields a four-state router:

    zero valid term       -> expand representation / neural code family
    one consensus class   -> issue proof-carrying candidate
    many classes, strong   -> use MDL + invariants + joint judge
    many classes, weak     -> diversify modality and spend no more same-family TTT

## Candidate-cache specification for the next replay

Every proposal should be stored as a normalized record, independent of the
model that generated it:

    task_id, test_index, family, seed, output_grid, output_hash,
    demo_exact, grid_valid, shape, palette, palette_bound_ok,
    mdl_length, proposal_logprob, family_logprob,
    invariants, object_trace, execution_trace, wall_seconds.

Selection should first reject `grid_valid == false`, failed demo verification,
and hard shape/palette contradictions. It should then aggregate by exact
`output_hash`, not by program text. To correct for correlated samples from one
model, estimate family evidence with an effective sample size

    n_eff = (sum_j w_j)^2 / sum_j(w_j^2)

and use a Dirichlet-smoothed family mixture rather than raw vote counts. For an
output class `z`,

    P_hat(z) = sum_f P(f|D) P_hat(z|f,D).

The selected pair maximizes `P_hat(a) + P_hat(b)` over distinct output classes.
If the pair is tied, prefer the pair with more distinct proof families and
lower correlated error risk. This is the exact pass@2 objective under the
estimated posterior, and it is replayable without rerunning the models.

## Updated queue after the theorem

| priority | experiment | cheap success criterion |
|---:|---|---|
| 1 | replay class-posterior pass@2 selection on any saved candidate cache | beats current probmul top-2 at fixed candidates |
| 2 | measure zero-transition rate of the current Leg C DSL | router identifies representation gaps, not just timeouts |
| 3 | add typed object/relational terms with behavior-signature deduplication | more verified output classes per second, no precision loss |
| 4 | structural mid-trajectory gate | at least 20% compute saved with no solve loss on dev folds |
| 5 | TRM as high-volume proposal generator, NVARC as verifier | positive marginal coverage on NVARC misses |
| 6 | offline joint judge, only after licensing review | order-stable and better than majority/probmul replay |

The next concrete loop action remains a replayable selector. It is the safest
high-value experiment because it can produce a measurable gain without model
loading, Kaggle quota, or a leaderboard submission.

## Iteration 3 — selector prototype and constraint-value estimate

### Prototype result

Added `experiments/pass2_selector.py`, a model-agnostic CPU prototype. Its
self-test constructs ten correlated dense-model votes for `wrong` and one
proposal each from `program` (`correct`) and `recursive` (`alternate`). With
explicit family priors, it selects `wrong` and `correct`, retaining the
minority family in the pass@2 set while not allowing ten correlated samples to
count as ten independent families. The repository's 45 tests still pass.

The first equal-prior version selected by arbitrary hash tie-break, which was a
useful negative result: family priors and tie policy must be calibrated on held
out tasks, not left implicit.

### Constraint-value estimate

On the 172 public-evaluation test outputs, the paranoid shape battery produced
an exact dimension prediction on 63.4% of outputs. The empirical palette
superset bound averaged 6.22 allowed colors per output, versus 10 possible ARC
colors. Treating cell values as independent only for a rough scale estimate,
this removes about 327 bits of color-choice entropy per output on average for
which the bound applies. This is not a claim that cells are independent; it is
an indication that palette constraints can materially reduce the neural search
branching factor.

Important safety correction: `palette_superset_bound` is an empirical bound
from observed demo behavior, not a theorem for every possible ARC task. Exact
palette rules may be hard gates; the generic superset bound must remain a soft
ranking/pruning signal unless backed by an invariant or a typed program proof.

### New research input

Two complementary papers reinforce the same decomposition:

- execution-guided neural program synthesis outperforms test-time fine-tuning
  on controlled compositional OOD generalization, supporting a CEGIS/program
  leg for compositional ARC-2 rather than simply more TTT;
  <https://arxiv.org/abs/2507.15877>;
- compositional neuro-symbolic reasoning reports gains from object-level
  representations, neural transformation proposals, and cross-example
  consistency filtering, a close blueprint for the typed object algebra;
  <https://arxiv.org/abs/2604.02434>.

The loop's current best thesis is therefore:

    72% = broad, complementary proposal coverage
          × exact execution/falsification
          × calibrated class-level pass@2 selection
          × dynamic compute allocation.

More TTT on one proposal family attacks only one factor and is unlikely to
bridge the full gap from the 33.89 lineage to the low 70s.

## Iteration 4 — stability under demonstration deletion

### H8 — Leave-one-demo-out version-space stability (priority: high)

For a task with demonstrations `D = {d_1, ..., d_k}`, synthesize candidate
families from each reduced set `D \ {d_j}`. Let `C_j` be the set of test-output
classes produced by those candidates. Define the ambiguity/stability signal

    stability(D,t) = 1 - H(mean_j P(C_j)) / log |union_j C_j|,

with the zero-denominator case defined as stable. A candidate class recurring
under several demo deletions is more robust than one that exists only because a
single example pins down a brittle accidental detail.

This is not a substitute for correctness: ARC demonstrations are intentionally
small and a removed example may contain essential information. It is a
label-free ranking feature that can be combined with MDL and exact demo proof.
Its sharpest use is ambiguity routing: high stability plus one class suggests a
proof-like answer; low stability suggests spending attempt 2 on a different
class or invoking a different solver family.

**Cheap falsification.** On folds 0–3, compare stability-weighted selection to
MDL-only selection on the same candidate cache. Require improvement in pass@2
without increasing runtime by more than 5%. The shadow fold remains untouched
until a major milestone.

### H9 — Counterfactual invariant probes (priority: high)

When a program is verified on demos but multiple programs survive, generate
small *counterfactual* inputs by applying known nuisance transformations to the
demo inputs: D8 geometry, color renaming, translation within canvas, object
permutation, and background changes where legal. Do not use hidden labels. Score
each hypothesis by whether its behavior respects invariants implied by the
demonstrations and the chosen transformation group.

This turns free-form model confidence into an executable metamorphic test. It is
especially useful for detecting programs that memorise absolute coordinates or
specific color IDs when the demonstrations establish a relational rule.

**Failure mode.** ARC-AGI-2 includes contextual symbolic meaning, so not every
color or geometric permutation is legal. Each probe must carry a justification
from the task's observed invariants; unjustified probes are evidence only for a
soft penalty, never a hard rejection.

## Licensing gate update

The attached Nemotron Lightning model is distributed under OpenMDW-1.1. The
license grants broad rights, but the OpenMDW FAQ says version 1.1 was not yet on
the SPDX list, and the OSI review was still in progress when checked. Therefore
the model is a promising technical proposal generator, but it is not yet an
unconditional choice for a prize-eligible path whose code must use OSI-approved
licensing. Keep a permissively licensed fallback plan (Qwen/SOAR/TRM plus
MIT/Apache components) and treat model eligibility as a release checklist item.

References: <https://openmdw.ai/license/1-1/>,
<https://openmdw.ai/faq/>, and the OSI review thread:
<https://lists.opensource.org/pipermail/license-review_lists.opensource.org/2026-August/006126.html>.

## Iteration 5 — packaging audit and multi-test task coupling

### Negative result: deletion stability is vacuous for current property rules

Leave-one-demo-out replay was run on public evaluation tasks with at least four
demos. For the current size-rule battery, all 29/29 full predictions were
stable under every deletion and all 29 were correct. The same pattern held on
the 260 training outputs where the baseline battery fired. This does not prove
H8 useless; it proves that H8 must be applied to *program/version-space
families*, not to already-global shape formulas. H8 is demoted as a standalone
shape gate and retained as a program-induction feature.

### Packaging finding: the Nemotron path needs a release gate

The generated Nemotron notebook correctly switches to the competition test
file during a rerun and does not load test solutions. However, its Kaggle model
metadata points at a private user-uploaded model slug, while the official page
permits external data only when freely and publicly available. The technical
probe is useful, but the private model source must be replaced with a public,
license-cleared source—or excluded from the prize path—before submission.
The README's phrase “cannot produce or submit a competition submission” also
describes the probe inaccurately: the generated notebook is explicitly a
submission notebook. This is a documentation/release-risk issue, not evidence
of solver accuracy.

### H10 — Joint posterior, output-level action across multiple test inputs (priority: high)

ARC tasks contain one shared transformation rule but can contain multiple test
inputs. The current NVARC decoder ranks each split test output independently.
The structural constraint is still useful during inference: a candidate program
induces a vector

    v_h = (h(t_1), ..., h(t_m))

across all test inputs. The posterior should therefore be estimated over
vectors/programs, not independent grids. However, the final action must respect
the actual submission metric. If `A_j` is the set of at most two outputs
written for test position `j`, then

    U(A_1, ..., A_m) = sum_j P(Y_j in A_j | D).

The optimizer factorizes: for each `j`, choose the two highest-mass marginal
output classes under the vector posterior, even when those two classes are not
the components of one shared pair of programs. The submission format permits
this. A same-program pair remains a valuable *coherence diagnostic* or a
fallback when a deployment policy requires one rule pair, but it is a
restriction and is not the official optimum. If task-level solved rate is also
desired, add

    U_task(p,q) = P(for every j, Y_j in {p(t_j),q(t_j)} | D)

as a separate reporting metric; it must not replace the official
output-weighted objective.

**Cheap falsification.** Replay candidate vectors on multi-test tasks only.
Compare independent marginal selection against the constrained shared-vector
diagnostic at identical candidate recall. This costs no model inference and
tests whether coherence improves the posterior estimate enough to offset its
restriction. The production path should use the marginal selector.

### H11 — Output-class posterior is better than program posterior

When several programs are extensionally identical on all test inputs, they must
be collapsed into one output vector before ranking. Conversely, two programs
that fit every demo but differ on only one test input should remain distinct.
This quotient removes syntactic voting inflation while preserving exactly the
ambiguity that matters to pass@2. It is the natural bridge between symbolic
program proofs and neural grid candidates.

## Iteration 6 — joint multi-test selector prototype

Extended `experiments/pass2_selector.py` with `TaskCandidate`, a constrained
`select_task_program_pair` coherence diagnostic, and the production
`select_task_output_pairs` marginal selector. Its self-test confirms both the
correlated-family case and a counterexample where the official optimum combines
two output classes that no single pair of programs emits. The task-level vector
posterior is preserved for evidence, while the final write is correctly
output-level. This is not yet an accuracy result because no saved multi-family
candidate cache exists.

**Integration note.** The notebook's existing Leg-C `vote_outputs` already
writes one pair per test position, which is the correct action shape for the
official metric. The upgrade should be to feed it family-normalized vector
posterior marginals and verified candidate metadata—not to force one shared
program pair into every test position.

## Iteration 7 — latent roles and guarded composition

ARC-AGI-2 explicitly emphasizes symbolic interpretation, compositional
reasoning, and contextual rule application. A fixed “color 2 means X” or a
flat list of unconditional grid operators is mismatched to all three. The
following representation is the current moonshot design.

### H12 — Latent color-role algebra (priority: highest theoretical upside)

Treat colors as task-local symbols with latent roles, not as globally meaningful
integers. For each input object `o`, derive a structural feature vector
`phi(o)` that excludes color ID. Infer a role assignment `r(c,o,context)` from
cross-demo evidence, then apply a transformation in role space and render back
through a task-local color map.

One useful factorization is

    output = render_{pi_out}( T( parse_{pi_in}(input) ) ),

where `pi_in` and `pi_out` are partial color-to-role maps and `T` acts on roles,
geometry, and relations. The maps are inferred jointly with `T`, subject to
cross-example consistency. A color change is therefore explained by role
change (e.g. “marked object”, “container”, “destination”) rather than by a
memorized integer substitution.

**Why this is different from color permutation augmentation.** Augmentation
assumes a known symmetry and permutes colors globally. H12 infers when a color
is semantically bound to an object role and when it is merely a nuisance label.
The same mechanism can represent “the unique red object becomes the color of
the enclosing frame” without assuming red has a universal meaning.

**Falsification.** On public train/evaluation folds, compare a role-invariant
object encoder against raw-color encoding on tasks whose demo color identities
change or whose output introduces/removes colors. Require improvement in
verified program recall or marginal candidate coverage, not just more plausible
traces. If role inference cannot be made deterministic from the demos, it must
emit multiple role assignments as separate posterior classes.

### H13 — Guarded typed composition with effect signatures (priority: highest)

Represent a task program as a guarded composition

    if G_1(scene): A_1(scene)
    elif G_2(scene): A_2(scene)
    else: A_0(scene),

where each `G` is a typed predicate over an object/relational scene graph and
each `A` is a typed action with preconditions and postconditions. Examples of
types include `ObjectSet`, `RelationGraph`, `Canvas`, `ColorRoleMap`, and
`Grid`. Actions declare effects such as `preserves_background`, `changes_shape`,
`duplicates_objects`, or `recolors_roles`.

The effect system gives a cheap proof filter: a candidate action whose declared
effect contradicts an invariant observed in every demo is unrepresentable or
rejected before pixel execution. Guarded branches handle contextual rules
without exploding the grammar into many unrelated flat programs.

**Search strategy.** Infer guards from discriminative features between demo
examples; infer actions from object correspondences; compose only when the
producer's output type satisfies the consumer's precondition. Memoize by
`(demo_behavior_signature, effect_signature)`. This is a typed version of
execution-guided synthesis, with composition controlled by semantics rather
than token probability.

**Proof obligation.** For every accepted program, emit: object extraction,
correspondence map, guard truth table on all demos, action trace, and exact
rendered outputs. A candidate that cannot provide this certificate remains a
neural-only proposal and cannot displace a proof-carrying incumbent.

### H14 — Transductive MDL over program plus output (priority: high)

Among exact-demo-fit hypotheses, rank the joint description

    L_total(h,t) = L(h) + L(h(t) | h,D) + L(exception_set).

The second term favors outputs that are natural consequences of the inferred
rule rather than arbitrary large canvases; the exception term makes hidden
coordinate memorization expensive. For copy-like tasks, a delta encoding of
the output relative to the input is short; for object-replication tasks, a
short repetition/composition code wins. This formalizes the human “simplest
rule” prior while retaining exact pixel verification.

**Caution.** Output compressibility must be a tie-break among demo-consistent
programs, never a replacement for the transformation proof. A visually simple
wrong output can be shorter than the correct one.

## Revised architecture

The strongest design now has two coupled but separate spaces:

    raw grid -> role/object scene graph -> guarded typed programs
             -> exact execution -> output-class posterior -> pass@2 pair

The neural models propose parses, guards, and programs. The symbolic engine
checks types, effects, demo equality, metamorphic invariants, and output
validity. The selector sees only normalized candidate artifacts. This prevents
the common failure in which an LLM's fluent explanation, a DFS log-probability,
and a TRM vote are incorrectly treated as comparable confidence numbers.

## Core theory: proof-carrying hypothesis selection

### 1. Version-space view

Represent a solver hypothesis as a typed transducer `h : Grid -> Grid` from a
small object/array DSL. Define

    V(D) = { h in H : h(x_i) = y_i for every demonstration i }.

For a test input `t`, hypotheses are grouped into output equivalence classes

    C_z = { h in V(D) : h(t) = z }.

The ideal two-attempt selector chooses the two classes with largest posterior
mass `P(C_z | D,t)`, not necessarily the two largest raw generation counts.
When a generator samples from proposal `q(h|D)`, an importance-corrected score is

    w(h) = exp(-L_MDL(h)/tau) / q(h|D),
    P_hat(C_z) = sum_{h in C_z} w(h) / sum_h w(h).

This gives a principled way to combine neural samples, DSL programs, and TRM
outputs without treating one model's log-likelihood as a universal probability.

### 2. Monotonic safe integration

If a new candidate is added while the old two candidates are retained, pass@2
cannot decrease: the submitted set is unchanged or enlarged. Therefore every
new solver leg should be integrated monotonically:

- never replace a verified or incumbent candidate merely because a new scorer
  prefers it;
- replace only when the new candidate occupies an unused output class or has a
  stronger proof under a predeclared tie-break;
- preserve the old candidate as attempt 2 whenever the new leg is uncertain.

This is an exact property of the metric, not a heuristic. It protects against
ensemble regressions and makes ablations interpretable.

### 3. Proof-carrying candidate score

Use a lexicographic gate before any learned ranker:

    hard_valid(candidate) =
        demo_exact ∧ shape_ok ∧ palette_ok ∧ grid_valid ∧ timeout_ok.

Among hard-valid candidates, rank with

    S(h) = -lambda * program_length
           + alpha * invariant_count
           + beta * symmetry_equivariance
           + gamma * cross-model_support
           - delta * unexplained_trace_complexity.

`demo_exact` is necessary but not sufficient: many hypotheses fit three demos.
MDL and invariants provide an explicit bias toward the compact, object-level
rule humans tend to select. The learned judge may break ties, but must not
override a hard falsification result.

## Highest-value hypotheses in the queue

Scores below are expected *research value*, not measured accuracy. Each item
has a cheap falsification criterion so the loop can avoid spending a full
Kaggle run on an idea that fails structurally.

### H1 — Cross-family pass@2 optimizer (priority: highest)

**Claim.** The largest near-term gain is candidate selection, not more TTT.
Build an output-equivalence graph over NVARC, TRM, and verified-program
proposals. Estimate each class's posterior mass using calibrated, family-aware
weights and choose the pair maximizing union mass.

**Falsification.** On non-shadow folds, compare (a) current probmul top-2,
(b) vote-count top-2, (c) family-diverse top-2, and (d) the posterior/MDL rule.
If (c)/(d) do not improve oracle pass@2 on a fixed candidate cache, selection
is not the current bottleneck and generation gets priority.

**Why it could reach 70.** The 72.9% reference explicitly reports correct
minority recovery by joint judging. A two-attempt competition amplifies the
value of recovering a minority output class.

### H2 — Anytime structural gating (priority: highest)

**Claim.** A partial candidate trajectory reveals whether more search is useful.
At roughly halfway through decoding/program execution, compute cheap features:
component count, connectedness, bounding-box entropy, occupied-cell ratio,
color count, symmetry residual, and output-size compatibility. Stop or reroute
when the trajectory is degenerate; spend extra budget only on promising tasks.

**Evidence.** The structural-descriptor study reports within-task success
prediction from intermediate grids, transfer across solver architectures, and
large compute savings from early stopping and degenerate-trajectory detection.
Paper: <https://arxiv.org/abs/2606.09026>.

**Falsification.** Freeze a descriptor selected on fold 0–3, then evaluate it
on the untouched shadow fold only at a milestone. Require compute reduction
with no statistically meaningful loss in solved outputs.

### H3 — CEGIS with typed object algebra (priority: highest)

**Claim.** LLM-generated Python should be treated as a proposal language, not
the search space. Translate or constrain proposals into a typed DSL with
explicit object selection, relation, geometry, recoloring, replication, and
composition operators. Use counterexample-guided inductive synthesis (CEGIS):
execute on all demos, report the smallest cell/object diff, mutate only the
responsible clause, and reverify.

**Falsification.** Measure verified-program rate and unique verified output
classes per token/sec against the current 35-primitive Leg C sandbox. If typed
constraints reduce unique verified classes without improving top-2 coverage,
the DSL is too narrow or its proposal interface is wrong.

**Theoretical benefit.** A typed operator makes many invalid compositions
unrepresentable and makes the proof trace checkable. A fully demo-consistent
program plus a minimality certificate is stronger evidence than a fluent text
trace, while still leaving ambiguity visible as multiple surviving programs.

### H4 — TRM as a proposal distribution, not a final judge (priority: high)

**Claim.** The cheap TRM leg should maximize candidate coverage, then hand its
outputs to the NVARC scorer/proof layer. Do not ask a small recursive model to
calibrate its own uncertainty. Sample broadly, cluster exact outputs, and use
cross-family agreement or proof features for final ranking.

**Falsification.** On the same task cache, compare TRM-native top-2 against
cross-ranked TRM+NVARC top-2 at equal runtime. Record oracle pass@K, unique output
classes, and marginal coverage on tasks missed by NVARC.

### H5 — Group-quotient decoding (priority: high)

**Claim.** Geometric D8 transforms and legal color permutations should be
handled as a group action, not merely as independent augmentations. Canonicalize
the task, decode in quotient space, then lift a candidate back through the
group action. Enforce equivariance:

    h(g . x) = g . h(x)

for transformations `g` that preserve the task semantics. A candidate that
breaks this relation across augmented demos is penalized or rejected.

**Falsification.** Compare augmentation-as-votes against group-consistent
canonicalization on fixed outputs, measuring duplicate-candidate rate, exact
pass@2, and scoring time. The idea is rejected if the quotient removes useful
semantic asymmetry or increases false equivariance.

### H6 — Offline holistic judge (priority: high, license-gated)

**Claim.** A judge that sees all candidate traces together can recover a
correct minority candidate where independent per-candidate scoring cannot.
Use an attached open model only if its terms are acceptable for the intended
prize path. The current Nemotron Lightning artifact is marked `openmdw-1.1`,
not an OSI license; legal eligibility must be resolved before relying on it.

**Design.** Give the judge compact candidate cards, not 29 full raw traces:
candidate output, executable proof result, first failing/contradicting demo,
shape/palette facts, object map, MDL length, and family identity. Ask for a
ranking over output classes plus a short machine-readable reason. Deterministic
checks remain authoritative.

**Falsification.** Blindly permute candidate order and family labels. The judge
must remain stable and must beat majority/probmul on a fixed candidate cache;
otherwise it is only adding correlated noise.

### H7 — Constraint-aware time allocation (priority: high)

**Claim.** The 12-hour wall clock should be allocated by expected marginal
pass@2 gain per second, not evenly per task. For action `a` on task `i`, use

    value(a,i) = Delta P_i(pass@2 | a) / cost(a,i).

Run cheap CPU rules first, then choose among TTT, TRM samples, program search,
and judge calls using an anytime scheduler. Reserve a hard write/validation
buffer. Multi-test tasks should be scheduled by output-level marginal value,
because the official score is output-weighted even though the user-facing
summary may report task solve rate.

**Falsification.** Replay the scheduler on recorded per-task timings and
candidate caches. It must dominate static round-robin under the same total
budget on dev folds without touching fold 4 between milestones.

### H15 — Offline compiled transform library (priority: highest)

**Claim.** The model should spend its 12-hour budget retrieving and adapting a
small, public, executable library of transformations, not rediscovering every
primitive from natural language. A recent Kaggle discussion describes an
offline-generated library of 67 verified `transform()` functions; this is only
partial community evidence, not a score receipt, but it is aligned with the
current program-induction bottleneck.

The safe form is a *library of programs, not answers*: each entry is a typed
AST/bytecode transform plus a canonical scene-graph signature, nuisance-group
behavior, MDL length, and a proof harness. At inference time, retrieve by
signature, adapt object/color roles, and require exact agreement on every
demo before the entry can affect either submission attempt. A retrieved entry
that fails a demo is discarded, never repaired by changing its predicted
test output directly.

**Theorem (library safety and monotonicity).** Let `L` be a finite library of
executable transformations and let `V_L(D)` be the entries that reproduce
all demonstrations. If the true task rule `h*` is extensionally represented
by some `l in L` on the test input, then exact demo verification implies
`l in V_L(D)` and `l(t) = h*(t)`. Therefore adding `L` to an existing solver
cannot lower pass@2 when merge semantics retain the old candidates; it can
only add a correct output class or consume time. The runtime risk is handled
by a fixed retrieval/verification budget and a fallback to the incumbent.

**Design constraint.** Do not index by raw grid hash or memorized task ID.
Index by role-normalized object relations, size ratios, connected-component
topology, and input/output delta signatures. This makes the library a compact
grammar prior rather than public-evaluation answer memorization.

**Falsification.** On folds 0--3, compare (a) no library, (b) raw text
exemplars, and (c) typed retrieval plus exact verification at equal wall time.
Require more verified *unique output classes per second* and no increase in
demo-consistent false positives. A library that only increases duplicate
programs is rejected.

**Prototype result.** `experiments/compiled_library.py` now implements the
safe retrieval seam: invariant-signature distance, hard `demo_exact` gating,
MDL/ID deterministic tie-breaking, a distance cutoff, and a candidate limit.
Its self-test passes. This is an interface/result, not evidence that the
library has ARC coverage yet; real entries and fold replay are still required.

### H16 — Adaptive grammar expansion from zero-transition certificates (priority: highest)

**Claim.** When a task produces no valid candidate under the current DSL, more
search is mathematically wasted. Treat that event as a representation error
and ask a proposal model for the smallest missing typed operator or object
relation, then expand only that local grammar neighborhood.

For a task `D`, define `V_G(D)` under grammar `G`. If `V_G(D) = emptyset`,
the search budget inside `G` has zero expected content value after complete
enumeration. A proposal `q` should therefore be scored by the number of new
demo-consistent behaviors obtained after adding its operator, divided by the
operator's description length and verification cost. This is an adaptive
Occam loop: expand the grammar only when the current grammar has emitted a
certificate of insufficiency.

The highest-value expansions are likely relational and contextual: selecting
an object by a relation to another object, applying an action only under a
guard, and iterating until a fixed point. These should be added as typed
combinators with explicit effect signatures, not as dozens of flat templates.

**Falsification.** Log the zero-transition rate and the marginal number of
verified output classes after each expansion. If expansions produce mostly
long, one-off programs or reduce verification precision, the proposal prior
is too permissive.

### H17 — Universal-coding family calibration (priority: high)

**Claim.** Raw candidate likelihood is not a valid posterior when one family
emits many correlated samples. Use a family-level stochastic-complexity
correction. For family `f`, approximate its effective model count by the
number of distinct demo-consistent output classes (or a held-out estimate),
then score a candidate class `c` with

    log_score(c) = log evidence(c) - log effective_count(f)
                     - lambda * MDL(c).

Aggregate over families before selecting the two output classes. This is a
practical normalized-maximum-likelihood approximation: a broad family must
earn its posterior mass by producing distinct, verified behaviors, while a
narrow family is not allowed to win merely by emitting ten thousand copies
of one guess.

The existing CPU selector implements the first conservative approximation:
family-conditional mass, smoothing, hard-valid filtering, exact output-class
deduplication, vector-posterior construction, and independent marginal output
pairs for the official metric. It also exposes the constrained shared-vector
diagnostic. The next refinement is to replace the fixed family prior with a
held-out effective-count calibration.

**Falsification.** On a fixed candidate cache, compare raw vote, equal-family
mass, and effective-count correction. The correction must improve oracle
pass@2 or reduce candidate count at the same oracle score; otherwise it is
not worth the added calibration complexity.

### Iteration 8 — exact output weighting and multi-test exposure

Re-profiled the checked-in challenge files with no model calls:

| split | tasks | test outputs | multi-test tasks | test-count distribution |
|---|---:|---:|---:|---|
| training | 1000 | 1076 | 69 | 931×1, 63×2, 5×3, 1×4 |
| public evaluation | 120 | 172 | 49 | 71×1, 46×2, 3×3 |
| hidden-test shape | 240 | 259 | 17 | 223×1, 15×2, 2×3 |

This changes prioritization. Vector-level posterior construction is useful, but
the official action factorizes into independent marginal pairs. The special
multi-test logic directly affects only 36 of 259 hidden outputs (about 14%), so
it should be a cheap correctness layer, not displace larger candidate
generation and ranking work. Conversely, all budget allocation and progress
reporting must remain output-weighted: a three-test task is three scoring
opportunities, not one.

The hidden shape file contains no solutions and was used only for counts. No
hidden prediction or leaderboard score was computed.

### Iteration 9 — converged 4×L4 blueprint (theory, not a score claim)

The ideas now form one coherent solver rather than a bag of heuristics:

    input grids
      -> role-normalized object/relationship graph
      -> retrieve compact typed library entries
      -> neural proposals in missing grammar neighborhoods
      -> sandbox execution + CEGIS repair
      -> exact demo gate and invariant/effect proof
      -> output-class posterior with family correction
      -> shared-vector pair selection for multi-test tasks
      -> schema validation and submission

The four accelerators should be treated as independent proposal channels,
not four copies of the same search. One channel owns the existing NVARC/TTT
prior, one owns the typed program/repair prior, one supplies cheap recursive
or stochastic samples, and the remaining capacity is a holistic verifier or
extra program proposals. The scheduler uses `Delta pass@2 / second` estimated
from fold history, with a reserved final validation buffer. A task exits early
when it has a proof-carrying candidate and a second candidate class; it gets
more compute only when its posterior mass is diffuse or its zero-transition
certificate requests grammar expansion.

This is the key separation: generation is allowed to be high-recall and
messy, while acceptance is exact and conservative. The system can therefore
add an offline library, a new model, or a new operator monotonically without
making the incumbent worse. The 72% hypothesis is not “one model understands
72%”; it is that several partially independent proposal distributions plus a
hard verifier can cover complementary rule families, and pass@2 converts that
coverage into output accuracy.

**The decisive experiment.** Instrument the existing notebook so every
candidate is saved as a compact record (family, program/output hash, demo
proof, invariants, shape/palette facts, score, and time). Then replay only
folds 0--3. Measure the incremental oracle pass@2 of each channel, the
cross-family residual solve set, and unique verified classes per second. No
architecture change should be accepted without improving one of those
quantities at fixed wall time.

### H18 — Distributionally robust selection for the private rerun (priority: highest)

**Goalpost correction.** The checked-in campaign plan still lists the early
top-eight cutoff of 34.44. The current official public snapshot instead shows
72.08 for first place and 70.42 for second, while the public leaderboard is
only approximately half of the eventual test data. Therefore “reach 72” is a
high-end target, not merely the historical prize cutoff, and public-eval gains
must be treated as noisy evidence about the private rerun.

**Claim.** Candidate selection should optimize a distributionally robust
estimate, not the mean public-evaluation score. Partition training tasks into
coarse, leakage-safe structural groups (for example: object-count regime,
input/output size relation, connectedness, contextual-vs-compositional
descriptor, and multi-step depth). For solver configuration `a`, estimate

    robust(a) = mean_group_accuracy(a)
                  - k * uncertainty_group_accuracy(a)
                  - r * worst_group_regression(a).

Use grouped cross-validation and keep one entire structural family as a
shadow fold. A change is promoted only if it improves the mean while not
causing a lower-confidence-bound collapse in any sufficiently populated
group. This is a practical distribution-shift guard: the hidden set can
change proportions, but it should not invalidate a solver whose gains come
from multiple rule families.

**Why this matters for 72.** A public-only hand-coded rule can produce a
large apparent gain by exploiting the visible split and still fail the private
set. A verified program/library entry is safer because it must fit the current
task's demonstrations, but its *prior* should still be calibrated on grouped
training folds. The final objective is private output accuracy under the
official exact metric, not public leaderboard optimization.

**Falsification.** On training folds, compare mean-only promotion with the
robust criterion under artificial group reweightings. If robust selection
reduces held-out mean and does not improve worst-group or reweighted accuracy,
discard it. Do not tune group boundaries on public-evaluation solutions.

### Iteration 10 — candidate observability seam

Implemented `experiments/candidate_records.py` as the first instrumentation
piece. It provides stable grid normalization and SHA-256 output hashes,
family/candidate metadata, hard-validity preservation, and conversion into
output-level candidates or complete program vectors. Incomplete program
vectors are dropped during joint-posterior construction so unrelated programs
cannot be silently mixed. It also adapts the notebook's NVARC sample shape and
Leg-C verified-result shape while preserving raw evidence fields instead of
inventing calibrated weights. The module's direct self-test passes, and the
full repository suite is now 58 tests green.

This does not create candidate data or infer hidden correctness. Its value is
that one future notebook run can write NVARC/TRM/Leg-C records in a common
format, after which all selection hypotheses can be replayed without burning
another GPU run. The next implementation seam is a small notebook-side
adapter: emit one record per decoded output and one per verified Leg-C output,
then serialize records before the final merge.

### Iteration 11 — official metric oracle

Implemented `experiments/official_metric.py` with strict submission coverage,
exact two-attempt validation, and output-weighted pass@2 scoring. A synthetic
check confirms that one correct output out of a two-output task plus one
correct single-output task scores `2/3`, not a task-average surrogate. The
module and the full repository suite pass (61 tests total).

Against the checked-in public evaluation solutions, the scorer returns exactly
`1.0` for a perfect first-attempt submission and `0.0` for an all-zero
submission across all 172 outputs.

This matters because the notebook's diagnostic `validate_submission` and
selection benchmark are task-weighted for convenience. They must not be used
as the objective for H1/H10/H17/H18 replay. The future replay harness should
use this scorer for every candidate pair and report task solve rate only as a
secondary diagnostic.

### Iteration 12 — replay harness and coverage guard

Implemented `experiments/replay_harness.py`. It consumes normalized records,
applies the calibrated output-level selector, emits every required task/test
entry with two attempts, and scores through the strict official metric. Missing
candidate positions become explicit valid placeholders rather than silently
disappearing. Its self-test and the full repository suite pass (64 tests).

This closes the offline loop needed by the autoresearch method: future
candidate artifacts can be compared by changing only the selector or scoring
policy, without regenerating model outputs. The harness intentionally does
not use hidden labels to choose candidates; labels are accepted only by the
separate evaluation oracle during local fold replay.

### Iteration 13 — geometric-orbit lower bound (negative result)

Ran the new replay stack on the checked-in labeled splits using only the
eight D8 geometric transforms of each test input. The orbit contained the
exact target for 8/1076 training outputs (0.743%) and 0/172 public-evaluation
outputs (0.000%). The evaluation replay emitted 1,376 normalized records and
the calibrated two-attempt selector scored 0.0000.

This falsifies “add a geometry-only fallback” for ARC-AGI-2. It does not
falsify geometric augmentation: D8 remains useful when applied jointly to a
learned/programmatic transformation and when enforcing a justified
equivariance relation. The important distinction is between transforming the
input and transforming the inferred rule. The latter is the only plausible
H5 path.

### H21 — Group-conditioned posterior calibration (priority: high)

**Claim.** The family prior and candidate score should depend on the task's
visible structural group `g`, but the calibration must shrink aggressively for
small groups. For family `f` and group `g`, estimate a Beta/Dirichlet posterior
from grouped training-fold outcomes, with a global family prior as the
hierarchical parent. Use the resulting `P(correct output class | f,g)` to
weight candidates, while retaining hard demo verification.

This is more principled than one global family prior: a dense transduction
family may be strong on large grids, while a program family may dominate small
relational tasks. The hierarchical shrinkage prevents the 240-task hidden set
from being overfit to noisy tiny buckets. H18 then promotes a configuration
using group lower confidence bounds rather than public mean alone.

**Falsification.** Reweight fold-0--3 outcomes to the observed hidden-shape
distribution and compare global-prior, unregularized group-prior, and
hierarchically shrunk group-prior selection. Keep the group model only if it
improves reweighted output pass@2 without lowering ordinary held-out score or
creating unstable priors in low-count groups.

### Iteration 14 — public/private structural shift

Implemented `experiments/structural_groups.py` and compared only
challenge-visible input features between the 120-task public evaluation set
and the 240-task hidden-test shape file. Total-variation distances were:

| feature | TV distance |
|---|---:|
| test area bucket | 0.4500 |
| test palette bucket | 0.3500 |
| test-count bucket | 0.3375 |
| training area bucket | 0.3458 |
| test component bucket | 0.1958 |
| demo-count bucket | 0.0958 |
| square/non-square | 0.0125 |

The largest shifts are not subtle: public evaluation has 74/120 tasks with
maximum test area above 400 cells, versus 40/240 hidden tasks; hidden tasks
have many more small inputs and fewer multi-test tasks. This is direct evidence
against tuning a selector or time allocator solely for public-eval averages.
It also supports H18's grouped lower-confidence-bound criterion. The features
are safe for inference because they use only train inputs and test inputs; no
hidden test outputs were read.

### Iteration 15 — fixed-primitive compiler lower bound (negative)

Implemented `experiments/verified_primitives.py`, a deliberately small
demo-verified compiler covering D8/transpose, crop-to-content, four gravity
directions, 2--3× tiling/upscaling/downscaling, and consistent cellwise color
maps. Every candidate is checked against every training pair before being
applied to a test input.

On the 1,000-task training split, only 19 tasks produced at least one verified
primitive; the resulting candidates contained 18/1076 exact test outputs
(1.67%), and the calibrated two-attempt replay scored 1.77%. On the 120-task
public evaluation split, zero tasks produced a verified primitive and recall
was 0/172. The module itself passes its targeted tests and the full suite is
now 77 tests green.

This is not evidence that the implementation is broken: synthetic tests cover
each operator and the training split supplies a small nonzero sanity signal.
It is evidence that a fixed flat primitive library is nowhere near sufficient
for ARC-AGI-2 evaluation. H16/H20 therefore move up in priority: the solver
needs neural-guided relational/contextual grammar expansion and execution
feedback, not merely more enumeration of these operators.

### H22 — Bipartite object correspondence as the primitive (priority: highest)

**Claim.** The atomic search unit should be a correspondence between input and
output objects, not an unconditional grid operator. For each demo, construct
object sets `O_in` and `O_out` with role-neutral shape descriptors. Infer a
low-cost partial matching `pi` plus typed relations (same anchor, translation
offset, containment, adjacency, alignment, or unmatched add/delete). The
candidate program then acts on roles and relations:

    P = (object selectors, matching pi, relation predicates, actions).

Use a global assignment objective rather than greedy nearest-object matching,
retaining the top few assignments when symmetries make `pi` ambiguous. A
program survives only if one consistent correspondence/role assignment
explains every demonstration. This turns contextual rules into a compact
relational algebra and allows neural proposals to name the missing relation
without writing unrestricted pixel code.

**Complexity intuition.** If a grid has `n` salient objects, assignment is
polynomial (Hungarian-style matching is `O(n^3)` for a fixed cost matrix),
while unconstrained cell-level hypothesis search grows exponentially in the
number of cells. The hard part is not matching itself but choosing invariant
descriptors and preserving ambiguity until later evidence resolves it.

**Falsification.** Replace the profiler's conservative greedy matching with
top-k global correspondence search on folds 0--3. Measure verified program
recall, unique output classes per second, and false verified rate. Reject the
representation if it cannot explain more demo-consistent behaviors than the
fixed primitive compiler at comparable CPU cost.

### Iteration 16 — object-delta inventory

The new profiler scanned every training demonstration and counted tasks whose
*every* demo contained each label. Labels overlap; the profiler is a routing
signal, not a mutually exclusive ground-truth taxonomy.

| stable demo motif | training tasks | public-eval tasks |
|---|---:|---:|
| object add | 709 | 99 |
| object delete | 702 | 100 |
| grid resize/reframe | 315 | 37 |
| object recolor | 114 | 12 |
| object move | 89 | 12 |
| object shape transform | 77 | 6 |

The dominant add/delete counts often occur together because the simple shape
matcher deliberately refuses uncertain correspondences; they should not be
read as proof that literal addition and deletion explain 70--83% of tasks.
The robust conclusion is narrower: object correspondence, unmatched-object
handling, and contextual relations deserve first-class grammar support. The
profiler passes its tests and uses only demonstration pairs.

### Iteration 17 — global correspondence ambiguity audit

Ran the exact top-k assignment DP on all demonstration pairs with at most ten
objects per side (`k=4`). On training, 2,624/3,232 pairs were within the exact
cap; 720/2,624 (27.44%) had multiple minimum-cost assignments. On public
evaluation, 247/358 pairs were within the cap; 98/247 (39.68%) had tied best
assignments. The remaining 608 training and 111 evaluation pairs exceeded the
cap and were explicitly reported rather than approximated.

This is partial positive evidence for retaining a correspondence version
space: ambiguity is substantially more common on evaluation than training,
so greedy matching can erase precisely the uncertainty ARC-AGI-2 needs. It is
not yet evidence of solver accuracy—the current cost is a structural heuristic
and the DP does not infer actions. The next gate is to attach typed relation
and action synthesis to the top-k matches, then compare verified candidate
coverage against the fixed primitive compiler.

### Iteration 18 — quotient-valued graph transduction

The correspondence result suggests a sharper mathematical object than a single
best program. Let `G_in` and `G_out` be object-centric scene graphs, `M` a
partial bipartite correspondence, and `T` a typed graph transducer with guards,
effects, and an execution trace. An explanation is

    z = (M, T, trace),       y(z) = execute(z, G_test).

The solver should enumerate explanations only while they reproduce every
training pair exactly, but it should *quotient* them by their predicted test
grid:

    Z_y = { z : y(z) = y },
    P(y | D, test) ∝ Σ_{z∈Z_y} exp(-L(z) - λ·U(z)).

Here `L` is a description length for the correspondence, typed actions, and
guards; `U` penalizes unsupported or weakly grounded structure. This has two
consequences. First, tied correspondences are not discarded merely because a
geometric cost has an arbitrary tie-break—each can contribute evidence to its
output class. Second, programs that differ syntactically but make the same
prediction do not waste the selector's probability mass as if they were
independent discoveries. This is the appropriate bridge between H14/H17's
MDL/posterior view and H22's correspondence version space.

For the competition utility, the decision rule is exact once this posterior is
calibrated: because each test output contributes additively and the submission
allows two attempts, the Bayes-optimal pair is the two highest-mass *output
classes*. A shared program pair can still be retained as a coherence diagnostic,
but it is not the objective-maximizing action when test outputs are scored
independently. This theorem is the reason diversity belongs in candidate
generation and calibration, not as an unconditional override of posterior
ranking.

The implementation gate is therefore narrow: synthesize typed actions over the
top-k exact matchings, merge by output hash, and measure output-class recall,
calibration, and official pass@2 on folds 0--3. Do not claim a gain from the
correspondence audit itself. Falsify the idea if correspondence marginalization
does not recover any verified candidates that greedy matching misses, or if its
description-length correction worsens held-out output calibration at equal
compute.

### Iteration 19 — typed action consensus audit (diagnostic negative)

Added a proof-oriented synthesis seam that converts each top-k matching into
typed local effects (`move`, `recolor`, `move_recolor`, `transform`, `add`,
`delete`, or `identity`) and intersects canonical families across all demos.
The key intentionally removes object indices and concrete color ids, but still
retains exact shapes, palette cardinalities, and displacement vectors.

On the full challenge files, the current family key was too strict:

| split | tasks | greedy family stable | tasks with a top-k-only recovered stable family | skipped over cap |
|---|---:|---:|---:|---:|
| training | 1,000 | 16 | 17 | 291 |
| public evaluation | 120 | 0 | 0 | 50 |

This is a useful negative result, not a solver score. It falsifies the naive
claim that exact per-object shape/displacement tuples are already the right
task-level abstraction. ARC-2 requires quotienting at least over object
cardinality, relative relations, and role-conditioned transformations before
cross-demo consensus can become a proof gate. The correspondence hypothesis
survives only in the weaker form: retain latent matchings, but synthesize
relational action schemas rather than literal object-local signatures.

The next design is a relation-normalized family key. Replace absolute object
attributes with predicates over a scene graph (same-color, same-shape,
nearest, aligned, inside, between, border-touching) and express effects as
quantified rules over those predicates. A family is admitted only if the
quantified rule reproduces every demo and its effect set is closed under the
guard. Falsify it if relation-normalized families still produce no held-out
verified candidates, or if they increase family count without improving
output-class recall under the official metric.

### Iteration 20 — relation-normalized family audit (partial positive)

Implemented the next quotient: action schemas retain effect type, equality or
change relations, coarse area/palette relations, motion axis, and clipped scene
context (`left/right/above/below/aligned` counts), while discarding object ids
and exact motion distances. On the same `k=4`, ten-object-cap audit:

| split | tasks | greedy family stable | tasks with a top-k-only recovered stable family | stable family count | skipped over cap |
|---|---:|---:|---:|---:|---:|
| training | 1,000 | 77 | 75 | 186 | 291 |
| public evaluation | 120 | 3 | 4 | 8 | 50 |

The increase from 16 to 77 training tasks and from 0 to 3 public-evaluation
tasks is evidence that exact shape/displacement tuples were overfit. The four
public-evaluation tasks with a top-k-only stable family are particularly useful
targets: they are the smallest controlled set where marginalizing matching
ambiguity changes the proof candidate set. None of these counts is a solved
output or leaderboard estimate; the schema still has no executor and the
correspondence cost remains heuristic.

The next proof obligation is to add an executor for only the closed subset of
schemas whose effect is fully specified (single-axis moves, homogeneous
recolors, and explicit unmatched-object add/delete). It must replay every demo
exactly, reject any schema with an unbound target location or role, and emit
the test grid as a candidate only after that gate. This keeps the relation
quotient from becoming an unconstrained pattern matcher.

### Iteration 21 — closed object executor lower bound (negative)

Added an executor with a strict proof gate for minimum-cost correspondences.
It can execute only exact single-object moves, exact recolors, deletions, and
fixed-anchor additions; shape transforms, ambiguous source fingerprints,
collisions, out-of-bounds writes, and higher-cost correspondence explanations
are rejected. The minimum-cost restriction is essential: without it, a shape
transform can be memorized as delete-plus-add.

The executor passed its unit tests but verified **0/1,000 training tasks and
0/120 public-evaluation tasks**. This is a lower bound on a deliberately closed
DSL, not evidence that correspondence is useless. It falsifies the narrower
engineering shortcut that exact object fingerprints can be carried from the
first demo into every other demo. The executor needs role variables selected
by relations, not literal `(shape, colored_shape)` keys.

### H23 — graph anti-unification for role induction (priority: highest)

The next moonshot is to anti-unify verified per-demo traces into a least
general relational program. For each demo, represent objects as variables with
attributes (shape, size, palette role, position) and edges with predicates
(same-color, same-shape, nearest, aligned, inside, between, border-touching).
Given top-k correspondences, compute a graph least-general-generalization:

    LGG({trace_1, ..., trace_n}) = the most specific schema subsuming every trace.

Concrete colors, object ids, exact distances, and demo-specific cardinalities
become variables only when the corresponding relation is preserved across all
demos. An action is admitted only when its guard selects a unique role set in
each demo and its effect set is closed under that guard. This is the formal
replacement for carrying the first demo's fingerprints unchanged.

The search can remain finite: enumerate top-k matchings, anti-unify only
bounded graph fragments, and rank schemas by `description_length +
unsupported-variable-penalty`. Proof obligations are exact replay on every
demo, unique role selection, and no writes outside the declared effect set.
The output posterior still quotients all surviving schemas by predicted test
grid before selecting the two highest-mass classes.

Falsify H23 if bounded graph anti-unification produces no additional
demo-verified candidates beyond the fixed primitive compiler, or if its role
guards are not stable on a held-out fold. A mere increase in schema count is
not a success; the required signal is new exact candidate coverage or better
calibration at fixed compute.

### Iteration 22 — bounded graph LGG profile (partial positive)

Implemented the first H23 kernel: id-free role predicates over bounded scene
graphs, action observations from explicit correspondences, and a least-common
schema that keeps only invariant effect fields and guard predicates. On the
top-1 correspondence of each demo (no solutions file used):

| split | tasks | equal-length traces / LGG tasks | fully typed LGG tasks | LGGs with source guard | LGGs with target guard | cap skips |
|---|---:|---:|---:|---:|---:|---:|
| training | 1,000 | 169 | 135 | 167 | 164 | 291 |
| public evaluation | 120 | 9 | 7 | 9 | 9 | 50 |

This is a stronger signal than the literal-family audit: the role-level schema
survives demo variation on 7 public-evaluation tasks, whereas the closed
fingerprint executor verified none. It is not candidate correctness—the LGG
does not yet select roles on a test grid or execute its effects—and top-1
correspondences leave ambiguity untreated. The immediate proof gate is to
compile the fully typed schemas into guarded selectors and replay them on all
demos, then compare exact output-class coverage against the fixed primitive
lower bound. Only schemas with unique role selection and closed effects may
enter the posterior.

### Iteration 23 — unique-role proof profile (partial positive)

Added the first guarded-selector gate. For the bounded single-action subset,
`select_roles(grid, guard)` returns the objects satisfying the id-free LGG
predicates; a schema is proof-admissible only when both source and target
guards are nonempty and select exactly one object in every demonstration.

| split | bounded LGG tasks | single-action LGGs | unique-role tasks | ambiguous/empty-role tasks | cap/no-LGG |
|---|---:|---:|---:|---:|---:|
| training | 169 | 57 | 52 | 5 | 831 |
| public evaluation | 9 | 3 | 3 | 0 | 111 |

The 3 public-evaluation tasks with unique roles are the first concrete H23
targets for a relational executor. This result proves only identifiability of
the selected demo roles; it does not prove the test-time effect, translation
magnitude, color-role mapping, or output correctness. Top-1 correspondences
were used, so ambiguity must still be marginalized in the final candidate
generator.

The next executor should represent effects as relations between selected roles
(for example, “move selected source to the position obtained by applying the
demo-inferred displacement rule”) rather than source fingerprints. It must
replay demos exactly, select one role on each test input, and reject any effect
whose parameter is not grounded by a preserved relation. This is the next
meaningful bridge from H23 evidence to candidate-output recall.

### Iteration 24 — relational role-effect executor lower bound (negative)

Compiled the unique-role, single-action LGGs into a proof-gated executor. The
source role is selected by the generalized guard; moves preserve the selected
object while applying the first-demo displacement, recolors apply the first
demo positional color map, and deletions clear only the selected role. A
program enters the candidate set only after exact replay of every demo.

| split | compiled demo-verified tasks | emitted test candidates | exact test outputs |
|---|---:|---:|---:|
| training | 1/1,000 | 0 | 0 |
| public evaluation | 0/120 | 0 | 0 |

This is a negative lower bound, not a failure of H23 role induction. It
falsifies the simpler effect hypothesis that a role-generalized selector can
reuse a literal displacement or literal recolor map from the first demo. The
remaining effect must itself be relational: destination should be expressed as
a preserved graph relation (nearest object, border, alignment, reflection,
or another selected role), and color changes must use latent palette roles.

The next proof target is relational transport. Infer a bounded effect equation
from each demo trace, anti-unify the equation rather than its numeric result,
and execute it only when all referenced roles are uniquely selected. For
example, `anchor(target) = anchor(source) + delta` is admissible only when
`delta` is invariant; otherwise search equations such as “move to the cell
adjacent to role B” or “reflect across role B's axis.”

### Iteration 25 — two-role relative transport lower bound (negative)

Implemented a relational transport primitive with equation

    anchor(mover') = anchor(reference) + invariant_offset.

The mover and reference are selected by nonempty id-free guards; the mover's
shape and colors are preserved; collisions and out-of-bounds writes reject the
candidate; and the program must replay every demo exactly. The prototype passes
its unit tests but compiled **0/1,000 training tasks and 0/120 public-evaluation
tasks**, producing no test candidates.

The failure is a useful scope boundary: requiring exactly one mover and exactly
one stationary reference in every demonstration is too narrow for scenes with
additional effects or changing object counts. It does not falsify relational
transport in general. The next version must be compositional: identify one
transport clause inside a multi-action trace, preserve unrelated objects by an
explicit frame condition, and anti-unify the reference relation across all
correspondence alternatives. A clause may enter a program only if its guarded
effect set is disjoint from the frame or the interaction is explicitly typed.

### Iteration 26 — compositional frame-condition profile (partial positive)

Implemented a separation-logic-style audit over top-1 correspondence traces.
Each clause receives source and target footprints; a task passes the structural
composition gate when its LGG is fully typed, every role guard is unique and
nonempty, clause footprints are pairwise disjoint, and every changed cell is
covered by an active clause. The frame condition then certifies that all cells
outside the declared effects are preserved.

| split | bounded traces | equal-length LGGs | fully typed | unique roles | disjoint/frame-valid | composable proof tasks |
|---|---:|---:|---:|---:|---:|---:|
| training | 709 | 169 | 135 | 66 | 114 | 45 |
| public evaluation | 70 | 9 | 7 | 3 | 8 | 3 |

This is the first shortlist where multi-action composition is structurally
admissible: 3 public-evaluation tasks survive every current proof predicate.
It is not executable correctness—the clauses still lack relational effect
parameters and no test outputs were generated. The frame rule is nevertheless
valuable because it permits an active transport/recolor clause to be verified
locally while guaranteeing that unrelated objects are not rewritten. The next
implementation must compile each active clause, replay the composition exactly,
and reject any clause whose guard overlaps another clause's write footprint.

### Iteration 27 — frame-role executor lower bound (negative)

Compiled multi-action LGGs into frame programs: identity clauses act as a
frame, active clauses use generalized source guards, and all writes are
performed together after unique-role checks. The executor accepts only move,
recolor, and delete clauses with first-demo effect parameters, rejects adds and
transforms, and requires exact replay of every demonstration.

It passes its tests but compiles only **1/1,000 training tasks and 0/120
public-evaluation tasks**, producing no test candidates. This confirms that
frame conditions solve the safety/composition problem but not the effect
induction problem. Literal first-demo displacements and color maps remain too
specific even when multiple local clauses compose correctly.

### H24 — finite symbolic effect-equation library (priority: highest)

Replace literal effect parameters with a bounded term language over selected
role anchors and attributes. Candidate terms include constant translation,
translation relative to another role, reflection across a role-defined axis,
nearest/aligned/border placement, copying a shape from a role, and palette-role
maps. Enumerate terms by description length, instantiate them on every demo,
and retain only equations that reproduce all outputs while satisfying the frame
condition.

For a selected role `A` and reference roles `B,C`, the search is finite over
terms such as `anchor(A) + (anchor(B)-anchor(C))`, `reflect(A, axis(B,C))`, or
`nearest(A, relation(B))`. A term is proof-admissible only if every variable is
uniquely guarded, every read role is unchanged or explicitly sequenced, and
the resulting write footprint is closed. This turns “guess the displacement”
into exact equation induction and naturally handles demo-varying layouts.

Falsify H24 if the equation library adds no exact candidates over the frame
lower bound, or if it increases candidate volume without improving official
output-class recall/calibration at equal search time. The priority is coverage
per CPU second, not the number of equations discovered.

### Iteration 28 — symbolic effect-equation extraction lower bound (negative)

Added a finite typed equation kernel with constant offsets, reference-plus-
offset terms, and source-plus-reference-delta terms. Terms are enumerated from
the first observation, ranked by description length, and admitted only when
they exactly fit every observation.

The synthetic proof tests pass, including a reference-relative equation whose
numeric displacement changes with layout. The first data-facing extractor was
deliberately narrow: it required exactly one unchanged-shape move and at least
one stationary reference under the top-1 correspondence. It found only 1/1,000
eligible training tasks and 0/120 public-evaluation tasks, with 0 stable
equation fits in either split.

This is a negative lower bound on the extractor, not on equation induction.
It shows that the equation search must start from role-aligned clauses inside
multi-action traces, tolerate several stationary/reference roles, and use
correspondence alternatives rather than requiring one greedy matching. The
finite term language remains a useful proof substrate once those inputs are
available.

### Iteration 29 — action-trace alignment over top-k correspondences (partial positive)

Replaced sorted clause pairing with a bounded exact action-level assignment DP.
The cost prioritizes action kind and coarse effect semantics, while treating
role guards as soft evidence. Full alignments are then anti-unified, and an
anytime cap limits top-k correspondence combinations.

With `k=4` and `max_hypotheses=1`:

| split | bounded tasks | sorted top-1 LGG | aligned top-1 LGG | top-k aligned LGG tasks | aligned schema count |
|---|---:|---:|---:|---:|---:|
| training | 709 | 169 | 151 | 207 | 151 |
| public evaluation | 70 | 9 | 9 | 13 | 9 |

The aligned top-1 count is lower on training because some sorted traces are
not full semantic bijections under the stricter assignment cost. The important
signal is the top-k rescue: correspondence alternatives raise the candidate
task set from 169 to 207 on training and from 9 to 13 on evaluation. This is
still schema evidence, not output correctness; the next gate must attach role
guards and relational effect equations to each aligned schema, then exact
replay and official output-class scoring decide whether the extra hypotheses
are useful or merely combinatorial noise.

### Iteration 30 — aligned-clause proof profile (diagnostic negative)

Added local-index preservation through action-level alignment, so each LGG
hypothesis retains the source/target object used by every demo trace. With
`k=4` and one anytime hypothesis per correspondence combination:

| split | tasks with aligned hypothesis | unique-role hypotheses | grounded-effect hypotheses |
|---|---:|---:|---:|
| training | 207 | 71 | 1 |
| public evaluation | 13 | 3 | 0 |

The 207/13 hypothesis counts reproduce the top-k alignment expansion, and the
71/3 unique-role counts show that alignment does not erase the earlier H23
identifiability signal. The grounded-effect check accepts only constant move
equations, exact recolor targets, and deletions; its collapse to 1/0 is a
controlled negative result. The remaining bottleneck is effect-equation
expressivity, not merely correspondence or role selection.

The next gate is to fit H24 terms to aligned role anchors and reference-role
assignments, then use the frame condition to compose those clauses. Any new
candidate must still pass exact demo replay; increased schema count alone is
not progress.

### Iteration 31 — aligned equation fitting (diagnostic negative)

Preserved local indices through top-k trace alignment and fitted H24 terms to
the resulting source/target anchors. The bounded reference search enumerated
stationary identity clauses as possible role `B`, while direct terms used the
source role `A`; every equation was required to fit all aligned demos.

| split | aligned hypotheses | hypotheses with any grounded equation | hypotheses with reference-relative equation | total equation evidences |
|---|---:|---:|---:|---:|
| training | 207 | 3 | 0 | 3 |
| public evaluation | 13 | 1 | 0 | 1 |

This is a diagnostic negative for the current reference extractor, not for
H24. The alignment/role signal survives, but stationary-only references are
too sparse and the surviving constant terms do not explain held-out layouts.
The next formulation must allow a reference role that is itself transformed,
use its *post-effect* anchor in the equation, and sequence or jointly solve
the coupled clauses. The correct proof object is therefore a small system of
guarded equations plus an execution order/frame proof, rather than one scalar
offset attached to one clause.

### Iteration 32 — post-effect relation equations (partial positive)

Expanded H24 reference search from stationary identity clauses to every aligned
reference clause, including objects that also transform. For each active
clause/reference pair, the system computes a coarse post-effect relation
(direction, axis, and clipped distance bucket), intersects the reference role
guard across demos, requires that guard to select exactly one input object per
demo, and retains only relations invariant across all aligned traces.

With `k=4`, one bounded aligned hypothesis per task, and at most eight reference
choices per clause:

| split | aligned hypotheses | hypotheses with relation equations | uniquely guarded relation evidences | total relation evidences |
|---|---:|---:|---:|---:|
| training | 207 | 23 | 75 | 75 |
| public evaluation | 13 | 2 | 11 | 11 |

This is the first positive H24 signal on real ARC-2 traces: post-effect
relations survive even when the reference object itself is transformed, while
the earlier stationary-only equation extractor found none on evaluation. It
remains schema evidence, not a solved output—the relation descriptor is coarse,
role/effect sequencing is not compiled, and top-k posterior ranking is not yet
measured.

The next proof gate is a simultaneous clause executor: select all uniquely
guarded roles, solve relation equations in a dependency order (or as a bounded
constraint system), apply disjoint writes, and replay every demo exactly. Reject
cycles without a jointly solvable anchor system and reject relations that leave
multiple admissible test placements.

### Iteration 33 — unique relational-placement proof (diagnostic negative)

Added exact anchor enumeration for every fitted post-effect relation. For each
relation equation, the target object's shape is placed at every in-bounds
anchor satisfying the coarse direction/axis/clipped-distance relation while
all other output objects are blocked. The equation is placement-admissible
only when the demonstrated target anchor is the sole candidate on every demo.

| split | hypotheses | with relation equations | relation evidences | uniquely determining placements | hypotheses with one |
|---|---:|---:|---:|---:|---:|
| training | 207 | 25 | 79 | 8 | 2 |
| public evaluation | 13 | 2 | 10 | 0 | 0 |

The result falsifies the idea that a coarse post-effect relation by itself is
enough to generate a deterministic ARC-2 output. It does not reject
relational equations: the relation signal survives, but placement must also
use source-to-target relation transitions, shape orientation/geometry, and a
joint occupancy/frame constraint. The correct solver object is a coupled
finite constraint system over all role anchors, not an independent placement
per clause.

The next proof gate will enumerate those coupled systems, reject systems with
multiple test solutions, and retain every exact solution as a separate output
class for the official pass@2 posterior.

### Iteration 34 — coupled anchor CSP substrate (theoretical positive)

Implemented a bounded finite constraint solver for target-anchor systems. Its
variables are role anchors; domains enforce grid bounds, shape fit, and blocked
frame cells; relation constraints enforce the post-effect predicates; and
non-overlap is checked incrementally. MRV ordering and explicit node/solution
budgets keep the search compatible with an anytime Kaggle pass.

The soundness statement is direct: every returned assignment satisfies every
declared relation, every shape is in bounds, no write intersects the frame or
another role, and the solver never labels a budget-exhausted search as
unsatisfiable. With `max_solutions=2`, distinct satisfying assignments remain
available as separate output classes for the official pass@2 posterior.

The CSP passes focused tests but has not yet been connected to real aligned
ARC-2 clauses. The next gate is to construct one system per aligned H24
hypothesis, add source-to-target relation-transition constraints and effect
dependencies, then require a unique test assignment before emitting a grid.

### Iteration 35 — coupled role-CSP integration lower bound (negative)

Connected aligned H24 relation equations to the anchor CSP. The compiler
requires every `move` clause to have a guarded post-effect relation, fixes
identity/frame clauses at their input anchors, blocks unrepresented output
cells, and demands exactly one satisfying target assignment on every demo.
Only pure move-plus-identity programs are admitted; recolor/transform/add
clauses are rejected until their effects are grounded.

The CSP integration compiled **0/1,000 training tasks and 0/120
public-evaluation tasks**, producing no candidates or exact test outputs. The
negative is attributable to the compiler's deliberately narrow effect algebra:
the CSP's soundness and bounded-search tests pass, but real aligned hypotheses
rarely consist solely of moving roles with already-fitted relations. The next
extension must add typed recolor and transform effect variables, preserve the
frame rule, and solve all active clauses jointly rather than relaxing the proof
gate.

### H19 — Constraint-guided output completion (priority: high)

**Claim.** Shape and palette predictors should be more than token masks. Treat
the output as a finite constraint-satisfaction problem whose variables are
cells and whose hard constraints come from exact demo-inferred invariants:
dimensions, allowed colors, preserved background, object counts, symmetries,
and typed program effects. Use neural logits only as a soft cost for the
remaining variables, then solve or sample the constrained completion.

For output variables `y`, let `C(D,t)` be the proven constraint set and
`ell(y)` the negative model log-likelihood. The candidate generator solves

    y_hat = argmin_y ell(y)  subject to  y in C(D,t).

For multiple feasible completions, preserve each exact grid as a separate
output class and rank with the family-calibrated posterior. This is strictly
safer than independently patching cells: every completion remains globally
valid under the same invariant set. It also converts a high-entropy 30×30
token decode into a smaller search over unconstrained cells.

**Caveat.** An empirical superset bound is not a hard constraint. A constraint
enters `C` only when it is proved by a typed program or a rule that reproduces
all demonstrations with an invariant-preservation argument; otherwise it is a
soft ranking feature. This preserves recall when a public-eval-derived rule is
wrong on a hidden task.

**Falsification.** On folds 0--3, compare unconstrained DFS, hard exact
constraints, and soft constraints at equal decoder time. Record valid-candidate
recall, unique output classes per second, and oracle pass@2. Reject the method
if any hard constraint lowers recall on a fold where its proof precondition
holds, or if it merely reproduces the same candidate classes more slowly.

### H20 — First-divergence trace repair (priority: highest)

**Claim.** A failed program should be repaired at the earliest intermediate
state that disagrees with a demonstration, rather than by asking for a new
global program. Let a deterministic program be a composition

    s_0 ->[o_1] s_1 ->[o_2] ... ->[o_k] s_k.

For a demo, let `j` be the first index where the observed state differs from
the reference state. The prefix through `s_(j-1)` is evidence-preserving; a
repair search should mutate only the operator/guard and typed inputs that can
affect `s_j`, then re-execute the unchanged suffix. This is a causal
localization rule, not merely a better prompt. If the repaired prefix is
verified and the suffix is deterministic, all earlier demo states remain
correct by construction.

The Leg-C runtime currently returns final outputs and a partial failure
location. Upgrade the typed path to emit compact intermediate states, object
maps, and effect signatures. The repair prompt receives the first divergent
state plus the responsible clause, while the candidate record stores the
trace hash and proof status. This should reduce mutation branching and stop
the model from repeatedly rewriting already-correct perception code.

**Falsification.** On a fixed set of partial candidates, compare global
repair, final-output diff repair, and first-divergence repair at equal model
calls and sandbox time. Require higher verified-programs/second and at least
as many distinct verified output classes. Reject it if trace overhead erases
the search reduction or if localization is unstable under equivalent object
orderings.

**Prototype result and limitation.** `experiments/trace_repair.py` implements
state normalization, deterministic trace hashes, first-divergence detection,
and repair-target mapping; its self-test passes. The causal guarantee applies
only when reference intermediate states are available from a typed program
decomposition, verified generator, or other grounded trace. Final demo outputs
alone do not identify the correct intermediate states, so a model must not
invent them and call the result a proof. In the absence of grounded traces,
the safe fallback is final-output execution diff plus ordinary CEGIS.

## What is probably not the moonshot

- More epochs of the same per-task LoRA TTT: existing research and the local
  plan indicate early gains saturate; it does not address missing DSL coverage
  or candidate selection.
- Retraining on the same public NVARC corpus: the current checkpoint already
  descends from that corpus; it is unlikely to create the missing capability.
- A larger monolithic model without a proof/selection layer: scale alone has
  historically underperformed diverse refinement systems on ARC-AGI-2.
- Public-eval-tuned hand-coded rules: a prior CPU symbolic engine reportedly
  collapsed on hidden evaluation. Only rules verified on each task's demos are
  safe to use as hard constraints.

## Recommended experiment order

1. Cache candidate outputs and per-candidate metadata from the existing NVARC,
   Leg C, and TRM legs on folds 0–3.
2. Run H1 selection replay; no model or Kaggle submission needed.
3. Add H2 trajectory descriptors and replay the same cache/timing trace.
4. Upgrade Leg C with CEGIS execution-diff feedback (H3), then re-run H1.
5. Add H20 first-divergence traces to the typed repair path, then measure H3.
6. Align H23 clauses across correspondence alternatives; compile H24 symbolic
   effect equations into the coupled anchor/occupancy CSP, then execute only
   unique test solutions.
7. Add H19 constrained output completion only behind proved invariants.
8. Add H21 grouped calibration and H18 robust promotion checks.
9. Add TRM proposal sampling and cross-family reranking (H4).
10. Test group quotienting (H5) and the time allocator (H7) separately.
11. Only then evaluate an offline judge (H6), after license and Kaggle packaging
   review.
12. Touch the shadow fold only at a major milestone; record it as a new ledger
   row before any scored submission.

## Current status

- Repository tests: 179 passed, including the selector, compiled-library,
  candidate-record, official-metric, replay-harness, trace-repair, and
  geometric/structural-group/verified-primitive/object-delta/correspondence
  action/relational-family/closed-executor/graph-LGG/guarded-role/role-effect/
  relational-transport/compositional-clause/frame-role/effect-equation/trace-
  alignment/aligned-clause/aligned-equation/relation-equation/placement/
  palette-role-map/placement/CSP/coupled-role/augmentation-scoring/cache/
  compute-allocation/domain-adaptation/hierarchical-calibration/fold-
  calibration regressions.
- Local symbolic audit: complete, results recorded above.
- Rules/recon verification: complete for this loop iteration.
- GPU model inference: not run locally; no score is being claimed.
- Multi-test selector: CPU self-test and regression tests passed; empirical
  replay is pending candidate vectors.
- Research state: active; next loop action is to connect real candidate records
  to class/vector replay, then measure H1/H10/H17/H18/H19/H20/H21/H22/H23 on folds 0–3. The library and
  adaptive-grammar ideas remain theory-only until that replay exists.

## Iteration 36 — H12 latent palette roles: exact proof gate

### Hypothesis

Colors should be represented as task-local role symbols.  A role-conditioned
recolor map can therefore explain cases where one source color maps to
different target colors depending on a color-blind object's structural role:

    output(cell) = M(input_color(cell), role(object(cell), scene))

The role must not contain the raw color id.  The first executable slice used
three nested signatures: exact normalized shape plus context; context without
shape; and a coarse size/boundary/scene-degree quotient.  Object matches were
accepted only when shape and anchor gave a unique target object, backgrounds
were preserved, and every demo replayed exactly.  Test execution was measured
in two modes: complete coverage is a hard proof gate; partial coverage leaves
unseen roles unchanged and is only a proposal signal.

### Result

`experiments/palette_role_maps.py` and its four regression tests implement the
slice.  On the 1,000-task training split, 30 tasks admitted a conditioned role
map (level 0: 30, level 1: 25, level 2: 3); on the 120-task evaluation split,
only 1 task did.  Partial application emitted 58/1,076 train and 2/172
evaluation candidates, but exact hidden-label recall was 0 in both splits.
Complete role coverage emitted no test candidates.  The candidate is therefore
a falsified standalone channel and is not promoted to a hard solver rule.

### Theoretical conclusion

The role factorization is expressive, but exact conjunctions of local shape,
boundary, and scene context do not identify a role that transfers from demos to
test inputs in this corpus.  The next viable H12 branch is to infer an
equivalence class of roles jointly with the transformation—e.g. object ranks,
relation signatures, and palette permutation constraints—rather than lookup a
single full structural key.  Any such relaxation must retain a counterexample
split: if it maps the same abstract role to conflicting colors across demos,
it must branch into posterior output classes or abstain.

This experiment used no test solutions during fitting and no GPU inference.

## Iteration 37 — Holistic anti-consistency under the Kaggle budget

### New external evidence

The newly inspected ARC-AGI-2 research record strengthens the selection thesis
more than another narrow symbolic primitive.  The MDS paper reports 72.9% on
the verified semi-private set and 76.11% on its public evaluation run.  Its
reported mechanism is 29 heterogeneous candidates followed by three holistic
judges that read full traces together; first and second judge choices receive
2 and 1 points respectively.  The paper's post-hoc comparison attributes +7
solved public instances to holistic selection over majority voting, all from
minority hypotheses, while synthesis added +1.  The same report also states
that 22/39 misses had no correct candidate and 17/39 were selection failures.

These numbers are not a promise for this repository: the paper uses external
frontier APIs and its public run is self-reported.  They do isolate the highest
leverage question for our legal notebook: once a candidate cache exists, can a
local judge or verifier recover minority candidates without sacrificing the
existing NVARC/TRM coverage?

### Formal selection seam

`experiments/judge_aggregation.py` implements the auditable final operation:
each of three ranked pairs contributes weights `(2, 1)` to distinct output
classes, and the two highest-scoring classes are retained.  A vote-only,
position-debiased control is included.  Four tests cover malformed judges,
distinctness, ties, and minority retention.  This is deliberately only the
aggregation theorem; it does not pretend that a local scalar heuristic can
replace a holistic trace judge.

### Budget design for four L4 GPUs

The actionable hypothesis is to reserve a small final judging tranche rather
than spend all 12 hours on correlated generation:

    GPU 0--2: NVARC/TTT candidate generation and verified rescoring
    GPU 3:    TRM candidate generation, then local judge/council
    CPU:      exact grid validation, output-class deduplication, aggregation

The judge input should contain compact demonstrations, candidate grids, and
structured traces (object map, first divergence, invariant checks, and family
id), not every raw token.  It should emit ranked output hashes plus a short
reason; a synthesis candidate is admissible only when it is independently
schema-validated.  Candidate order must be shuffled between judge calls to
control position bias.  The mathematical target remains per-test-output
posterior mass, not one shared program pair, because the competition score is
additive over outputs.

### Falsification gate

Do not spend a Kaggle submission on this until the existing notebook writes a
complete candidate cache.  Replay the same cache with majority vote,
family-calibrated output classes, deterministic structural scoring, and the
three-judge aggregation seam.  Promote the council only if it increases oracle
pass@2 on held-out training folds or recovers correct minority classes at equal
candidate coverage.  If no correct candidates exist, selection cannot help;
that residual is a generation/representation problem and should receive the
next 12-hour tranche.

This iteration used public research and local synthetic tests only; no
competition submission or external API call was made.

## Iteration 38 — Context compression as a theorem, not a prompt tweak

### Constraint

The holistic-judge paper's 30k--80k-token prompt cannot be copied into the
current 8k-context Qwen notebook.  Sending all raw grids and repeated traces
would force truncation precisely on the hard tasks where minority hypotheses
matter.  The judge therefore needs a representation whose compression is
lossless for output identity and lossy only for ranking evidence.

### Construction

`experiments/candidate_cards.py` defines the seam.  First quotient candidates
by exact output hash and retain the lowest-MDL representative.  Then emit a
bounded card containing family, hash prefix, dimensions, palette, inferred
background, object count, occupied cells, input-difference count, weight, and
MDL.  The raw grid stays attached to the representative and is never rebuilt
from the card.  A card token estimate provides a CPU budget before invoking a
local judge.  Two tests prove deduplication, representative selection, and
metadata determinism; the full repository suite is now 152 tests.

On a synthetic 29-candidate 30x30 stress case, exact-class deduplication left
10 classes and reduced a conservative raw-grid character proxy from about
20,010 tokens to 270 card tokens (~74x).  This is a budgeting illustration,
not a performance claim; real candidate diversity must be measured from a
Kaggle cache.

### Theoretical boundary

Cards are not a sufficient statistic for ARC correctness: two grids can share
all listed descriptors while implementing different mechanics.  Therefore the
local judge must receive raw grids and structured traces for a small shortlist
of disagreement classes.  The safe protocol is:

    exact hash quotient -> card shortlist -> raw-grid/trace judge -> hash output

If the shortlist is truncated, preserve at least one representative per solver
family and per high-disagreement structural cluster.  This prevents the
compression layer from silently deleting the only minority hypothesis.  The
card layer can affect ranking only; schema validation and output rendering
remain independent.

## Iteration 39 — Make the 12-hour run replayable

The notebook already writes rich NVARC samples to compressed per-position
files, but the research harness previously normalized only individual samples.
`experiments/cache_adapter.py` now accepts the in-memory decoder shape

    {"task_id_test_index": {"view_or_subkey": sample}}

and converts every valid sample into the common `CandidateRecord` schema.  It
also reports records, covered task positions, exact output classes, and family
counts without reading any solution labels.  Tests cover normalization,
duplicate output classes, malformed keys, and direct bz2/pickle directory
loading; the full suite is now 155 tests.

This is strategically important: one Kaggle run can now be treated as a data
collection experiment.  After the run, selection algorithms, card compression,
judge prompts, family priors, and pass@2 policies can be compared offline from
the same candidate evidence.  A second GPU run is justified only by a measured
generation or coverage change, not by an unobservable end-to-end score delta.

## Iteration 40 — Correct marginalization over augmentation views

### Derivation

Let `g` be a uniformly sampled nuisance transform (D8 geometry plus a legal
color permutation), and let `n_g(y)` be the candidate's teacher-forced NLL
under the transformed task.  The invariant likelihood is

    P(y | task) = (1/|G|) * sum_g exp(-n_g(y))

so the correct log score is `logmeanexp_g(-n_g(y))`.  The notebook's current
mean-NLL score instead ranks the geometric mean of view likelihoods.  These
are not equivalent: mean NLL heavily penalizes one difficult view, whereas the
uniform nuisance model should preserve a candidate that has strong likelihood
on at least one valid view while still accounting for all views.

### Implementation

`experiments/augmentation_scoring.py` adds stable log-mean-exp scoring and
output-class ranking.  `CandidateRecord` now preserves the complete
`score_aug` tuple in `augmentation_nlls`; the old scalar mean remains for
backward compatibility.  Four scorer tests plus the adapter regression pass,
bringing the full suite to 159 tests.

This is a selector hypothesis, not an accuracy claim.  It must be replayed on
the same NVARC cache against `score_kgmon`, arithmetic mean NLL, and
`score_full_probmul_3`.  The promotion criterion is an oracle pass@2 gain on
held-out training folds without reducing candidate-class coverage.  Because
augmentation views share model weights and are correlated, do not multiply
their likelihoods as if they were independent samples; the group marginal is
the intended conservative operation.

## Iteration 41 — Re-anchor the target to the live public leaderboard

The current public leaderboard snapshot puts `nvbanana` at 72.08 and
`rabbithole` at 70.42, while ranks 3--8 are currently in the mid-30s to low
40s.  Kaggle states that the public board is approximately half the test data,
so this is not evidence that the final private top eight will remain in that
shape; it is a calibration point for the user's expected low-70s frontier.
The official competition overview also confirms the non-negotiables for the
design: notebook submission, at most 12 hours, internet disabled, freely
available external data/models allowed, and exactly two predictions per test
output.

The new practical evidence is a Kaggle discussion describing a 67-function
offline transform library verified on training examples.  This validates the
library-plus-proof direction already prototyped here, but it does not imply
that a static library reaches the frontier: the library is a candidate source,
and unseen mechanics still require proposal generation and selection.  The
release must also preserve public availability and licensing for prize
eligibility.

### Updated frontier thesis

The low-70s target is not one miraculous primitive.  It is the product of

    candidate coverage × minority recovery × private generalization.

The first factor is dominated by representation/model diversity; the second
by trace-preserving holistic selection; the third by leakage-safe calibration
and avoiding public-evaluation-specific rules.  The next empirical milestone
is therefore the first full decoder cache, followed by fold replay of the
augmentation marginal, family posterior, candidate cards, and judge council.
No additional Kaggle submission is justified before that replay exists.

## Iteration 42 — Marginal-value allocation of the 12-hour budget

### Derivation

For a family `f` with calibrated effective hit rate `q_f` per additional
candidate, and `n_f` candidates already spent, the independent-candidate model
gives

    P(next candidate discovers a new correct class) = q_f(1-q_f)^n_f.

With cost `c_f` seconds, the greedy anytime priority is this marginal gain
divided by `c_f`.  The rate must be estimated from held-out task folds after
deduplicating correlated samples; raw view count is not a valid `q_f`.

`experiments/compute_allocator.py` implements the bounded greedy planner and
four tests.  It respects a hard wall-clock budget, applies geometric decay to
repeated same-family proposals, and naturally switches to a slower high-hit
family when its gain per second becomes better.  The suite is now 163 tests.

### Kaggle mapping

Use the first cache to estimate, per family and structural task group:

    q_f = unique newly solved output classes / eligible task positions
    c_f = wall time per eligible candidate

Reserve a fixed CPU/write buffer and treat incomplete positions as zero
coverage.  During a run, route cheap structural checks first, then allocate
remaining GPU time to the family with the largest posterior lower-confidence
gain per second.  If the lower confidence bound falls below zero after
correlation correction, stop that family and preserve time for an independent
proposal family or holistic selection.  The formula is a planning model, not a
hard guarantee; it is promoted only after fold replay shows no coverage loss.

The point estimate is too optimistic during cold start, so the allocator now
also exposes the exact Beta-posterior expectation

    E[q(1-q)^n] = B(alpha+1, beta+n) / B(alpha, beta).

`greedy_posterior_plan` ranks this uncertainty-integrated gain per second and
supports explicit candidate caps.  Four additional tests cover the posterior
identity, evidence/cost interaction, and invalid parameters.  The full suite
is now 166 tests.  This is the preferred planner for the first cache; after
enough observations it converges toward the point-rate rule while remaining
less sensitive to lucky early folds.

## Iteration 43 — Private-distribution-aware calibration

### Problem

The public evaluation set is not the private test set.  The local structural
audit already found sizeable input-only distribution differences between them,
so a family that wins on public evaluation can be misallocated on the hidden
test.  The hidden challenge inputs are available before submission, while
their outputs are not; this permits covariate calibration without label leak.

### Estimator

Partition tasks by challenge-visible structural group `g`.  Under the
covariate-shift assumption `P(success | g, family)` is stable, estimate the
private target rate as

    sum_g P_private(g) * E[p_success(family,g) | training folds].

`experiments/domain_adaptation.py` implements Dirichlet-smoothed group
frequencies, capped target/source importance weights, groupwise Beta posteriors,
pooled fallback for unseen groups, and Kish effective sample size.  Five tests
cover normalization, shifted weights, unseen-group fallback, target weighting,
and weight concentration; the suite is now 171 tests.

### Safety gate

This is not a license to tune against hidden outputs: only test-input features
may form the target distribution.  If importance weighting collapses effective
sample size or a group has no labeled support, use the pooled posterior and
report the uncertainty.  Calibrated family priors should be selected by
leave-group-out training folds, then evaluated once on public evaluation; the
hidden input distribution only determines deployment weighting.  Promotion
requires improvement in held-out fold estimates and no catastrophic
worst-group regression.

### Data audit

On the checked-in input files, a composite key formed from all current
`task_features` produced 403 training groups and 156 hidden-test groups.  With
capped importance ratios (cap 5), the resulting Kish effective sample size was
294.87, maximum realized ratio 2.72, and every hidden group had at least one
training analogue.  However, public-evaluation versus hidden-test composite
TV was 0.8125 with only 26 shared groups.  This confirms that the composite key
is too granular for direct family calibration even when train-to-hidden support
exists.  The next version must use hierarchical/featurewise pooling and treat
the 0.8125 figure as a sparsity warning, not as a claim about semantic task
shift.

## Iteration 44 — Hierarchical pooling for sparse private groups

The composite-key audit implies a partial-pooling model.  Let `p_0` be the
global family success rate and `p_{j,v}` the rate for one feature/value pair.
Estimate the deployment rate in log-odds space as

    logit(p_hat) = logit(p_0) + mean_j lambda_j[logit(p_{j,v}) - logit(p_0)]

where `lambda_j = n_{j,v}/(n_{j,v}+kappa)`.  A rare feature therefore has a
small correction, while repeated evidence can move the prediction away from
the global prior.  This avoids the high-variance 403-way lookup while keeping
input-only hidden-test composition useful.

`experiments/hierarchical_calibration.py` implements Beta-smoothed global and
feature posteriors, shrinkage corrections, bounded sigmoid output, and missing
feature fallback.  Five tests and the full suite (176 tests) pass.  This is a
calibration component only: feature definitions and `kappa` must be frozen by
leave-group-out training folds, and any deployment gain must be reported with
effective sample size and worst-group uncertainty.

## Iteration 45 — Fold calibration on candidate-class coverage

The calibration target must be family capability, not raw vote volume.  Given
records for one family and one task-position, first deduplicate by exact output
hash, then count one success if any class equals the labeled training output.
This produces the Bernoulli observations required by Iterations 42--44 and
prevents sixteen correlated augmentation views from becoming sixteen wins.

`experiments/fold_calibration.py` implements this aggregation and predicts
target-set family rates through the hierarchical featurewise posterior.  It
has three tests covering duplicate-view collapse, wrong-class failures, and
prediction on unlabeled target inputs; the full suite was 179 tests at this
stage.

The resulting end-to-end calibration chain is now explicit:

    decoder cache -> output-class coverage -> fold/group posterior
                   -> hidden-input reweighting -> cost-aware allocation

No calibration number is reported yet because the repository contains no
actual 12-hour decoder cache.  Once one exists, this chain is the first replay
to run before modifying the notebook or consuming another Kaggle quota tranche.

## Iteration 46 — CEGIS version-space control for the verified library

### Theorem and design

Let `H(D)` be the finite set of executable programs that reproduce every
demonstration in task `D`.  Any program outside `H(D)` has posterior probability
zero under the exact-demo likelihood, so retaining it can only dilute the
posterior of the task's possible outputs.  For a test input `x`, define the
output quotient

    q_x(h) = execute(h, x)

and aggregate posterior mass over equal values of `q_x`.  Since the Kaggle
objective is additive across test outputs and accepts two distinct guesses,
the Bayes action for each output is the two largest quotient masses.  Syntactic
program multiplicity is not evidence: candidates are normalized within a
solver family before family priors are applied.

`experiments/cegis_version_space.py` implements this control layer.  It offers
an exact hard-gate filter, a conservative next-demo discriminator
(`n - largest prediction bucket`), family-normalized output posteriors, and
independent top-two actions for multi-test tasks.  The proof obligation is
explicit: a library entry must carry an executor and reproduce all demos before
it can affect a prediction.  This is the part a static 67-function library
cannot provide by itself when several transforms remain observationally
equivalent on the demonstrations.

### Falsification tests

Five CPU-only adversarial tests pass.  They cover exact elimination by a
counterexample, maximum-partition demo choice, quotienting syntactic
duplicates, protection against a ten-copy correlated-vote flood, and the
official independent top-two action.  The complete repository suite is now
184 tests.  This is a theoretical/infrastructure positive, not a score claim;
the open empirical question is whether a sufficiently broad offline library
can populate the version space on ARC-AGI-2 eval without a hosted model.

### Deployment consequence

The first 12-hour run should spend model or CPU time on *new equivalence
classes* rather than duplicate programs.  As soon as a candidate fails one
demo, CEGIS removes its entire syntactic descendant family.  At decode time,
store only `(program_id, family, demo_certificate, output_hash, score)` and
reconstruct the two outputs from quotient masses.  This gives the holistic
judge a compact, proof-carrying candidate set while preserving minority
classes that a raw vote would erase.

## Iteration 47 — Occam/PAC-Bayes gate for overfit explanations

Exact demo consistency is necessary but not sufficient.  A sufficiently
expressive library can contain many programs that interpolate the few demos,
especially when the program was generated by a language model.  Under the
explicit assumption that demonstrations are conditionally sampled from the
same task rule and that programs have a prefix-free code, the standard Occam
union bound gives, for candidate `h`,

    true_error(h) <= empirical_error(h)
                     + sqrt((L(h) ln 2 + ln(2/delta)) / (2 n))

where `L(h)` is description length in bits and `n` is the number of demos.
The result is not a claim that ARC's examples are iid; it is a transparent
regularizer and a diagnostic for when the assumption is carrying too much
weight.

`experiments/occam_gate.py` implements the bounded penalty, candidate score,
and deterministic ranking.  Four tests pass: finite-sample shrinkage,
short-vs-long exact explanations, empirical failures overriding equal
complexity, and invalid-observation rejection.  The full suite is now 188
tests.

### Falsification boundary

The bound becomes vacuous for one or two demos, and it can be invalid when
ARC examples are deliberately selected to expose a rule rather than sampled
randomly.  Therefore it must never delete the only surviving candidate.  The
safe use is triage: short certified programs receive more judge/context budget,
while high-complexity interpolants remain in the second-pass pool only if an
independent family supports their output.  This yields a concrete three-way
separation: hard demo proof, complexity-based prior, and holistic selection.

## Iteration 48 — Typed cost-ordered composition as the search substrate

### External architectural evidence

The public `aicpp` DSL-engine branch describes a deterministic symbolic engine
whose core is typed primitive composition, bounded search depth, cost ordering,
explicit symbolic solutions, and reusable structural memory.  The associated
Kaggle discussion confirms the practical packaging route: a compiled library
can be shipped as a dataset and used in an offline notebook, subject to the
competition's license and runtime rules.  This is consistent with the current
campaign's no-finetuning constraint and with the 67-function offline-library
evidence, but it is not evidence of a leaderboard score.

### Formal search reduction

For a unary typed DSL, let `E_t(k)` be the number of reachable expressions of
type `t` and cost at most `k`.  Without typing, a depth-`d` search grows like
`|P|^d`.  With type signatures, only edges whose input type equals the current
output type are legal, so the recurrence is

    E_t(k) = 1[t = start] +
             sum_{p: out(p)=t} E_in(p)(k - cost(p)).

The recurrence is an exact count for this restricted grammar and makes the
pruning mechanism measurable before any grid execution.  It also exposes a
failure mode: an expressive but weakly typed DSL merely moves the explosion
into the type graph.  Types must therefore encode semantic roles (grid,
object-set, object, mask, color, integer), not only container shapes.

`experiments/typed_search.py` implements deterministic increasing-cost unary
path enumeration with explicit type and cost bounds.  Four tests pass for
ill-typed pruning, cost order, duplicate symbolic paths, and invalid bounds;
the full suite is now 192 tests.

### Deployment prescription

Precompile a typed primitive graph outside Kaggle, ship the shared object and
its license metadata as a dataset, then use the 12-hour notebook for
task-conditioned parameter binding and exact demonstration execution.  Keep
the library's output as symbolic traces, not just grids: traces provide the
CEGIS counterexample location, MDL cost, and candidate-card explanation.  Use
the four L4s for independent proposal families or judge passes; do not spend
GPU time enumerating type-invalid programs that the CPU grammar can reject.

## Iteration 49 — Four-GPU schedule as a coverage/recovery optimization

The official metric reduces the deployment objective to expected solved output
count, not number of generated samples.  Let `C_f(t)` be the expected number
of *new output classes* produced by family `f` after compute `t`, and let
`S(t_sel)` be the probability that the selector recovers the correct class
from the accumulated candidates.  A useful approximation is

    E[score numerator] ~= sum_outputs P(correct class covered)
                           * P(selector retains it | covered)

This makes raw sample count an invalid optimization target: correlated
augmentations can increase `C_f` by almost zero and can even hurt `S` by
flooding the context with duplicates.

### Recommended 4xL4 allocation

Use four complementary lanes, with a CPU proof/serialization process running
continuously:

1. a high-throughput neural proposal lane with temperature/augmentation
   diversity;
2. a second neural or checkpoint lane selected for error complementarity;
3. a deterministic typed-library/DSL lane for short verified programs and
   counterexample-guided repair;
4. a late-stage holistic judge/repair lane, fed only compressed candidate
   cards, with enough time reserved for the final output quotient.

The exact proportions should be chosen by the Beta-posterior marginal-gain
planner from Iteration 42 after the first cache.  A safe cold-start schedule
reserves roughly the final tenth of wall time for selection, validation, and
serialization; it never lets proposal decoding consume the submission buffer.
If a lane's lower credible marginal gain per second falls below another lane's,
move only the next tranche of time, preserving already-produced candidates.

### Why this is the moonshot constraint

The MDS result indicates that heterogeneous candidate generators plus holistic
judging can recover minority correct answers that majority voting misses.  The
Kaggle setting removes remote judge calls, so the equivalent must be compiled
into the notebook: proof-carrying cards, local deterministic judges, and a
family-aware posterior.  This turns the 4-GPU budget into a sequential decision
problem rather than a batch-size contest.  No schedule claim is promoted to a
score until a real decoder cache shows (a) new-class coverage, (b) judge
recovery on held-out folds, and (c) no private-distribution collapse.

## Iteration 50 — Adaptive early stopping as value of information

### Motivation from the strongest disclosed result

The MDS paper's candidate pipeline generates up to 29 candidates in stages
and stops early when agreement is strong.  It also reports that the expensive
selection step is where minority hypotheses can be recovered.  Its explicit
future-work recommendation is adaptive routing: spend expensive modalities
only on uncertain tasks.  This is directly compatible with the 12-hour
notebook, where every second spent on a redundant candidate competes with
judge and serialization time.

### Derivation

For output position `j`, let `p_j` be the current posterior over exact output
classes.  The Bayes utility of submitting two outputs is

    U_j = max_{a != b} p_j(a) + p_j(b)
        = p_j^(1) + p_j^(2).

The unresolved mass is `1 - U_j`.  If lane `f` has calibrated probability
`q_f` of discovering a useful new class per eligible position and the
selector retains that class with probability `r_f`, then a first-order
expected gain for a lane of cost `c_f` is

    Delta_f / c_f = q_f r_f * sum_j (1 - U_j) / c_f.

The sum is intentional: ARC's official score is output-weighted, so a
two-test task contributes two opportunities rather than one.  Use entropy or
effective class count only as diagnostics; entropy alone is not the stopping
criterion because a diffuse posterior can still have its mass concentrated in
two classes.

`experiments/adaptive_routing.py` implements posterior entropy, effective
class count, current top-two utility, unresolved mass, expected lane gain, and
a thresholded continue/stop decision.  A follow-up falsification found that a
candidate-only posterior could falsely assign certainty to a wrong consensus.
The module now accepts a position-specific unknown/uncovered reserve, which is
not selectable but contributes to uncertainty and routing value.  Seven
synthetic tests pass; the full suite is now 204 tests.

### Safe deployment rule

Estimate `q_f` and `r_f` from deduplicated, leakage-safe training folds, with
featurewise private-input calibration from Iterations 43–45.  Do not infer
them from the hidden answers or from a single lucky public submission.  Run a
cheap lane first; after each tranche, recompute output-class posteriors and
allocate the next tranche to the largest credible `Delta_f/c_f`.  Stop only
when every remaining lane is below the reserved-buffer threshold.  If the
posterior is empty or malformed, route to a fallback proposal lane rather than
silently emitting no candidate.

## Iteration 53 — Submodular allocation for complementary families

### Problem with independent decay

The Iteration 42 allocator models repeated samples with a geometric decay,
but it does not distinguish a second checkpoint that repeats the first one's
errors from a modality that covers different task positions.  The MDS analysis
reports persistent exclusive coverage between text, image, and code families,
so family choice should depend on residual coverage after prior lanes.

### Coverage surrogate

For held-out task positions `i`, let `q_{f,i}` be the calibrated probability
that proposal lane `f` contributes a useful new output class.  For a selected
set `S`, use

    F(S) = sum_i [1 - product_{f in S}(1 - q_{f,i})].

This is monotone submodular under the conditional-independence surrogate.  The
greedy marginal for lane `f` is

    Delta_f(S) = sum_i q_{f,i} product_{g in S}(1 - q_{g,i}),

and the scheduling priority is `Delta_f(S)/cost_f`.  A duplicate lane loses
priority automatically; a complementary lane retains gain on positions that
the selected set leaves uncovered.  When measured conditional rates are
available, they should replace the product approximation without changing the
allocator interface.

`experiments/submodular_allocator.py` implements validation, expected
coverage, residual marginal coverage, and deterministic budgeted greedy
selection.  Four tests pass for diminishing duplicate gain, complementary
lane preference, budget/coverage behavior, and invalid rates/shapes.  The
full suite is now 208 tests.

### Deployment consequence

Calibrate `q_{f,i}` by structural group and output position from fold replay,
not by total sample count.  Treat each checkpoint/modality/temperature bundle
as a separate proposal lane.  Reserve the judge lane as a downstream cost in
the same knapsack, because additional coverage has no value if it cannot be
selected into the two official outputs.  This gives a principled reason to
run four meaningfully different lanes on four L4s instead of four replicas of
the same sampler.

## Iteration 52 — Candidate-set coverage is a separate latent variable

The unknown-reserve correction is more than an implementation detail.  Let
`K` be the observed output classes and let `u_j = P(y_j not in K)` be the
coverage-failure probability for output position `j`.  If the observed class
posterior is `p_j`, the selectable mass is `(1-u_j)p_j`, so the official
pass@2 utility is

    U_j = (1-u_j) * (p_j^(1) + p_j^(2)).

Even perfect agreement (`p_j^(1)=1`) yields only `1-u_j` utility.  This
separates two failure modes that are often conflated:

1. ranking failure — the correct class is present but not in the top two;
2. coverage failure — no generated candidate represents the correct class.

The first is addressed by holistic judging and family-aware selection; the
second requires another proposal family or targeted synthesis.  Entropy over
the observed classes cannot detect the second failure, so it is only a
diagnostic.  The adaptive router now carries `u_j` explicitly and adds it to
unresolved mass, preventing early-stop errors on a unanimous but incomplete
candidate pool.

### Calibration prescription

Estimate `u_j` on held-out training folds as the rate at which the correct
output class is absent after a specified budget/family mix.  Fit it by
structural group with hierarchical shrinkage, then use only challenge-visible
features to reweight toward the hidden test distribution.  Do not set `u_j=0`
because a single model produced one answer; that assumption is exactly what
the MDS generation-failure table falsifies.  If no calibration exists, use a
conservative floor and report the resulting extra routing cost rather than
silently converting model agreement into proof.

## Iteration 51 — Gated synthesis of complementary partial traces

### Motivation

The MDS results report a small positive contribution from synthesis and
identify targeted synthesis as an open opportunity.  Synthesis is dangerous
on ARC because two fluent explanations can each fit the demos while their
combination invents an untested rule.  The correct abstraction is therefore
constraint conjunction, not text blending.

### Safety rule

Represent each candidate trace by typed partial clauses `C_i`.  Two traces are
eligible for synthesis only if they come from different families, have no
conflicting assignments on shared clause keys, and have enough disjoint
constraint support.  Their proposed synthesis is the compatible union

    C_syn = C_i union C_j,   if C_i restricted to overlap = C_j restricted to overlap.

The union is merely a proposal.  It must pass the same CEGIS exact replay and
frame/role proof gate as every other program.  If a hard-verified candidate
already exists, or the current top-two posterior leaves too little unresolved
mass, synthesis is skipped; this prevents unnecessary mutations and anchoring.

`experiments/synthesis_gate.py` implements compatible constraint union,
Jaccard-style clause complementarity, deterministic cross-family pair choice,
and the uncertainty/verification gate.  Five adversarial tests pass for
compatible union, conflict rejection, complementarity, positive cross-family
selection, and verified/resolved stopping.  The full suite is now 202 tests.

### Deployment consequence

The late fourth lane should not ask a model to “try another answer” globally.
It should receive two compressed cards, expose their clause intersection and
union, and be prompted to synthesize only the union.  If the union fails one
demo, the failure is attributable to a specific clause and can be repaired or
discarded.  This converts synthesis from an unconstrained third guess into a
bounded search operator with a measurable acceptance certificate.

## Iteration 54 — Quotient arbitrary object IDs before correspondence search

### Observation

The correspondence audit found tied best assignments on 27.44% of processed
training object pairs and 39.68% of processed evaluation pairs. Some ties are
real automorphisms, but some arise because an object-centric representation
has arbitrary node names. Treating every naming permutation as a distinct
hypothesis wastes the exact-search budget and increases the apparent posterior
ambiguity.

### Construction

For a labeled scene graph `G=(V,E)`, define its semantic quotient as the
equivalence class under node permutations that preserve node and edge labels.
For small `|V|`, an exact canonical code is the lexicographically least
labeled adjacency code over all permutations. For larger graphs,
Weisfeiler--Leman refinement repeatedly replaces a node color with its current
color plus the multiset of labeled neighbor colors. The refined color
multiset and edge colors are permutation-invariant, although WL is not a
complete graph-isomorphism test.

`experiments/graph_canonicalization.py` implements both paths: exact canonical
labeling up to a node bound and deterministic WL signatures above it. Four
tests pass for ID-invariance, separation of small path/triangle graphs, larger
graph permutation invariance, and malformed graph/bound rejection. The full
suite is now 212 tests.

### ARC consequence

Use canonical graph signatures to cache scene parses and to partition
correspondence assignments into semantic role cells before running top-k DP.
Do not collapse nodes that remain indistinguishable after refinement: those are
genuine automorphisms and must remain multiple candidates until a relation,
effect, or output quotient resolves them. This is a safe search reduction,
not a new solver primitive, and it should be measured by exact assignment
counts and unique output classes on held-out folds.

## Iteration 55 — Semantic posterior quotient for correlated decoder copies

### Observation

Family normalization prevents a large decoder family from flooding the
posterior, but it does not remove multiplicity within that family. Ten
near-identical programs that emit the same wrong grid can still look like ten
votes. This is especially dangerous when a language model samples temperature
variants or paraphrases one latent program.

### Construction

For each family `f` and test input `x`, partition survivors by exact predicted
output `y`. Replace each partition by one semantic class whose weight is the
maximum program prior in that partition, using `2^(-MDL)` as an optional
prefix-code factor. Normalize classes within the family, then apply the family
prior. The ordinary posterior remains available because genuinely independent
derivations should not automatically be collapsed.

`experiments/cegis_version_space.py` now exposes
`semantic_posterior_outputs` and a `collapse_correlated` switch on
`cegis_solve`. Two adversarial tests show that ten same-family copies no longer
outvote one distinct output class and that the shortest/best-prior
representative controls the class weight. The full suite is now 214 tests.

### ARC consequence

Use the semantic quotient for samples from one checkpoint, prompt, or decoder
trace cluster; retain independent families and genuinely different predicted
grids. This turns pass@2 selection into a posterior over output equivalence
classes rather than a vote over strings. The switch must be driven by
provenance, because collapsing truly independent derivations would throw away
useful evidence.

## Iteration 56 — Capture--recapture reserve for unseen output classes

### Observation

Observed top-two mass and entropy are conditional on the classes a proposal
panel happened to emit. They cannot detect a correct class absent from that
panel. Two independent proposal panels provide a weak but explicit coverage
diagnostic: overlap estimates how much of their shared class population they
have recaptured.

### Construction

Deduplicate each panel by exact output grid. Let `n1`, `n2` be the unique class
counts and `m` their overlap. Under approximately independent equal-probability
sampling from a shared finite class population, use the Chapman estimate
`N_hat = (n1+1)(n2+1)/(m+1) - 1`; the unknown class fraction is
`max(0, N_hat - |A union B|)/N_hat`. With empty or disjoint panels, return an
unreliable full reserve rather than false precision. `reliable` requires at
least two recaptures.

`experiments/unknown_mass.py` implements this diagnostic and four tests cover
duplicate removal, full overlap, disjoint panels, and nested-grid canonicalization.
The full suite is now 218 tests.

### ARC consequence

Feed the estimate as a capped/shrunk unknown mass `u_j` into the adaptive
router, not as a literal probability of correctness. High overlap permits
early stopping only when the known top-two posterior is also strong; disjoint
panels should trigger a complementary proposal lane. Because ARC output
classes are highly nonuniform and panels are not iid, this is a routing signal
that must be calibrated on held-out folds, never a leaderboard claim.

## Iteration 57 — Put the semantic quotient on the production selector seam

### Observation

The CEGIS semantic posterior is useful only if the final submission selector
uses the same independence model. Otherwise a later aggregation layer can
reintroduce the exact copy-count failure it was designed to remove.

### Construction

`experiments/pass2_selector.py` now accepts `collapse_correlated` for scalar
output selection, coherent task-vector selection, and official independent
per-test-output selection. In collapsed mode, each family/output (or
family/vector) cell contributes the maximum prior-weighted MDL representative;
in ordinary mode, weighted multiplicity remains available for genuinely
independent candidates. A regression test makes the distinction visible: a
20-to-1 decoder-copy imbalance changes the baseline ranking but cannot remove
the distinct semantic class from the collapsed top-two set. The full suite is
now 219 tests.

### ARC consequence

Production provenance should label candidates by checkpoint, prompt lineage,
temperature branch, and repair trace. Collapse within a correlated lineage,
then aggregate across independent families before selecting two exact output
classes per test input. This matches the additive Kaggle metric while keeping
the second attempt as genuine semantic coverage.

## Iteration 58 — Live-source constraint refresh

### Finding

The live search did not reveal a reproducible public 70%+ Kaggle recipe. The
current official rules still require competition code/data sharing if shared,
and constrain the notebook execution environment; the ARC guide describes the
evaluation as constructing outputs from demonstrations, with a 120-task
private evaluation reference. A community discussion also illustrates that
individual task interpretations can be disputed, so exact replay certificates
and explicit abstention are safer than silently forcing a rule.

### Decision

Do not spend a submission or a 12-hour GPU run on an unverified public claim.
The current moonshot design remains: heterogeneous proposal generation,
proof-gated deterministic replay, semantic output quotienting, unknown-class
reserve, and late holistic selection/repair. Any future Kaggle run should
measure coverage and selector recovery separately on the fixed shadow/dev
folds before consuming the limited submission schedule.

## Iteration 59 — Correlation groups inside a generator family

### Observation

The previous binary choice—collapse a whole family or preserve all samples—is
too coarse. A family can contain several independent checkpoints, while each
checkpoint can emit many correlated temperature/paraphrase variants. Treating
the entire family as one correlation unit either overcounts or over-collapses.

### Construction

`Candidate` and `TaskCandidate` now carry an optional `correlation_group`.
Collapsed selection first takes the best prior-weighted MDL representative per
output class within each group, applies the existing smoothing, and then
averages the resulting distributions across groups. The same operation is
implemented for complete task vectors before deriving the official independent
per-test marginals. Omitting the field preserves the one-lineage baseline.

Three new selector tests cover copy-count removal, independent-lineage
preservation, and independent output pairing. The full suite is now 221 tests.

### ARC consequence

The notebook should assign groups at generation time, for example
`checkpoint/prompt/temperature/repair-chain`, and persist them in candidate
cards. Group IDs must be semantic provenance, not output hashes. The selector
then estimates evidence at the lineage level: one checkpoint cannot buy 20
votes, but two distinct checkpoints can move a class posterior. This is the
most direct implementation of diversity as effective sample size under the
four-L4 constraint.

## Iteration 60 — Preserve correlation metadata through the replay boundary

### Observation

Correlation-aware selection is invalid if candidate normalization discards the
lineage metadata before aggregation. The existing record schema carried family,
weights, and MDL but had no durable field for checkpoint/prompt/repair lineage.

### Construction

`CandidateRecord` now stores an optional `correlation_group`, serializes it in
append-only records, and propagates it through selector candidates and complete
task vectors. The NVARC adapter accepts an explicit group from the sample
metadata; the Leg-C adapter groups a verified program and its alternate by the
program identity. Vector adaptation preserves a group only when every test
position agrees, otherwise it abstains from inventing provenance. Four new
tests cover scalar adaptation, vector adaptation, NVARC metadata, and the full
suite, which now has 224 passing tests.

### ARC consequence

The notebook's candidate cache must persist lineage fields alongside the output
hash. Recommended group key:
`<checkpoint>/<prompt-template>/<temperature-branch>/<repair-chain>`. A missing
group means “unknown correlation,” so deployment should choose the conservative
single-lineage collapse policy rather than silently treating samples as iid.
This closes the last metadata-loss path between generation and pass@2 output.

## Iteration 61 — Separate invariant search buckets from merge-safe graph keys

### Observation

The Iteration 54 WL signature is invariant under object-ID renaming, but WL is
not a complete graph-isomorphism test. Two non-isomorphic scene graphs can
therefore share a WL signature. Using that signature directly to merge cached
scene parses would create a silent false-positive correspondence transfer.

### Construction

`experiments/graph_canonicalization.py` now exposes
`collision_safe_cache_key`. It returns the exact canonical signature below the
permutation bound and returns `None` for WL-only graphs. The ordinary signature
remains available for cheap routing/bucketing; callers must perform exact
isomorphism or retain separate entries before merging a WL bucket. Two tests
cover the safe/unsafe boundary; the full suite is now 226 tests.

### ARC consequence

Use WL to choose which structural solver lane or correspondence bucket to
visit, never as proof that two large scenes are equivalent. Cache exact small
graphs by canonical key; for larger graphs cache a list under the WL bucket
and require a second-stage label/edge certificate before reuse. This costs a
small amount of CPU only on bucket collisions and protects every downstream
role/effect hypothesis from an invalid quotient.

## Iteration 62 — Carry the correlation quotient through final replay

### Observation

The selector supported lineage-aware collapse, but the replay harness still
used its historical baseline call. This made the new theory easy to omit at
the exact point where candidate classes become Kaggle attempt slots.

### Construction

`build_submission` and `replay_score` now expose `collapse_correlated` and pass
it through to scalar pass@2 selection. An end-to-end regression constructs a
20-copy decoder flood, one same-lineage alternative, and one independent
class: baseline replay misses the truth, while collapsed replay retains it
and receives the exact score. The full suite is now 227 tests.

### ARC consequence

The production notebook must choose the correlation mode once per cache and
record that choice in its run manifest. For model samples with known lineage,
call `replay_score(..., collapse_correlated=True)` (or the equivalent notebook
path); use baseline mode only for an explicitly independent sample design.
This closes the final submission-boundary mismatch between the research
posterior and the emitted two-attempt JSON.

## Iteration 63 — Measure coverage failure separately from selector failure

### Observation

The central objective decomposes into two different probabilities, but the
existing replay score exposed only their product: (1) the correct output class
is present in the candidate set, and (2) the selector retains it in one of two
attempts. A selector change cannot repair missing coverage, and a generator
change can look useless if selection is simultaneously poor.

### Construction

`experiments/coverage_recovery.py` adds a labeled-fold evaluator that reports
output-weighted candidate coverage, conditional selector recovery, exact
output score, fully-covered-task rate, and solved-task rate. It deduplicates
only exact hard-valid output classes, uses the same selector and correlation
mode as replay, and ignores records outside the labeled fold. Five tests cover
empty coverage, perfect recovery, covered-but-not-selected truth, collapsed
recovery, and output-vs-task weighting. The full suite is now 232 tests.

### ARC consequence

Every candidate-generation or judge experiment must report the tuple
`(coverage_rate, selector_recovery_rate, output_score)` on non-shadow folds.
Promote a new lane only when it adds truth-class coverage or improves
conditional recovery at equal compute; do not promote on plausible-looking
outputs or task-weighted averages. Keep the shadow fold untouched until a
milestone, then use it once as the final guard before spending a Kaggle run.

## Iteration 64 — Promotion gate for coverage/recovery evidence

### Observation

The research now has the right metrics, but a future experiment could still
be promoted on a noisy end-to-end score, an unequal fold sample, or an
unrecorded 12-hour configuration. That would reintroduce the testing and
selection bias the loop is designed to avoid.

### Construction

`experiments/promotion_gate.py` adds `RunManifest` and `decide_promotion`.
The manifest validates the four-GPU/12-hour boundary, safety buffer, selector
mode, fold mode, code revision, and model artifacts. The gate requires equal
development sample sizes, no coverage/recovery/score regression, and at least
one genuine coverage or conditional-recovery gain; a score-only gain is
rejected. A milestone can additionally require explicit shadow-fold
verification. Five tests cover these invariants; the full suite is now 237
tests.

### ARC consequence

The promotion protocol is now:

1. freeze the run manifest and candidate cache provenance;
2. evaluate baseline and candidate on identical non-shadow folds;
3. report coverage, recovery, output score, and task metrics;
4. promote only through `decide_promotion`;
5. touch fold 4 once at a milestone, then decide whether the scarce Kaggle run
   is justified.

This is the stopping rule for experiments, not a reason to stop the research
loop: it prevents spending a 12-hour run on an unidentifiable change.

## Iteration 65 — Proof-guided evolutionary program induction

### Observation

The current typed search is exhaustive only for a narrow unary grammar, while
the planned 70%-regime path requires broader program coverage. Free-form
evolution is unsafe: it can trade exact demonstration replay for plausible
code and spend the 12-hour budget exploring irrelevant mutations.

### Construction

`experiments/evolutionary_induction.py` defines immutable operation genomes,
deterministic one-edit insert/delete/replace mutations, first-divergence-local
guided mutations, exact demonstration fitness, MDL cost, and a Pareto frontier
that maximizes correct demos while minimizing description length. Parents are
included in the competition but dominated parents may disappear. Five tests
cover bounded deterministic mutation, local causal edits, exact fitness,
accuracy/complexity tradeoffs, and recovery of an exact mutant. The full suite
is now 242 tests.

### ARC consequence

Use evolution only after a model or DSL seed proposes a genome. Run a cheap
generation as follows: execute the parent on demonstrations, locate the first
divergence in its trace, mutate only the responsible operation when possible,
replay every child exactly, and retain the Pareto frontier. Fully exact
children become CEGIS candidates; partial children remain repair hypotheses,
never direct test outputs. This makes program evolution a coverage lane rather
than an unbounded third guess.

### Limitation

The module proves search-control properties, not ARC accuracy. Deployment
still needs an AST/DSL sandbox, a complete primitive library, per-program
timeouts, and held-out coverage/recovery evidence before promotion.

## Iteration 66 — Make evolutionary search bounds executable

### Observation

An evolutionary lane can still violate the intended compute budget if a
seeded genome already exceeds the configured maximum operation count. In a
12-hour competition notebook, a descriptive bound is not enough: the search
must reject an over-budget seed before generating children.

### Construction

`mutate_genome` now rejects any genome whose operation count exceeds
`max_steps`; all insertions, replacements, and deletions therefore remain
within the declared bound. The existing five evolution tests still pass, and
the full suite remains 242 tests green.

### ARC consequence

Each task's evolution lane should derive `max_steps` from the remaining wall
clock and the typed primitive cost table, reject oversized model-produced
programs immediately, and serialize the bound in the run manifest. This makes
the proposal allocator's cost estimate an enforceable invariant rather than a
post-hoc explanation.

## Iteration 67 — Robust pass@2 selection under uncertain family priors

### Observation

The selector's family mixture is estimated from sparse folds and may be badly
miscalibrated on the private distribution. A point prior can suppress a
minority lineage even when the evidence supports a broad interval of plausible
mixtures.

### Construction

`experiments/robust_selector.py` represents each active family weight as an
interval `[l_f,u_f]` with a feasible simplex constraint. For any output pair,
the expected covered mass is linear in family weights. Its exact worst case is
therefore obtained by starting at all lower bounds and allocating the residual
mass to the families with the smallest pair support; the best pair is the
one with maximum worst-case mass. The module also reports per-class lower and
upper masses and applies correlation collapse before mixture optimization.
Four tests cover fixed mixtures, exact worst-case pair selection, infeasible
interval rejection, and lineage collapse. The full suite is now 246 tests.

### ARC consequence

Use point-prior selection when fold calibration is dense and stable; otherwise
run robust selection as a conservative ablation. A pair should survive only if
its worst-case class mass remains competitive, which protects attempt 2 from a
single overconfident family estimate. Interval widths must be learned from
held-out folds and never tuned on the shadow fold.

### Limitation

The current robust API handles scalar output classes. Extending the same linear
optimization to task vectors is straightforward only after deciding whether
the uncertainty is shared across positions; the official additive metric still
requires independent per-position marginal actions.

## Iteration 68 — Joint robust selection for multi-test tasks

### Observation

The scalar robust selector minimized family-prior uncertainty separately for
one output. For a task with multiple test inputs, the unknown family mixture is
shared across positions. Taking independent per-position worst cases can be
strictly more conservative than the actual additive objective and can choose a
suboptimal pair tuple.

### Construction

`experiments/robust_selector.py` now accepts complete `TaskCandidate` vectors.
For each position it enumerates the legal one/two-class actions, computes each
family's additive coverage across positions, and applies the interval-simplex
linear minimization once to the entire action tuple. Enumeration is exact up
to an explicit `max_joint_actions` cap; exceeding that cap raises instead of
silently claiming exactness. Two tests cover shared-prior optimization and
explicit action-space failure. The full suite is now 248 tests.

### ARC consequence

Use this robust vector mode only after vector candidates are complete and the
number of output classes is small enough for the declared cap. Otherwise use
the ordinary independent marginal selector with a calibrated point posterior,
and log the approximation. The official additive metric still permits
different output pairs at different positions; the robust improvement is that
their uncertainty is coupled correctly before ranking.

## Iteration 69 — Static resource budgets for generated programs

### Observation

The existing subprocess sandbox bounded CPU, memory, file size, and open file
descriptors, but a generated source could still be pathologically large or
AST-heavy before execution. A blacklist alone does not make an evolutionary
program lane predictable under the 12-hour notebook budget.

### Construction

`kaggle_nemotron_probe/probe_core.py` now rejects source above 32,000
characters or an AST above 6,000 nodes before compilation. The Linux resource
limits and subprocess timeout remain the runtime backstop. Two tests cover
source and AST exhaustion while the existing valid candidate and unsafe-import
tests remain green; the full suite is now 249 tests.

### ARC consequence

Use the same static budgets for any evolution/Leg-C verifier: parse, validate,
count nodes, then execute only demo candidates that pass both the safety and
exact-replay gates. The limits should be recorded in the run manifest and
calibrated against the largest accepted programs on development folds; they
must not be tuned on hidden labels.

### Limitation

AST validation plus subprocess limits are defense-in-depth, not a proof of
Python sandbox security. A production notebook should prefer a typed DSL or a
separate disposable process with no credentials and no network access.

## Iteration 70 — Separate structural validity from proof status

### Observation

`hard_valid` is a structural gate for admissible ARC grids, not evidence that
the generating rule reproduces the demonstrations. The previous schema could
therefore make a valid neural output look proof-backed in logs even though it
had never passed demo replay.

### Construction

`CandidateRecord` now carries `proof_status` with the explicit values
`unverified` and `demo_verified`. NVARC samples remain eligible candidates but
are always tagged unverified; claimed proof metadata from a neural sample is
not trusted. Leg-C verified programs/alternates and deterministic verified
primitives are tagged demo-verified. Selector eligibility remains controlled by
`hard_valid`, so this correction does not throw away useful neural coverage.
Four new tests cover validation, adapter propagation, and the distinction; the
full suite is now 251 tests.

### ARC consequence

Candidate cards and run manifests should display both fields. Only
`demo_verified` outputs may be promoted directly to attempt 1 under the hard
proof rule; unverified outputs may enter the posterior, judge, or attempt 2
only after the configured selection policy. This prevents “valid grid” from
being confused with “verified transformation” during a 12-hour run.

## Iteration 71 — Provenance-preserving holistic-judge cards

### Observation

The candidate-card compressor deduplicated exact output classes using MDL
alone. Because cards are the local judge's view of the candidate pool, this
could replace a demo-verified program with a shorter unverified neural output
and erase the distinction introduced in Iteration 70.

### Construction

`experiments/candidate_cards.py` now carries `proof_status` and
`correlation_group` into each card and emits both in the bounded prompt line.
Exact-output representatives are selected by provenance first
(`demo_verified` before `unverified`), then MDL and candidate id. Four card
tests pass and the full suite is now 253 tests.

### ARC consequence

Holistic judging now receives the same evidence semantics as replay and
selection: an exact output class cannot hide its verified representative, and
correlated lineages remain inspectable. This is a presentation/selection
correction, not a new source of hidden-label information; the raw submission
grid remains unchanged.

### Limitation

`demo_verified` still means replay-verified on the available demonstrations,
not a proof of hidden-test correctness. The final promotion policy must keep
that distinction and may need to show all representatives when provenance
classes conflict rather than compressing to one card.

## Iteration 72 — Finite version-space certificates

### Observation

Demo replay only proves membership in the surviving program version space. If
multiple rules fit every demonstration but diverge on a test input, a binary
`demo_verified` label overstates what has been identified. The official metric
allows two outputs per position, so ambiguity width itself is a measurable
constraint on recoverable score.

### Construction

`experiments/version_space_certificate.py` filters a finite program library
exactly, quotients predictions into semantic output classes, and reports per
test-input ambiguity count, forced output, semantic posterior mass, and exact
pass@2 coverability. A task is `task_forced` only when every test position has
one surviving output; a two-class version space is fully coverable by the
official two attempts even when it is not forced. Three synthetic tests pass;
the full suite is now 256 tests.

### ARC consequence

Promotion should track three separate states: no demo-consistent rule,
demo-consistent but test-ambiguous rule, and test-forced rule. A candidate with
three or more surviving semantic outputs at a position cannot be made
universally correct by pass@2 alone; compute must instead target a
discriminating invariant, an additional independent family, or a selective
abstention policy. This gives the 12-hour allocator a proof-oriented reason to
spend budget on ambiguity reduction rather than merely generating more copies.

### Limitation

The certificate is exact only relative to the enumerated library. An omitted
program can add a hidden output class, while correlated or neural samples do
not become independent hypotheses merely because they are numerous. The
posterior mass is therefore a planning signal, not a hidden-test guarantee.

## Iteration 73 — Ambiguity-aware promotion gate

### Observation

A candidate cache can improve labeled-fold score while widening its finite
version space. That is a dangerous promotion: the apparent gain may come from
overweighting one output class, while the new cache makes hidden behavior less
identified and less coverable by two attempts.

### Construction

`experiments/ambiguity_gate.py` aggregates certificate states into unresolved,
forced, pass@2-coverable, and mean-ambiguity metrics. `promotion_gate` accepts
these summaries optionally and rejects regressions in any of those quantities
in addition to its existing coverage/recovery/score checks. Three tests cover
the state decomposition and strict promotion behavior; the full suite is now
259 tests.

### ARC consequence

The deployment decision is now a conjunction: do not spend the 12-hour run on
a cache that buys development score by making the rule less identifiable.
Prefer a lane when it increases forced or two-class-coverable positions without
increasing unresolved mass or semantic ambiguity. For positions with more than
two surviving outputs, route compute toward discriminating invariants or a new
independent family, not more correlated samples.

### Limitation

This gate is only as reliable as the finite candidate library and its lineage
labels. It cannot detect an omitted rule family and it does not prove that a
two-class posterior contains the hidden answer; it is a conservative promotion
filter, not a score predictor.

## Iteration 74 — Verified-language closure frontier

### Observation

The current object executor is intentionally closed: it can prove identity,
move, recolor, add, and delete effects, but rejects shape/topology/resize
changes. Without measuring that boundary, a 12-hour run can waste compute
repeating an executor failure that is actually a missing language operator.

### Construction

`experiments/closure_frontier.py` compares each training delta profile with the
closed language, then separates `closed_verified`, `language_gap`,
`search_gap`, and `empty` tasks. The diagnostic reports unsupported delta
labels and the number of exact closed programs; three tests cover supported
recolor, resize language gaps, and deterministic dataset ordering. The full
suite is now 262 tests.

### ARC consequence

Use the frontier before allocating GPU search: `language_gap` tasks need a new
typed operator or a compositional representation, while `search_gap` tasks
need better correspondence/role search under the existing language. This
creates a falsifiable route to the low-70s: measure the marginal hidden-fold
candidate recall of each new operator family, and promote only if it closes a
frontier without reducing verified coverage elsewhere.

### Limitation

Delta labels are conservative diagnostics, not a complete semantic ontology;
shape changes may be represented as delete+add and therefore require a richer
correspondence analysis. The closure result is relative to the current
top-k/max-object search caps and must not be read as a dataset-wide theorem.

## Iteration 75 — Counterfactual metamorphic probes

### Observation

Exact demo replay cannot distinguish a relational program from a memorizer when
the demos never vary the nuisance factor. Absolute coordinates, literal color
IDs, and orientation-specific rules can therefore survive the proof layer and
still fail on a hidden test.

### Construction

`experiments/metamorphic_probes.py` adds label-free equivariance evidence for
non-identity D8 transforms, on-canvas translations, and explicit color swaps.
Each probe records a justification and compares
`h(g(x))` with `g(h(x))`; failures are soft evidence unless the task has
explicitly established the corresponding nuisance symmetry. Three tests cover
identity/D8 agreement, coordinate memorization failure, and soft color-probe
semantics. The full suite is now 265 tests.

### ARC consequence

After demo verification and before final ranking, spend CPU on a bounded probe
battery. Downweight a candidate that violates a symmetry supported by the
task's observed invariants, while retaining a second candidate when the
symmetry is merely hypothesized. This is a cheap way to turn invariance into
falsifiable evidence and should improve minority-class selection without
using hidden labels or extra GPU decoding.

### Limitation

ARC semantics are contextual: color swaps and geometric transforms are not
universally legal. The implementation therefore reports evidence rather than
hard-rejecting candidates; the caller must derive the justified probe policy
from the demonstrations and log any soft-only use.

## Iteration 76 — Transductive program-plus-output MDL

### Observation

Among demo-consistent hypotheses, program MDL alone ignores the cost of the
test-time consequence. A memorizing program can be short while emitting a
dense, structurally surprising grid; a relational rule often yields a sparse
delta from the input. This is useful as a tie-break, but unsafe as a proof
replacement.

### Construction

`experiments/transductive_mdl.py` adds a conditional output code: equal-shape
outputs pay a sparse delta cost for changed positions/colors, while shape
changes pay a full-grid code. `rank_transductive` combines this with program
MDL, preserves proof provenance, and can restrict to demo-verified candidates
or one representative per output class. Three tests cover sparse-vs-dense,
shape-change cost, and proof/output-quotient behavior. The full suite is now
268 tests.

### ARC consequence

Use joint MDL only after exact demo filtering and metamorphic/effect checks,
and only to order competing output classes or decide which candidate becomes
attempt 1. It formalizes the human preference for a compact consequence while
leaving pass@2 diversity intact. A 12-hour run should log whether this changes
selector recovery on held-out folds; no gain means remove it from the path.

### Limitation

The code is an explicit approximate code, not a universal Kolmogorov measure;
simple-looking wrong outputs can still win. The color alphabet and shape code
are fixed priors, so they must remain frozen across folds and never be tuned on
private labels.

## Iteration 77 — Guarded contextual composition

### Observation

Many ARC rules are piecewise: an action applies only to scenes satisfying a
predicate, with a different default otherwise. A flat transform library must
either duplicate templates for every context or accept an unverified neural
branch explanation.

### Construction

`experiments/guarded_programs.py` adds executable guarded branches with a
fallback action. The verifier emits a branch truth table and accepts a program
only when every demo selects at most one branch and exactly reproduces its
target; overlapping guards and wrong fallbacks fail closed. Three tests cover
piecewise proof, overlap rejection, and fallback replay. The full suite is now
271 tests.

### ARC consequence

Use guarded composition as a typed coverage expansion after object/role
extraction: infer small predicates from visible scene features, pair them with
verified actions, and retain only exact, exclusive partitions. This is a
principled way to cover contextual tasks within a 12-hour schedule without
free-form program explosion. The truth table also becomes judge-visible proof
metadata and a direct source of version-space ambiguity when branches overlap.

### Limitation

This iteration proves and executes supplied guards but does not yet synthesize
the guard vocabulary or calibrate its complexity on folds. A guard that is
exclusive on demos can still partition hidden scenes incorrectly, so
metamorphic probes and finite version-space checks remain necessary.

## Iteration 78 — Bounded guard/action product search

### Observation

Guarded composition is only useful if guard discovery is finite and auditable.
An unconstrained branch synthesizer can fit demonstrations by memorizing
scene-specific predicates, recreating the same overfitting problem as a large
language-model beam.

### Construction

`experiments/guard_search.py` generates a deterministic vocabulary from visible
height, width, object-count, and color-presence features, then enumerates a
bounded one-guard/action/fallback product. Every result passes the guarded
truth-table and exact replay proof; the candidate cap is enforced explicitly.
Three tests cover finite feature generation, contextual-rule recovery, and
bound enforcement. The full suite is now 274 tests.

### ARC consequence

Allocate a small CPU search tranche to `guard_search` after flat primitives
miss a task. It can add contextual coverage at predictable cost and produces
an interpretable branch certificate for the holistic judge. The falsification
criterion is strict: if held-out folds do not improve unique verified output
coverage per second, the guard vocabulary should not enter the Kaggle path.

### Limitation

The current vocabulary is intentionally small and one-branch only; it misses
relational and multi-branch guards. Demo exclusivity does not guarantee hidden
exclusivity, so this is a coverage proposal lane, not an automatic promotion
to attempt 1.

## Iteration 79 — Opt-in grounded shape-transform extension (negative)

### Observation

The closure frontier marked shape/topology changes as a missing operator, so a
minimal grounded `transform` operation was added behind an opt-in flag. The
hypothesis was that pairing a uniquely fingerprinted source object with a
demo-grounded target shape would immediately recover a meaningful slice of
the language-gap tasks.

### Construction

`closed_object_executor` now supports shape-changing and move-plus-recolor
operations only when `allow_shape_transform=True`; the original closed
language remains the default and its rejection test is preserved. The
expanded executor passed its focused tests and the full suite is now 275
tests. On the visible challenges, the top-k correspondence pipeline still
verified 0/1,000 training and 0/120 evaluation tasks with the extension.

### ARC consequence

This is a hard negative: adding one more grounded effect is not enough. The
dominant problem is upstream scene correspondence, multi-object decomposition,
and likely grid-level/contextual composition. Do not spend a 12-hour GPU run
simply sampling more shape-transform explanations; prioritize better
correspondence hypotheses and compositional traces, then re-measure exact
verified coverage.

### Limitation

The result is specific to the current top-k assignment, object extractor, and
single-program replay path. It does not prove shape transforms are rare or
useless; it proves only that this minimal implementation did not close the
real-data gap. The opt-in flag prevents the negative experiment from changing
the established baseline.

## Iteration 80 — Correspondence ambiguity and global-lookahead audit

### Observation

The real-data closure failure could have been caused by choosing only the
locally cheapest correspondence. On the visible first training pair,
198/1,000 tasks exceed the 10-object exact-assignment cap; among the bounded
tasks, 180/1,000 have tied best correspondences and only 396/1,000 have a
unique best among the four retained alternatives. Evaluation has 39/120 cap
cases and 31/120 tied-best cases.

### Construction

`verified_closed_programs` now has an opt-in `minimum_cost_only=False` mode
that replays every retained top-k correspondence across all demonstrations.
The extension was measured with and without grounded shape transforms: it
still verified 0/1,000 training and 0/120 evaluation tasks. The focused
executor test and full suite pass; the suite is now 276 tests.

### ARC consequence

Local assignment cost is not the dominant blocker. The next useful hypothesis
must represent role-conditioned and multi-object behavior across demonstrations
directly—likely a trace/graph anti-unification or neural proposal lane—rather
than just widening top-k or adding isolated effects. Keep the all-top-k mode
for diagnostic ablations only; it costs more CPU without observed coverage.

### Limitation

The ambiguity statistics use only the first training pair and four alternatives
and therefore are not a complete task-level assignment entropy. The negative
result does not rule out a better global assignment objective; it rules out this
specific local-cost relaxation under the current fingerprint executor.

## Iteration 81 — Trace alignment as the viable symbolic seam

### Observation

The closed object executor has no real-data verified coverage, but the existing
trace anti-unification layer exposes a non-empty seam: on visible
training/evaluation tasks it finds 169/9 equal-length traces, 135/7 fully
typed LGGs, 63/3 unique-role profiles, and 42/3 composable proof profiles.
These are diagnostic profiles, not solved outputs.

### Construction

With a smaller bounded beam (`k=2`, four aligned hypotheses) on the first 100
tasks of each split, top-1 sorted LGG succeeded on 14 training and 8
evaluation tasks; pairwise alignment succeeded on 11 and 7; top-k aligned
trace search reached 17 and 10. The measurement is label-free and uses
challenges only. The repository remains green at 276 tests.

### ARC consequence

The next symbolic investment should be trace-level candidate generation:
retain multiple aligned LGGs, ground their guards/effects, and replay complete
programs through the version-space/proof gates. Alignment is one of the few
current components showing a real non-zero path on competition-shaped data,
while widening isolated object effects has shown zero coverage.

### Limitation

The alignment audit is capped and only measures schema existence, not exact
test-output recall or runtime-adjusted gain. The top-k improvement on a 100-task
sample must be replayed on fixed folds before it can justify any GPU budget.

## Iteration 82 — Executability audit of trace schemas (negative)

### Observation

Trace/LGG schema counts are not equivalent to candidate coverage. A schema can
retain invariant action labels while still lacking grounded effect parameters,
unique roles, or a safe multi-action renderer.

### Construction

On the full visible training/evaluation challenges, `frame_role_executor`
compiled 1/1,000 and 0/120 tasks, respectively, and emitted 0 test candidates
on both splits. The single-action `role_effect_executor` had the same
1/1,000 and 0/120 compile counts and zero test candidates. The compile counts
use challenges only; available solutions were consulted only for the optional
correctness counter. No GPU or hidden-label experiment was used.

### ARC consequence

Do not promote trace schemas as if they were proof-carrying programs. The
next implementation target is a grounded trace-to-candidate compiler that
retains multiple correspondence/alignment hypotheses, solves effect
parameters across demos, and fails closed on unresolved roles. Until it emits
test grids, trace alignment belongs in proposal routing and compute allocation,
not final selection.

### Limitation

These compilers intentionally enforce strict unique-role and frame conditions,
so the negative result bounds the current proof language rather than all trace
reasoning. Relaxing gates without a new certificate would trade zero coverage
for unmeasured false positives.

## Iteration 83 — Competition constraint refresh

### Observation

The official ARC-AGI-2 competition materials were rechecked before freezing
theoretical recommendations. The current page states one submission per day,
up to two final submissions, CPU/GPU notebook runtime of at most 12 hours, no
internet access in the committed notebook, `submission.json` as the required
file name, and permission to use freely/publicly available external data and
pretrained models. The competition overview also advertises the L4x4
accelerator pool and the two-attempt evaluation framing.

### Construction

The run manifest already enforces the 4-GPU/12-hour boundary and a safety
buffer; this checkpoint adds the submission cadence, offline, artifact-name,
and two-final-submission constraints to the research assumptions. Official
links are retained below for re-verification before any actual Kaggle commit.

### ARC consequence

The optimal schedule is a single offline artifact-producing run with all
candidate generation, verification, selection, and JSON writing self-contained
in the notebook. Use the two final submissions only for genuinely orthogonal
selector policies supported by fixed-fold evidence, never as a substitute for
an unverified third guess or an internet call.

### Limitation

Competition timelines and platform hardware availability can change. The
notebook should re-check the live Overview/Rules pages immediately before
submission, while preserving the no-internet execution assumption inside the
committed code.

## Iteration 84 — Grounded aligned-trace compiler (negative smoke)

### Observation

Trace alignment preserved multiple plausible schema hypotheses, but no module
converted those hypotheses into concrete test-time grids. The missing bridge
was a compiler that retained local object indices through alignment, grounded
move/recolor/delete parameters on a reference demo, and replayed the result on
all demonstrations.

### Construction

`experiments/aligned_trace_compiler.py` enumerates bounded top-k local traces,
aligns them while retaining source/target indices, anti-unifies action schemas,
grounds only safe move/recolor/delete clauses, and exact-replay gates every
program. A multi-action synthetic proof and two failure/budget tests pass. On
the first 100 training and evaluation tasks with `k=2`, it emitted 0 verified
programs on each split. The full suite is now 279 tests.

### ARC consequence

The negative smoke result says alignment alone is not the missing ingredient;
the concrete effect parameters and role guards need a proposal source that can
describe relations beyond the current bounded predicates. Keep aligned LGGs as
structured evidence for the neural judge/repair lane, but do not allocate GPU
time to this compiler as a standalone solver.

### Limitation

The compiler intentionally supports only constant move, same-anchor recolor,
and delete effects, and fixes one alignment per retained trace combination.
It is a strict lower bound on grounded trace coverage, not a claim that
trace-level synthesis cannot work with richer relational parameters.

## Iteration 85 — Lineage-aware verified-program aggregation

### Observation

Leg C currently majority-votes verified program predictions by raw output
count. Because all samples in one generation batch share a model, prompt, and
task context, that count can exaggerate confidence and suppress an independent
verified minority class—the exact failure mode the two-attempt objective makes
expensive.

### Construction

`experiments/verified_ensemble.py` groups verified program predictions by an
explicit correlation group, gives each group one normalized semantic output
distribution, chooses the shortest MDL witness within a class, and returns at
most two classes. Collapse can be disabled to recover program-level evidence.
Three tests cover correlated-vote resistance, MDL representative choice, and
the pass@2/validation boundary. The full suite is now 282 tests.

### ARC consequence

For Leg C, tag all programs produced by one sampling batch with a shared
lineage id, preserve independently generated batches as separate groups, and
run this aggregator before merging into `submission.json`. Keep raw-majority
as a fold-tested control; promote the lineage-aware variant only if it
improves conditional recovery at fixed verified coverage. This is CPU-only and
fits the final merge buffer.

### Limitation

Correlation groups are metadata assumptions, not observed statistical
independence. Grouping an actually independent sample too aggressively can
hurt, while splitting a correlated lineage recreates vote flooding. The group
definition and temperature must be frozen before shadow-fold evaluation.

## Iteration 86 — Opt-in lineage correction at the probe boundary

### Observation

The standalone verified-ensemble design was not yet usable by the actual
Nemotron probe/cache path. If lineage correction exists only in the research
harness, the notebook's verifier will continue to raw-majority vote and the
deployment artifact will silently diverge from the theory.

### Construction

`kaggle_nemotron_probe/probe_core.py` now exposes an opt-in
`rank_verified_outputs` API and optional `correlation_groups`, `mdl_lengths`,
and `collapse_correlated` arguments on `evaluate_responses`. Default behavior
is unchanged; collapse mode reports normalized class masses and witness groups
for offline comparison. A CPU test proves that a correlated batch cannot hide
an independent verified class. The full suite is now 283 tests.

### ARC consequence

A future notebook build can compare raw-majority and lineage-corrected Leg C
selection from identical verified programs before any submission. The safe
deployment protocol is: preserve raw results, run both selectors on fixed
folds, promote only conditional-recovery gains with no coverage loss, and keep
the lineage mode behind an explicit flag until that evidence exists.

### Limitation

The probe API does not infer correlation groups automatically; the caller must
provide them. The current Nemotron sampling loop still defaults to raw
majority, so no production behavior has changed and no leaderboard gain is
claimed.

## Iteration 87 — Selective robustness for the two-attempt action

### Observation

The official score is additive over test outputs, but uncertainty about which
solver family is reliable is shared across all outputs of a task. A single
globally robust pair can therefore sacrifice a high-confidence easy output to
protect a different ambiguous output, while independently worst-casing every
position is too pessimistic because it lets the adversary change family priors
per position.

### Construction

Let (q_f(z \mid j)) be the output-class distribution for family (f) at
test position (j), and let the unknown family mixture (\pi) lie in a
calibrated interval-constrained simplex. For an allowed pair (A_j), the
conditional expected score is

\[
  U_j(A_j,\pi)=\sum_{z\in A_j}\sum_f \pi_f q_f(z\mid j).
\]

The point-posterior policy chooses the two largest marginal masses. The
shared-prior robust policy chooses a complete set of per-position pairs that
maximizes \(\min_{\pi} \sum_j U_j(A_j,\pi)\). The selective rule keeps the
point-posterior pair at position (j) unless its certified lower mass falls
below a fixed safety margin or the robust policy improves the shared-prior
lower bound by a frozen threshold; only then does it replace that position's
pair with the robust choice. The adversary remains shared across positions,
so the rule does not manufacture independent uncertainty.

### ARC consequence

This gives a principled merge policy for the two attempts: verified programs,
lineage-corrected neural outputs, and structural priors first form family
conditionals; the selector then spends robustness only where the posterior is
fragile. Easy verified positions retain their high expected value, while
ambiguous positions get protection against a wrong family-prior assumption.
The thresholds must be selected on shadow folds and frozen before the Kaggle
run; this is CPU-only and fits the final selector buffer.

### Limitation

The theorem assumes the family-conditional distributions and interval
calibration are meaningful. A bad family partition or an over-wide interval
can make the robust branch too conservative, and a threshold tuned on the
public distribution can overfit. No leaderboard gain is claimed until this is
replayed on held-out task folds.

## Iteration 88 — Balanced lineage sampling beats correlated batch voting

### Observation

The Nemotron path currently requests (k) completions from one shared
checkpoint/prompt/decoding context and treats verified outputs as raw votes.
This is the wrong evidence model when a batch repeatedly instantiates the same
mistake. More completions can increase token cost without increasing the
number of independent hypotheses.

### Construction

For lineage sample counts (n_1,\ldots,n_L), use the exchangeable
within-lineage correlation proxy

\[
  \operatorname{ESS}(n,\rho)=
  \frac{(\sum_i n_i)^2}
       {\sum_i[n_i+\rho n_i(n_i-1)]}.
\]

At fixed total sample count and fixed (L), the denominator is minimized by
balanced counts, since it is an affine function of (sum_i n_i^2). Thus
(8) samples should be allocated as (2+2+2+2) across four genuinely
different lineages before considering (8) samples from one lineage. At
(ho=1), equal two-sample lineages have ESS (4), whereas one eight-sample
lineage has ESS (1). The planning module
`experiments/lineage_sampling.py` makes this bound executable and rejects
invalid allocations.

### ARC consequence

For the 4xL4 schedule, the first control should stratify Leg C by independent
prompt/seed/temperature lineages, collapse candidates within each lineage,
then aggregate across lineages with the existing proof and semantic quotient
rules. A practical starting ablation is four lineages with two samples each;
if model-call overhead dominates, compare two lineages with four samples each
using the same total token budget. Preserve lineage IDs in the cache so raw
majority, group-normalized, and ESS-weighted selectors can be replayed from
one run.

### Limitation

The correlation (ho) is a planning parameter, not an observed ARC truth,
and prompt changes can alter per-lineage correctness rather than merely
decorrelate errors. Balanced allocation can lose if one carefully chosen
lineage is much stronger. The promotion gate therefore requires held-out
conditional recovery and verified coverage, not ESS alone. The notebook is
unchanged pending that comparison.

## Iteration 89 — Generated-notebook provenance synchronization

### Observation

The source-level probe had gained opt-in lineage-corrected ranking, but the
generated `notebook.ipynb` still embedded an older copy. A Kaggle execution
would therefore silently omit the research seam even though local unit tests
imported the newer source file.

### Construction

Rebuilt `kaggle_nemotron_probe/notebook.ipynb` through its canonical builder,
which embeds the current `probe_core.py` and related cells. A direct artifact
check now finds `rank_verified_outputs` and `collapse_correlated` in the
notebook, and the complete CPU suite remains green at 288 tests.

### ARC consequence

The research artifact and runnable notebook are again synchronized. Any future
Leg-C ablation must modify the source builder inputs first, regenerate the
notebook, and verify the embedded function names before treating a result as a
competition-relevant experiment. This prevents false confidence from testing
code that the submitted artifact does not contain.

### Limitation

Synchronization proves packaging consistency, not GPU dependency correctness,
model availability, or hidden-label performance. The notebook still keeps
lineage correction opt-in and makes no leaderboard claim.

## Iteration 90 — Eight-sample lineage schedule sensitivity

### Observation

The ESS proxy is highly sensitive to within-lineage correlation at the exact
sample count used by the current Leg-C default. For eight samples, the
calculated ESS for (1,2,4,8) balanced lineages is respectively:

| assumed \(\rho\) | 1 lineage | 2 lineages | 4 lineages | 8 lineages |
| --- | ---: | ---: | ---: | ---: |
| 0.25 | 2.909 | 4.571 | 6.400 | 8.000 |
| 0.50 | 1.778 | 3.200 | 5.333 | 8.000 |
| 0.75 | 1.280 | 2.462 | 4.571 | 8.000 |
| 0.90 | 1.096 | 2.162 | 4.211 | 8.000 |

### Construction

These values come directly from the tested
`lineage_effective_sample_size` implementation with equal allocation within
each candidate lineage count. The table is not a correctness estimate; it is
an evidence-efficiency stress test for choosing the number of independent
prompt/seed/temperature strata.

### ARC consequence

If even moderate correlation exists, four lineages with two samples each are a
stronger first ablation than one eight-sample batch while preserving enough
within-lineage sampling to discover a second output class. Eight one-sample
lineages maximize nominal ESS but may lose model quality and incur extra
prompt/launch overhead. The first GPU comparison should therefore hold total
tokens fixed and compare (1\times8), (2\times4), and (4\times2), with
lineage-normalized selection and verified-output recovery as the endpoints.

### Limitation

The table assumes exchangeable correlation and equal lineage quality. It does
not justify spending eight separate model calls if vLLM batching or prompt
diversity changes the quality distribution. The result remains a shadow-fold
ablation, not a submission decision.

## Iteration 91 — Stratified lineage controls in the runnable Leg-C runner

### Observation

The ESS recommendation was only a planning artifact until the actual
Nemotron runner could execute multiple decoding groups and carry their
boundaries into verification. Without that seam, a future experiment would
still be forced to compare raw votes from one batch against a different code
path.

### Construction

`kaggle_nemotron_probe/nemotron_induction.py` now exposes
`make_sampling_lineages`, `--lineages`, `--lineage-temperatures`, and the
opt-in `--lineage-aware` switch. The default is exactly one lineage, one
temperature, and legacy raw-majority behavior. When enabled, the fixed `k`
budget is split as evenly as possible, each group gets a deterministic offset
seed and frozen temperature, vLLM is called once per group, and verified
programs are ranked by the existing group-normalized output quotient. A
mocked flood test shows ten verified copies in one lineage cannot suppress two
independent verified lineages. The generated notebook was rebuilt and the
full suite passes 291 tests.

### ARC consequence

The proposed GPU ablation is now executable from the competition artifact:
hold `k` and total token budget constant, compare `LEGC_LINEAGES = 1, 2, 4`
with corresponding balanced batches, and enable `LEGC_LINEAGE_AWARE` only for
the group-normalized arm. Keep the default flags unchanged until shadow-fold
recovery, verified coverage, and wall-clock cost are measured. This preserves a
clean rollback to the known baseline while making the moonshot lineage
hypothesis falsifiable.

### Limitation

Distinct seeds and temperatures are operational strata, not proof of
independence. Multiple vLLM calls can increase overhead and reduce throughput,
and a temperature schedule can change correctness as well as diversity. No
GPU or Kaggle run was performed here.

## Iteration 92 — Four-replica Nemotron deployment is memory-fragile on L4

### Observation

Lineage independence would be strongest if each L4 hosted its own Nemotron
replica, but the attached checkpoint is not a comfortable one-L4 model. The
52 safetensors files occupy 21,561,882,284 bytes (about 20.1 GiB) before
vLLM/runtime allocations, KV cache, CUDA workspaces, and fragmentation.

### Construction

The current runner reserves `gpu_memory_utilization=0.88`, which is roughly
21.1 GiB on a 24-GiB L4. A one-GPU replica would therefore consume nearly the
entire reservation before serving a 16k-context request; four replicas would
also duplicate all weights and leave no robust activation/KV margin. The
existing tensor-parallel configuration instead divides the weights across all
four L4s and has already passed the local packaging/sandbox checks.

### ARC consequence

Do not replace tensor parallelism with four one-GPU Nemotron processes without
an actual Kaggle memory trace. The executable lineage ablation should keep one
TP=4 engine and issue balanced sequential decoding groups, holding total
tokens fixed. Physical replica diversity is a future option only with a
smaller checkpoint or an independently verified low-memory runtime. The four
L4 schedule should spend its parallelism on model capacity/context and its
lineage diversity on seeds, temperatures, prompt variants, and verified
post-processing.

### Limitation

File size is a lower bound, not a complete vLLM memory measurement. Weight
compression, paged KV, and different context limits could change the result;
the conclusion is a safety constraint, not a proof of impossibility. No GPU
was available locally to measure the actual runner.

## Iteration 93 — Missing-candidate denominator correction

### Observation

`fit_family_calibration` previously accumulated a family success/failure only
for task-positions where that family had emitted at least one candidate. A
sparse family could therefore look perfect by never attempting difficult or
expensive positions. That biased the posterior family rates used by compute
allocation and robust pass@2 selection.

### Construction

The calibration loop now enumerates every labeled task-position for every
active family. It marks the position successful iff the family’s exact output
class set contains the labeled output; an absent record is an explicit
failure. Feature-conditioned counts use the same denominator. Regression
coverage now includes a success plus a missing-position failure, and the full
suite passes 293 tests.

### ARC consequence

Family priors and hit rates are now conservative coverage estimates rather
than “success given that this family produced something.” This changes the
recommended 4xL4 allocator in an important way: a lane that produces few
verified candidates cannot earn a large budget merely from its occasional
successes. It must demonstrate held-out output-position coverage, after which
lineage diversity and selector recovery determine its marginal value.

### Limitation

The correction treats missing output as a family failure, which is correct for
coverage but does not distinguish “not scheduled yet” from “attempted and
failed.” Calibration folds must therefore use a fixed, documented budget and
record the eligible queue; otherwise the denominator can fairly penalize a
lane that simply received less compute.

## Iteration 94 — Eligibility-aware family calibration

### Observation

Counting missing candidates as failures fixes sparse-family optimism for a
fully scheduled fold, but adaptive routing creates a different ambiguity:
absence may mean the lane was never eligible under the budget policy. Treating
those unscheduled positions as failures would understate the lane's intrinsic
success rate and confound model quality with scheduler exposure.

### Construction

`fit_family_calibration` now accepts an optional per-family
`eligible_positions` mask. With no mask, every labeled position remains an
observation and missing candidates are failures. With a mask, only positions
where that family was actually eligible enter the intrinsic success/failure
posterior; missing candidates within the mask still count as failures. A new
regression test proves that an unscheduled position is excluded while a
scheduled missing position is penalized. The full suite passes 294 tests.

### ARC consequence

The 12-hour experiment ledger must persist two separate quantities for every
lane: eligible positions and recovered positions. Use the eligibility-aware
rate to choose the next lane, but use the unmasked coverage evaluator to
measure the final submission's actual score opportunity. This separation is
necessary for a valid comparison of one-batch versus stratified Leg-C, and for
any compute allocator that stops early on easy tasks.

### Limitation

An eligibility mask is only as trustworthy as the runner's scheduling log. If
the mask is reconstructed after a timeout or omits queue failures, it can
reintroduce bias. The production notebook does not yet emit this calibration
metadata, so no GPU allocation has been promoted from the theory.

## Iteration 95 — Nuisance-marginal log-evidence ranking

### Observation

The baseline `kgmon` selector scores an exact-output class as raw view support
minus mean augmented NLL. Raw support is a dangerous proxy when D8/color
views and repeated generations are correlated: twenty copies of a mediocre
grid can outrank one high-likelihood grid. The cache already contains beam
scores and eight augmentation NLLs, so this ranking failure can be tested
without another generation run.

### Construction

`kaggle_nemotron_probe/arc_decoder.py` now provides opt-in
`getter_log_evidence`/`score_log_evidence`. For an exact-output class, it
computes a log-mean-exp of the negative beam scores and a log-mean-exp of each
candidate's augmentation likelihood, then combines the two terms. The
log-mean treats view/sample copies as exchangeable nuisance evidence rather
than independent votes. Four CPU tests prove probability-space marginalization,
duplicate invariance, numerical empty handling, and rejection of a twenty-copy
support flood. The benchmark list includes the scorer, but `kgmon` remains the
submission default. The generated notebook was rebuilt and the full suite is
now 298 tests.

### ARC consequence

On a saved decoder cache, compare `kgmon`, `full_probmul_3`, and
`log_evidence` by candidate recall, top-two selector recovery, and wall-clock
cost on fixed folds. If log evidence improves conditional recovery at equal
coverage, make it attempt 1 or use it as the primary class ranking while
retaining kgmon as an orthogonal second-attempt policy. The two-attempt
objective makes duplicate-invariant ranking especially valuable: it frees the
second slot for a genuinely different output class instead of a support-count
copy.

### Limitation

Beam and teacher-forced NLLs are not calibrated probabilities and may double
count correlated evidence. The correct aggregation could require lineage
metadata and temperature calibration; this module deliberately uses a
conservative exchangeable average and is not promoted until labeled fold
replay shows a gain.

## Iteration 96 — Baseline-preserving log-evidence deployment seam

### Observation

Adding a scorer only to a local benchmark is insufficient: a future ablation
could accidentally compare one implementation and submit another. The
generated notebook's diverse-attempt cell previously hardcoded
`score_full_probmul_3` as its secondary policy.

### Construction

The builder now imports `score_log_evidence` and exposes
`LOG_EVIDENCE_ATTEMPT_2`. When false, the existing `full_probmul_3` control is
selected; when true, the exact same log-evidence function used by the cache
benchmark supplies attempt 2 while `kgmon` remains attempt 1. The default is
unchanged, the notebook was regenerated, and the complete suite remains green
at 298 tests.

### ARC consequence

The next no-quota shadow-fold comparison can toggle one flag and attribute any
change to a single ranking policy: baseline `kgmon` top-2, diverse
`full_probmul_3`, or diverse `log_evidence`. Promotion should require no loss
in candidate coverage, an improvement in conditional top-two recovery, and a
stable result across task folds. This makes the scoring moonshot runnable
without perturbing the known baseline by default.

### Limitation

The current repository has no decoder cache with hidden labels available for
this scorer, so its advantage is proven only against synthetic support-flood
cases. It must not be enabled for a Kaggle submission on theory alone.

## Iteration 97 — Canonical tie-breaking for decoder classes

### Observation

`score_sum` sorted equal-scoring output classes using insertion order. Because
the decoder cache is loaded from filesystem entries and sampled views, that
order is not stable evidence and can vary between otherwise identical runs.
Ties matter directly because the second pass@2 slot is selected from the
ranked list.

### Construction

All decoder selectors now sort by descending score followed by the canonical
tuple representation of the output grid. Two regression tests feed the same
tie in opposite dictionary orders to both `kgmon` and `log_evidence` and
require identical output ordering. The scorer remains mathematically
unchanged away from ties; the full suite passes 300 tests and the generated
notebook was rebuilt.

### ARC consequence

Cache replay, fold comparisons, and Kaggle artifacts now have deterministic
tie behavior. This removes a hidden source of run-to-run score variance and
makes selector ablations attributable to their scoring rule rather than
filesystem order. The canonical tie-break is safe to apply to the baseline
selector.

### Limitation

Deterministic lexical order is only a neutral fallback, not a semantic prior.
If ties are frequent and meaningful, a future proof/MDL or lineage witness
should replace the lexical fallback—but that change requires labeled fold
evidence.

## Iteration 98 — Sentinel-safe four-worker queue protocol

### Observation

The base solver's worker loop used `while not queue.empty():` followed by
`queue.get()`. On a managed multiprocessing queue, `empty()` is only a racey
observation; a worker can terminate while tasks remain in transit or visible to
another worker. Under the 12-hour cutoff that silently converts available GPU
time into zero-score placeholders.

### Construction

The starter already enqueues exactly one `None` sentinel per worker. The solver
now blocks on `queue.get()` inside `while True` and exits only on its sentinel
or the explicit deadline. A source-level regression guard rejects a return to
the `empty()` polling pattern and verifies the sentinel branch. The generated
notebook was rebuilt and the full suite passes 301 tests.

### ARC consequence

This removes a potentially catastrophic coverage failure from the four-L4
baseline without changing task order, model weights, or selection policy. The
12-hour run can now consume every queued task until the deadline, making the
cheap-first coverage policy and future expected-value scheduler measurable
rather than confounded by queue races.

### Limitation

The local test is a protocol guard, not a four-process Kaggle stress test. A
blocking `get()` assumes the starter reliably enqueues one sentinel per worker;
that invariant remains explicit in `starter.py` and should be checked in the
commit-run log.

## Iteration 99 — Partial Leg-C task skip correction

### Observation

The Leg-C merge path treated a task-level `verified=True` as sufficient reason to remove the entire task from the base queue. For multi-test tasks, that is unsound: one test position can have a verified Leg-C attempt while another has no attempt or only an unverified attempt. Skipping the task then loses the base solver’s candidates for the unresolved positions.

### Construction

The starter now skips a task from the base queue only when every output entry has a non-null Leg-C attempt. A partially covered task remains in the base queue; merge logic still overrides positions that Leg-C actually verified. Tests cover the source contract and executable partial-versus-complete behavior.

### ARC consequence

This creates a monotone coverage invariant for the ensemble: adding a Leg-C result cannot remove base coverage for an unresolved test position. It is especially relevant to ARC-AGI-2 because task-level metadata is coarser than the scoring unit, while the competition score is effectively position-level.

### Limitation

The correction assumes the result schema continues to represent each test position as an output entry with an `attempt` field. Malformed or incomplete Leg-C records are conservatively treated as unresolved and handed back to the base path; this protects recall but can spend extra inference time.

## Iteration 100 — Reflection is valuable only when conditional repair beats fresh sampling

### Observation

The current Leg-C lane spends its sample budget on independent program proposals,
then discards every demo-inconsistent program. Recent public ARC-AGI-2 material
reinforces the importance of iterative refinement and parallel proposal diversity,
but the most dramatic public result located in this pass uses an external Gemini
API/sandbox and therefore is not a legal or reproducible Kaggle deployment under
the offline notebook constraint. The useful transferable primitive is the
verified counterexample: a failed candidate identifies a concrete first-diverging
demo and can be repaired rather than thrown away.

### Construction

Let (p) be the probability that a fresh program proposal is demo-verified, and
let (q) be the conditional probability that a repair proposal succeeds after
receiving the candidate and its exact first mismatch. For two equal-cost proposal
slots, two independent samples succeed with probability

\[
1-(1-p)^2 = 2p-p^2,
\]

whereas one proposal followed by one conditional repair succeeds with

\[
p+(1-p)q.
\]

The repair pair is strictly better exactly when (q>p). With unequal costs,
the promotion rule is the measured marginal gain per second: use repair when
\((1-p)q/c_{repair}) exceeds the best available fresh-sample gain per second,
and reserve a repair only for candidates with an actionable mismatch. This is a
decision theorem, not an assumption that reflection is automatically useful.

### ARC consequence

The next high-value shadow-fold experiment should split a fixed Leg-C budget into
an initial proposal tranche and a repair tranche. The repair prompt should contain
only a bounded candidate source, the first failing demonstration index, the
candidate output, and the required output; the verifier remains the sole authority
for promotion. This couples naturally to the existing first-divergence/CEGIS
artifacts and preserves the two-attempt rule: a repaired verified output replaces
the failed candidate class, while an unresolved position falls back to the base
lane.

### Limitation

No repair loop is enabled yet. The repository has no saved Nemotron candidate
cache from which to estimate (p), (q), or repair cost, and a second generation
round could reduce total task coverage. The public 97.92% result is public-eval
and API-assisted evidence, not a claim about the offline competition score; it is
used only to motivate the refinement hypothesis.

## Iteration 101 — Make the Leg-C/base time contract executable

### Observation

The deployment cell exposed two independent time controls, `LEGC_BUDGET_H = 1.0`
and `BASE_RESERVE_H = 9.8`, but passed an explicit `--end-ts` computed only from
the base reserve. Because the induction runner gives `--end-ts` precedence over
`--budget-h`, the effective Leg-C ceiling was approximately 2.0 hours after
startup rather than the documented 1.0 hour. This was a configuration ambiguity,
not a model-quality result, but it could consume the budget intended for the
base solver or make ablations incomparable.

### Construction

The launch cell now sets the induction deadline to

\[
\min(t_0 + B_{LegC},\; t_{global\_end} - B_{base}),
\]

so both the declared Leg-C cap and the guaranteed base reserve are enforced.
A source contract test protects the two-sided bound, and the generated notebook
was rebuilt. The change does not alter the default model, lineage policy, or
selection flags; it only makes the documented schedule real.

### ARC consequence

The one-hour Leg-C hypothesis now leaves the base pipeline up to 10.8 hours
before the fixed ten-minute write buffer, rather than silently permitting a
second hour of induction. This is the conservative choice until a shadow fold
shows that extra verified-program yield per hour exceeds the base solver’s
coverage/recovery yield. It also makes future budget experiments interpretable.

### Limitation

The best allocation remains unknown because no four-L4 run or decoder cache is
available locally. If a later labeled fold proves a two-hour Leg-C phase is
superior, the budget can be changed explicitly by raising `LEGC_BUDGET_H` or
lowering `BASE_RESERVE_H`; it will no longer happen accidentally.

## Iteration 102 — Executable reflection economics for the second attempt budget

### Observation

The previous iteration gave a success-probability theorem, but left the cost
tradeoff implicit. A repair prompt may be longer or slower than a fresh batch,
and using it indiscriminately could reduce the number of task positions reached
before the Kaggle deadline.

### Construction

`experiments/reflective_budget.py` now exposes a model-independent decision
function. If (p) is the fresh proposal success rate, (q) is the conditional
repair success rate after an actionable first divergence, (c_f) is fresh cost,
and (c_r) is repair cost, the second-stage gains after failure are (p) and
(q). Therefore the cost-aware rule is

\[
\frac{q}{c_r} > \frac{p}{c_f}.
\]

The equal-cost special case reduces to (q>p). Four CPU tests cover the strict
tie rule, unequal costs, both pair-probability formulas, and invalid inputs.

### ARC consequence

Once a shadow run records candidate status, first mismatch, repair latency, and
whether the repaired program verifies, the policy can allocate the second-stage
Leg-C budget per output position. Candidates without an actionable mismatch do
not qualify for repair; they remain eligible for a fresh lineage or the base
solver. This turns “reflection” into an auditable value-of-information gate
rather than a vague extra generation round.

### Limitation

The module contains no model calls and is not wired into production. Estimating
(q) from the same tasks used to tune prompts would overfit; rates must come
from held-out folds or a pre-registered shadow run, with coverage and wall-clock
cost measured jointly.

## Iteration 103 — Preserve bounded counterexamples for future repair

### Observation

The reflection policy requires a concrete counterexample, but the verifier only
returned a string such as `demo 0 mismatch`. That identifies where a candidate
failed but not what the model must repair. Discarding the observed and expected
grids also makes a later repair experiment impossible to audit from the saved
candidate record.

### Construction

`CandidateResult` and the isolated verifier now retain the first failed demo
index, the candidate's normalized observed grid (when execution returned one),
and the expected demonstration grid. Exception failures retain the index and
expected grid but no fabricated observation. This telemetry is diagnostic only:
the exact all-demonstrations verifier gate is unchanged, and unverified programs
still cannot enter `induction_results.json` as promoted outputs. A regression test
checks the mismatch payload; the generated notebook is rebuilt from the probe
source.

### ARC consequence

The future repair prompt can be a bounded function of

\[
(\text{candidate source},\; d,\; \hat y_d,\; y_d),
\]

where (d) is the first failed demonstration. This is materially stronger than
free-form “try again” reflection: it gives the model a falsifiable local target,
while the verifier still checks every demo and the hidden test remains unseen.

### Limitation

One first mismatch is not a proof that the causal defect is local; the candidate
may contain multiple coupled errors. The repair lane must therefore retain the
original candidate as provenance, cap prompt size, and accept a repaired program
only after complete replay. No repair generation is enabled by this change.

## Iteration 104 — Persist bounded Leg-C repair telemetry

### Observation

The verifier now knows the first counterexample, but the production-shaped
induction runner previously discarded every failed candidate after ranking. A
future shadow fold could not estimate repair probability (q), repair latency,
or failure modes from `induction_results.json`.

### Construction

`nemotron_induction.py` now has an opt-in `--diagnostics` path. Each output may
persist at most a configured number of failed candidates, each carrying lineage,
source, status, first failed demo, and bounded observed/expected grids. Verified
candidates are omitted because their promoted output is already represented.
`LEGC_DIAGNOSTICS=False` is embedded in the generated notebook, so the default
submission artifact and result schema remain unchanged. A CPU test exercises the
bounded payload and the notebook was rebuilt.

### ARC consequence

A legal public-evaluation shadow run can now produce the sufficient statistics
for the reflection gate: actionable-failure count, repaired-verification count,
candidate source length, and wall-clock cost. The next experiment can compare

\[
q/c_r \quad\text{against}\quad p/c_f
\]

on pre-registered folds before allocating any competition-time repair calls.

### Limitation

Diagnostics are intentionally not a repair loop and do not expose hidden test
labels. The per-output cap can omit later failure modes, while retaining full
source text can increase disk use; both limits are explicit and adjustable only
in a shadow run.

## Iteration 105 — Deterministic bounded repair prompts

### Observation

Persisting counterexamples still leaves a reproducibility gap if the future
repair prompt is handwritten inside the runner. Prompt variation can change
repair rate, token cost, and hidden-test leakage risk at the same time, making a
shadow-fold comparison uninterpretable.

### Construction

`experiments/repair_prompt.py` defines a deterministic constructor over candidate
source, first failed demonstration index, observed grid, and expected grid. It
rejects non-rectangular/oversized/non-ARC diagnostics and oversized source, and
explicitly instructs the model not to use hidden-test data. Four CPU tests cover
normal payloads, exception observations, size limits, and invalid contexts. The
constructor is not wired into the production notebook.

### ARC consequence

The future reflection ablation can pre-register the exact prompt contract and
measure (q) without conflating prompt edits with budget allocation. A repaired
candidate remains subject to complete demo replay; prompt text is never treated
as evidence of correctness.

### Limitation

The constructor does not establish that the first mismatch is causally local or
that a model can use the feedback. Its value is experimental control and safety;
the repair lane still requires held-out fold evidence before deployment.

## Iteration 106 — Diversity-adjusted reflection after holistic-judge evidence

### Observation

The newly re-read MDS study reports a crucial negative result: prescriptive
iterative refinement systematically reduced candidate diversity, while
independent text/image/code generation followed by holistic trace judging
recovered correct minority hypotheses. Therefore the earlier reflection rule
(q/c_r > p/c_f) was incomplete: a repair can be locally more likely to verify
and still be globally worse if it stays in the same correlated hypothesis basin.

### Construction

`reflective_budget.py` now includes `repair_novelty = n_r`, the estimated fraction
of repair successes that add a distinct useful output class. The correct gate is

\[
\frac{q n_r}{c_r} > \frac{p}{c_f}.
\]

The old theorem is the special case (n_r=1). A regression test demonstrates a
high-(q), low-novelty repair losing to a fresh proposal. This preserves the
repair idea only as a late, evidence-gated fallback; it no longer treats
counterexample conditioning as automatically beneficial.

### ARC consequence

The next serious architecture target is diversity-preserving selection: use
independent lineages and heterogeneous reasoning cards first, then judge all
candidate traces jointly. A repair round is allowed only for actionable failures
whose measured novelty-adjusted gain exceeds the best fresh lane. This aligns
the local Nemotron design with the transferable part of MDS while avoiding its
API-dependent models and tools.

### Limitation

The MDS headline scores and ablations are not an offline four-L4 replication,
and (n_r) is unknown until candidate/output provenance is measured. The current
repository therefore keeps holistic judging and repair unpromoted; only the
decision calculus is executable.

## Iteration 107 — Re-anchor the top-8 target to the live board

### Observation

The current official Kaggle leaderboard snapshot does not support the assumption
that every top-eight score is already in the 70s. The visible public scores are
72.08 for rank 1, 70.42 for rank 2, 40.83 for rank 3, 37.22 for rank 4, 34.86
for ranks 5 and 6, and 34.44 for ranks 7 and 8. Kaggle states that the public
leaderboard uses approximately half of the test data and that final standings
use the other half.

### Construction

No solver code is changed. The ledger now treats 70% as a top-two/public-board
moonshot and treats “top eight” as a robust private-board rank objective whose
cutoff is currently about 34.4% on the visible board. This distinction prevents
selecting methods by a moving public threshold or mistaking leaderboard shock
for a proof that a 70s private score is required.

### ARC consequence

The deployment objective remains ambitious: maximize expected private exact
output coverage and pass@2 recovery, with a strong 70s scenario reserved for a
successful heterogeneous-judge breakthrough. Evaluation decisions should use
the 120-task public set only for leakage-safe development and should report
uncertainty, because public rank and private rank can diverge.

### Limitation

The live public board is not evidence of the private cutoff, and leaderboard
entries may change before the final deadline. This is a target calibration, not
a performance claim for the current notebook.

## Iteration 108 — Strict bounded holistic-judge contract

### Observation

The MDS evidence identifies holistic judging over a jointly visible candidate
pool as the main way to recover correct minority hypotheses, but the repository
only had final pair aggregation and compact cards. Without a strict input/output
contract, a local judge could be given an uncontrolled prompt or silently invent
a third grid that had no provenance or verifier path.

### Construction

`experiments/holistic_judge_prompt.py` now builds a deterministic, size-bounded
bundle from deduplicated `CandidateCard` objects and exact output grids. Its
parser accepts exactly two distinct full hashes already present in the bundle,
including fenced JSON, and rejects unknown classes, malformed responses, and
single-class output. Four CPU tests cover deterministic ordering, context
bounds, strict parsing, and malformed input.

### ARC consequence

This creates a safe local analogue of the holistic-judge stage: heterogeneous
candidate generators can be pooled, provenance and structural evidence can be
shown jointly, and the judge can only select existing output classes. The
official pass@2 action remains exact and auditable. In a future shadow fold,
compare this judge against lineage mass, MDL, and the weighted council on the
same candidate pool; no new model-generation budget is needed for the selector
comparison itself.

### Limitation

The compact bundle contains output grids and metadata, not the 30k–80k-token
reasoning traces used by the external MDS study. It is therefore a contract and
ablation seam, not evidence that a small local judge will recover the same
minority hypotheses. No judge model is wired into the submission notebook.

## Iteration 109 — Prompt-style strata for diversity-preserving generation

### Observation

The Leg-C lineage mechanism previously varied only seed and temperature. Those
samples still shared a highly prescriptive system prompt, so the lineages were
not independent in the representation sense. The MDS study reports that
prescriptive iterative prompting reduced useful diversity, making this a
plausible low-cost ablation target.

### Construction

`probe_core.py` now exposes three deterministic styles: the unchanged `strict`
control, a shorter `minimal` instruction, and a `freeform` instruction that
leaves the reasoning route open while retaining the safe function contract.
`SamplingLineage` and the runner can cycle these styles with
`--prompt-styles`; defaults remain one strict lineage, so legacy behavior is
preserved. CPU tests cover prompt determinism, invalid styles, and lineage
stratification. The generated notebook was rebuilt.

### ARC consequence

For a fixed (k), a shadow fold can compare `strict×k` against a pre-registered
style schedule such as `strict/minimal/freeform/strict`, measuring unique
verified programs, unique output classes, oracle pass@k, top-two recovery, and
tokens per output. Promotion should require output-class complementarity, not
just more parsed code. This directly tests whether representation diversity is
worth more than temperature diversity on the same Nemotron checkpoint.

### Limitation

The styles are still code-generation prompts and do not reproduce independent
text/image modalities. The freeform wording may also increase malformed code or
unsafe constructs, though the existing AST gate contains those failures. No
style schedule is enabled for a scored run without fold evidence.

## Iteration 110 — Preserve bounded reasoning traces for holistic adjudication

### Observation

The holistic-judge hypothesis depends on comparing reasoning traces, not merely
output grids. The opt-in Leg-C diagnostics added in Iteration 104 retained
extracted source and verifier feedback but discarded the original model text,
which contains the model's object/relational interpretation and failure rationale.

### Construction

Diagnostics now retain the first raw candidate text for each recorded failed
program, bounded by `--diagnostic-trace-chars` (default 6,000), alongside the
source, lineage, and exact verifier payload. Duplicate program hashes keep the
first deterministic trace. The generated notebook exposes the bound but keeps
`LEGC_DIAGNOSTICS=False`; one regression test confirms trace persistence.

### ARC consequence

This enables a fair shadow comparison among three selectors on the same pool:
lineage-normalized mass, deterministic card/council selection, and a local
holistic judge that sees bounded traces plus exact outputs. The selector may
choose only existing output classes, so better reasoning evidence cannot bypass
the two-attempt or verifier boundaries.

### Limitation

Truncating traces at a fixed character count is not equivalent to the external
study's 30k–80k-token context, and raw model explanations can be confidently
wrong. Trace text is therefore evidence for a judge ablation only; it never
certifies a candidate and is not persisted in the default competition run.

## Iteration 111 — Correlation-normalized holistic judge councils

### Observation

The holistic-judge seam normalized generator lineages but the final judge
council still treated every ranked pair as independent evidence. Ten copies of
the same judge prompt could therefore overwhelm one genuinely independent judge
even when the latter identified the correct minority output class.

### Construction

`experiments/judge_aggregation.py` now accepts optional `judge_groups`. Each
group contributes total weight one, so repeated judges in the same correlated
group are averaged rather than counted as independent votes; unlabeled judges
retain legacy one-vote behavior. Position-debiased aggregation supports the same
group correction. Two regression tests cover minority recovery under a ten-copy
flood and group-length validation.

### ARC consequence

The complete evidence hierarchy is now consistent: deduplicate exact outputs,
normalize correlated generator lineages, normalize correlated judge groups, and
then apply the pass@2 weighted council. In a shadow fold, judge groups should be
assigned by prompt/model/configuration lineage—not by arbitrary call order—and
the corrected council should be compared with raw majority and lineage mass.

### Limitation

Group normalization is only as good as provenance labels; falsely declaring
independent judges correlated loses useful evidence, while failing to group
copies recreates vote flooding. The council remains a selector over existing
outputs and has no hidden-label or correctness guarantee.

## Iteration 112 — Hard-validity poisoning guard for joint program vectors

### Observation

The additive/marginal task selector correctly uses complete program vectors as a
joint posterior, but the adapter that constructs those vectors silently ignored
`hard_valid=False` records. A vector with one structurally invalid output could
therefore contribute mass to every other test position, allowing coherence to
launder a bad candidate.

### Construction

`to_task_candidates()` now treats any hard-invalid record for a `(family,
program_id)` pair as poisoning the whole vector and drops it. Incomplete vectors
remain dropped as before. A regression test covers a valid first position paired
with an invalid second position.

### ARC consequence

Joint evidence is now monotone with respect to structural validity: a program
vector enters the posterior only if every output it supplies is hard-valid and
the vector is complete. This preserves the official per-output marginal action
while making shared-rule coherence a safe diagnostic rather than a validity
escape hatch.

### Limitation

`hard_valid` is a structural/schema gate, not semantic correctness. A fully
valid but wrong program can still influence the posterior and must be controlled
by family/lineage provenance, trace judging, and held-out recovery evidence.

## Iteration 113 — Coverage-constrained context allocation for holistic judging

### Observation

The released holistic-judging result supports showing complete candidate traces,
but the local notebook cannot afford to pack every raw trace into a small judge
context. A length-only truncation policy has a pathological failure mode: a
large correlated lineage can consume the context while a rare output class gets
no explanation. The judge then sees frequency as evidence and cannot recover the
minority hypothesis that the external result specifically says matters.

### Construction

Added `experiments/holistic_context.py`. Its policy has two phases:

1. group evidence by exact output hash and reserve one canonical witness per
   class, with deterministic priority/family/lineage/candidate tie-breaking;
2. fill remaining characters by greedy marginal coverage of unseen families,
   lineages, and output-family pairs, then priority and canonical identifiers.

The prompt builder can optionally append these traces after the exact grids.
If the trace budget cannot fit one witness per class, it fails closed so the
caller can use a cards-only bundle instead of silently removing minority
hypotheses. The parser still accepts only two known output hashes.

### ARC consequence

For a fixed context budget, output-class coverage is now a hard constraint and
provenance diversity is the first soft objective. This is a direct local
approximation to the MDS holistic-judge principle: preserve independent
candidate evidence, then let a joint judge rank existing classes. The coverage
claim is exact by construction; the semantic claim remains unproven until a
cached candidate pool is replayed through a judge.

### Limitation

One trace per class is only a witness, not a correctness proof. A bad class can
still receive a persuasive trace, and the greedy diversity utility is a policy
choice rather than a calibrated posterior. The next empirical test is a
shadow-fold comparison of cards-only, frequency-packed traces, and this
coverage-constrained policy under equal prompt characters. No production flag
is enabled.

## Iteration 114 — Factorization proof for the official pass@2 action

### Observation

ARC tasks often have multiple test inputs generated by one latent rule, so a
coherent program vector is valuable evidence. It is easy to make the stronger
but incorrect leap that both submission attempts must themselves be two
coherent whole-task programs. That objective can discard a complementary pair
of output classes that wins on the actual metric.

### Construction

Let `Z_j` be the hidden correct output class at test position `j`, and let
`A_j` be the set of at most two submitted classes for that position. For any
posterior over the complete latent vector `Z`,

`E[score | data] = sum_j sum_z 1[z in A_j] P(Z_j=z | data)`.

The feasible action set is the Cartesian product of the per-position choices,
so the Bayes-optimal `A_j` is independently the two largest marginal masses
`P(Z_j=z | data)`. Cross-position correlations affect how the posterior is
constructed and calibrated, but they do not couple the final action under the
additive official score. `select_task_output_pairs()` already implements this
rule; `select_task_program_pair()` is explicitly retained as a coherence
diagnostic, not a submission optimum.

### ARC consequence

The high-value architecture is: use task-wide program coherence to improve
candidate generation, posterior weighting, and judge context; then perform
the final two-class choice independently for each test output. This permits
attempt pairs assembled from different programs while remaining exactly
metric-optimal under a calibrated posterior. A coherent-vector selector may be
used as a prior or a robust fallback, but not as an unqualified replacement.

### Limitation

The factorization proof assumes a calibrated posterior and no cross-position
submission constraint beyond two classes per output. A shared uncertainty about
family priors can make a minimax objective coupled; that is a deliberate robust
ablation and should be compared against the Bayes action on fixed shadow folds.
The proof does not improve candidate recall, which remains the dominant
moonshot bottleneck.

## Iteration 115 — Fail-closed trace coverage at the judge boundary

### Observation

The context allocator can guarantee one trace witness for every class present
in its evidence input, but a diagnostic pipeline may omit traces for some
candidate classes. Silently building a partial trace bundle would reintroduce
the minority-hypothesis failure that the allocator was designed to prevent.

### Construction

The prompt builder now passes the complete set of exact card hashes as required
trace classes. `select_trace_evidence()` rejects missing required classes and
also rejects a budget too small for their witnesses. The safe caller behavior
is then explicit: omit raw traces and use the exact-grid/cards-only bundle, or
collect the missing bounded diagnostics before invoking a trace judge.

### ARC consequence

The trace path now has a monotonic safety invariant: enabling it cannot make an
existing output class invisible through accidental trace omission. This keeps
the selector a ranking layer over the full candidate quotient, while allowing
trace-based minority recovery when complete evidence is available.

### Limitation

Fail-closed behavior protects evidence coverage but may spend judge context or
fall back to a weaker cards-only prompt more often than necessary. A later
shadow-fold run must measure whether the trace judge's conditional recovery
gain justifies the context and diagnostic-storage cost.

## Iteration 116 — Position-conditioned value of information for lane routing

### Observation

The existing adaptive router applies one scalar novelty rate to every test
output. That is mathematically adequate only when a proposal lane has uniform
conditional discovery probability. ARC families are structurally heterogeneous:
a lane may be highly complementary for one group and nearly redundant for
another. A scalar average can therefore route compute toward the wrong lane.

### Construction

Added `decide_route_positionwise()`. For output position `j`, let `u_j` be the
current unresolved mass, `q_j` the fold-calibrated probability that the lane
discovers the correct class conditional on unresolved status, and `r_j` the
conditional selector recovery rate. The expected rescued-output count is

`Delta = sum_j u_j q_j r_j`,

with priority `Delta / lane_seconds`. The former scalar API is exactly the
special case `q_j=q` and `r_j=r`, preserving existing behavior. Length and range
guards reject mismatched or uncalibrated vectors.

### ARC consequence

The four-L4 schedule can now route by residual structural-group opportunity,
not global average novelty. The correct deployment statistic is conditional
new-class coverage on unresolved positions, multiplied by selector recovery;
raw sample count and aggregate task hit rate are insufficient. This composes
with the submodular family allocator: use positionwise `q_j` estimates for
conditional complementarity, then reserve the final judging/serialization
buffer.

### Limitation

The rule is only as good as the fold calibration. Sparse structural groups need
hierarchical shrinkage or a conservative unknown reserve, and the independence
assumption between sequential lane discoveries is not guaranteed. No GPU run
or leaderboard claim is made; the next empirical requirement is a fixed-budget
shadow-fold comparison against the scalar router.

## Iteration 117 — Preserve positionwise calibration through deployment

### Observation

Position-conditioned routing is only actionable if the fold calibrator retains
the task/test-position identity. The existing `target_family_rates()` averaged
all target positions into one scalar, erasing the very structural complementarity
needed by the new router.

### Construction

Added `target_family_position_rates()`, which projects each calibrated family
onto `(task_id, test_index)` using only challenge-visible features. The existing
`target_family_rates()` remains on its original task-level aggregate path, so
it is a clean scalar control; new callers can opt into the position map and
supply a vector of rates to `decide_route_positionwise()`.

### ARC consequence

The offline pipeline can now estimate and route lane value at the same scoring
granularity as ARC-AGI-2: each test output is an independent opportunity, while
task-level structure still informs the rate. This prevents easy/high-coverage
positions from subsidizing hard or structurally novel positions in the four-GPU
allocation decision.

### Limitation

The feature model is still a small hand-designed hierarchy and may not explain
the true task distribution. Position identity is retained for routing, not used
as a hidden-label proxy. Estimates must be fitted on leakage-safe folds and
validated across task splits before any scheduler flag is enabled.

## Iteration 118 — Per-test structural features, not copied task features

### Observation

The first positionwise projection retained the output index but still used the
same task-level maxima for every test input. On a multi-test task, this made
`q_j` identical by construction and reduced the new router to a mislabeled
scalar average.

### Construction

Added `task_position_features(task, test_index)`. It retains the task-wide
profile as context and adds area, palette, component-count, and square-shape
buckets computed from that specific visible test input. The calibration
projection now uses this feature map for each `(task_id, test_index)`.

### ARC consequence

The scheduler can distinguish, for example, a small single-object test input
from a large multi-component input within the same task and route a
complementary proposal lane accordingly. This aligns the calibration unit with
the official output-weighted score while preserving task-level information for
generalization.

### Limitation

The features remain coarse and hand-designed; they are routing covariates, not
a complete ARC representation. Adding too many per-position features risks
sparse-fold overfitting, so any richer descriptor must beat this baseline on
held-out groups with no coverage regression.

## Iteration 119 — Fit the same position features used at target time

### Observation

Adding position-level features only at target projection creates a train/target
feature mismatch: the calibrator has no position-bucket outcome counts, so its
hierarchical estimator silently falls back to the global/task-level rate. This
would make the apparent positionwise router more expressive than its learned
evidence.

### Construction

`fit_family_calibration()` now records feature outcomes from
`task_position_features(task, index)` for every labeled task-position. The
projection and fit paths therefore share the exact visible covariate map,
including task-wide context plus the specific test-input buckets. A regression
asserts that position-bucket outcomes are present in the fitted calibration.

### ARC consequence

The position-conditioned `q_j` values now represent learned structural
differences rather than untrained feature names. This closes a calibration
validity gap before any four-GPU scheduler can use the rates to choose a lane.

### Limitation

Per-position buckets increase the number of sparse cells. The existing
shrinkage is a pragmatic guard, not a proven hierarchical model, and feature
interactions are still omitted. Calibration must report support counts and
uncertainty by group before promotion.

## Iteration 120 — Preserve the scalar routing control

### Observation

Making positionwise covariates available must not silently alter the baseline
against which their value is measured. If the legacy scalar family-rate helper
also consumed the new position buckets, a later ablation could confuse a
feature-model change with a routing-policy change.

### Construction

Restored `target_family_rates()` to its original task-level feature path. The
new `target_family_position_rates()` is explicitly opt-in and is the only path
that uses per-test-input covariates. Both paths share the same fold-fitted
calibration object and remain leakage-safe.

### ARC consequence

The scalar router is now a clean control, while the positionwise router is a
single-variable extension that can be compared on identical candidate caches,
costs, and folds. This is essential for deciding whether the added scheduling
complexity actually buys output coverage toward the low-70s target.

### Limitation

The control-preserving split does not solve sparse calibration or establish
that position features improve prediction. It only makes that question
experimentally identifiable once the first full decoder cache is available.

## Iteration 121 — Posterior-gated Leg-C merge under a two-slot constraint

### Observation

The current merge promotes a demo-verified Leg-C output into `attempt_1` and
demotes the base `attempt_1` into `attempt_2`. If the base already had two
useful, complementary guesses, this necessarily discards its old second guess.
Demo verification proves consistency on visible examples, not hidden-test
correctness, so promotion is not automatically monotone under pass@2.

### Construction

Added `experiments/merge_policy.py`. Given a baseline pair and leakage-safe
output-class masses for all available candidates, `gated_pair_merge()` compares
the baseline mass with the best two-class proposal. It promotes only on a
strict configurable gain; equal mass preserves the baseline pair. Missing
baseline masses are treated as zero for staged calibration, and invalid or
duplicate pairs fail closed.

Counterexample: baseline masses `(0.45, 0.40)` and a verified newcomer `0.10`
make the forced replacement `(0.45, 0.10)` strictly worse. The safe production
design is therefore to emit family/lineage-normalized masses and run a unified
two-class selector, not to equate “verified” with “correct.”

### ARC consequence

This closes a subtle path by which a new solver family could lower score while
appearing to add a verified answer. Leg-C remains a candidate source; its
promotion value is measured against the full existing pass@2 pair at each
output position. The helper is not enabled in the notebook until masses are
available from a fixed shadow-fold calibration.

### Limitation

A posterior-gated merge can still be wrong when the calibration is
misspecified, and a strict gain gate may reject a genuinely correct rare class.
The next required evidence is an end-to-end fold replay comparing current
promotion, baseline preservation, and unified mass selection.

## Iteration 122 — Partial-cell fitness as an evolutionary search heuristic

### Observation

New external method evidence describes an ARC code-evolution population that
uses partial cell accuracy, mutation-strength diversity, and crossover to move
away from local optima before selecting two distinct outputs. The current
evolutionary prototype only counted demonstrations as wholly correct or wrong,
so near-miss programs supplied no gradient for candidate recall.

### Construction

`Fitness` now records cell matches and a larger-grid denominator in addition to
exact demonstration count. `pareto_frontier()` treats cell accuracy as a
secondary objective after exact-demo count and before MDL, while
`partial_cell_match()` penalizes shape mismatch and never counts padded
missing/missing cells as matches. Exact replay remains the only proof gate;
partial fitness only steers mutation populations.

### ARC consequence

A mutation that fixes a causal subregion can remain visible to the evolutionary
search even before it reaches exact-demo fitness, increasing the chance of
discovering a new verified output class. This is the portable part of the
external code-evolution idea. External transfer judging and hosted model calls
are excluded from the Kaggle notebook; any mutation/crossover policy must run
inside the self-contained budget.

### Limitation

Cell overlap is not semantic understanding and can reward locally plausible but
globally wrong programs. It is therefore deliberately subordinate to exact
demo correctness, MDL, and later hidden-test-free selector calibration. The
next test is a fixed-budget fold replay comparing exact-only evolution with
partial-cell steering at equal executor calls.

## Iteration 123 — Complementary bounded crossover for candidate recall

### Observation

One-edit mutation can require several sequential edits before reaching a
program that explains all demonstrations. Keeping only the current Pareto
frontier can also remove two partial programs whose useful clauses are
complementary. External code-evolution evidence reports crossover as a useful
escape operator, but the current prototype had no explicit recombination.

### Construction

Added `crossover_genomes()`, which enumerates deterministic one-point cut pairs
in both parent orders, deduplicates children, and enforces `max_steps`. It is
available through an explicit `include_crossover` flag in the evolution loop,
with a `max_parent_pairs` cap; the exact-only default remains unchanged.
Children still require execution on every demonstration; no child is trusted
because it resembles either parent.

### ARC consequence

Two partial hypotheses can now exchange prefixes/suffixes and expose a shorter
route to a demo-exact program, improving recall without an unbounded search
branch. The natural four-GPU mapping is to reserve a small recombination tranche
after diverse independent proposals, then retain only exact/MDL-competitive
children for test prediction.

### Limitation

One-point operation crossover assumes the genome ordering is meaningful; it can
produce syntactically valid but semantically incoherent programs. All cut pairs
can also grow combinatorially with population size, so a future scheduler must
cap parent pairs and compare new exact-class coverage per executor call.

## Iteration 124 — Bounded multi-radius mutation shells

### Observation

One-edit mutation is a local neighborhood. A correct program can be separated
from the current parent by a temporarily lower-fitness intermediate, so a
strict one-step Pareto loop can never propose it. Unbounded radius search would
violate the efficiency constraint and flood the verifier.

### Construction

Added `mutation_shell()`, a deterministic breadth-first union of one-edit
neighborhoods up to an explicit radius and candidate cap. `evolve_generation()`
accepts `mutation_radius > 1` only through this opt-in shell; first-divergence
guided repair rejects multi-radius requests and remains local. The cap makes
incompleteness visible rather than silently claiming exhaustive search.

### ARC consequence

The evolutionary lane can spend a small, pre-registered high-radius tranche on
hard tasks after ordinary mutations stall, giving it a route across neutral or
temporarily worse basins. The scheduler should compare exact-class coverage per
executor call and stop the shell when its marginal yield falls below an
independent proposal lane.

### Limitation

Canonical truncation is not an unbiased sample of the mutation graph, and
operation edit distance may not correspond to semantic distance. The shell is
therefore a recall proposal mechanism only; every result remains subject to
exact demonstrations, MDL, output quotienting, and pass@2 selection.

## Iteration 125 — Preserve unlabeled test-behavior diversity in the archive

### Observation

ARC demonstrations underdetermine the hidden rule. Two genomes can be equally
demo-exact but produce different outputs on the visible challenge inputs. The
previous Pareto relation allowed a shorter genome to dominate the longer one
solely by MDL, deleting a distinct pass@2 hypothesis before the output selector
could compare it.

### Construction

`Fitness` and `evaluate_genome()` now support an opt-in `probe_inputs` channel
that records each genome's frozen output signature on visible, unlabeled test
inputs. When both candidates carry probe signatures, MDL dominance is allowed
only within the same signature; distinct signatures survive as separate
frontier hypotheses. The default empty probe channel preserves prior behavior.

### ARC consequence

The evolutionary archive now protects the exact diversity that matters at
submission time: distinct candidate output classes on the hidden-input proxy,
without reading hidden labels or inventing semantic correctness. A future
selector can quotient these signatures by exact grid and feed them to the
family/lineage posterior and holistic judge.

### Limitation

Visible test behavior is an unlabeled proxy and can preserve many spurious
hypotheses; archive size must be capped and deduplicated by exact output class.
Probe execution also consumes time. The next empirical test is whether this
archive increases hidden-output coverage per executor call at fixed evolution
budget, rather than merely increasing frontier cardinality.

## Iteration 126 — Quality-gated behavioral diversity

### Observation

Behavioral diversity must not become an excuse to retain arbitrary garbage.
The first archive rule blocked all dominance whenever probe signatures differed,
which could preserve a low-demo-fitness program solely because it behaved
differently on an unlabeled input.

### Construction

The Pareto relation now allows a candidate with strictly higher exact-demo or
partial-cell fitness to dominate across probe signatures. Distinct signatures
only block a pure MDL tie among equally qualified hypotheses. This preserves
the useful underdetermination frontier while retaining the original
quality-vs-complexity tradeoff.

### ARC consequence

The archive protects plausible alternate hidden-input outputs without allowing
unbounded behavioral noise to crowd the evolution population. This makes
probe-diversity ablations interpretable: any retained alternate must at least
be competitive on visible evidence or complexity.

### Limitation

“Plausible” remains defined by visible-demo fitness and MDL, both imperfect
generalization proxies. The archive can still grow across many equally fit
signatures, so exact-output deduplication and a population cap remain required
before four-GPU deployment.

## Iteration 127 — Cap behavioral frontiers by probe-output coverage

### Observation

Quality-gated diversity can still create a large frontier when many equally
fit programs disagree on visible test behavior. Retaining all such programs
would consume verifier time and holistic-judge context, violating the same
budget that motivated the archive.

### Construction

Added `bounded_frontier()`. It first applies the quality/MDL Pareto filter, then
greedily selects at most `max_items`, maximizing newly covered
`(probe_position, output_signature)` pairs before exact-demo fitness, cell
accuracy, MDL, and deterministic tie-breaks. The cap is explicit and the
returned frontier remains reproducible.

### ARC consequence

The evolution lane can preserve complementary hidden-input proxy behavior under
a hard population bound instead of reverting to one modal program. This creates
a concrete interface for equal-call fold ablations: compare exact-class recall
as `max_frontier_items` varies, then feed only the retained quotient to the
pass@2 selector.

### Limitation

Greedy probe coverage is a surrogate for hidden-output coverage and has no
approximation guarantee for semantic ARC rules. A too-small cap can delete the
correct class, while a large cap wastes execution; the cap must be calibrated
on held-out tasks and kept separate from proof status.

## Iteration 128 — Real-data symbolic constraint envelope

### Observation

The local competition data is now present, so the symbolic property layer can
be measured on actual labels without any model call. The evaluation split has
172 test outputs. Under the current presets, baseline palette firing covers
155/172 (90.1%) but is only 70.32% precise; paranoid palette firing is 108/172
(62.8%) at 100% precision; strict is 117/172 (68.0%) at 99.15% precision.

### Construction

Ran `python -m symbolic.measure` on training and evaluation for baseline,
strict, and paranoid configurations, then replayed the joint property masks.
Strict shape firing is 111/172 (64.5%) at 99.10% precision, while strict
shape+palette firing overlaps on 88 positions and is jointly exact on 86/88.
The strict shape/palette union contains at least one exact property on 140/172
positions. The palette superset bound contains the labeled palette on all
172/172 evaluation positions (and 1075/1076 training positions).

### ARC consequence

The safe deployment envelope is not “use the most permissive predictor.”
Baseline palette output is too noisy to be a hard constraint. Strict
shape/palette predictions are a high-precision gate for 88 positions, while
the palette superset is a recall-preserving inclusion bound suitable for
filtering candidate outputs without excluding the truth on this evaluation
replay. These properties can prune candidate grids and route uncertain cases,
but they do not themselves solve the transformation.

### Limitation

This is property coverage, not exact output-class coverage; it cannot support a
70% leaderboard claim by itself. The one training bound violation warns that
even the apparent 100% evaluation bound needs a conservative fallback. The
next real-data milestone is to generate normalized candidate records from the
neural/DSL cache and measure whether these constraints improve exact recall
without rejecting correct candidates.

## Iteration 129 — Exact replay falsifies the standalone symbolic lane

### Observation

The real evaluation data allows the existing verified primitive and geometric
families to be tested at the actual output level. High-precision shape/palette
properties therefore cannot be mistaken for transformation recall.

### Construction

Ran the current finite verified primitive compiler, geometric D8 orbit
generator, and closed object-action executor on the 120-task evaluation set
(172 test outputs). Results: verified primitives emitted 0 records and covered
0/172 outputs; D8 emitted 1,376 records but covered 0/172; the closed object
executor verified 0 tasks and 0 programs. Replay pass@2 was 0.0 for the
primitive and geometric record sets.

### ARC consequence

These families are not GPU-worthy standalone proposal lanes for ARC-AGI-2.
Their safe value is upstream: property bounds can prune or route neural/code
proposals, and the executor can verify richer grounded programs once a model
supplies the missing correspondence/composition hypotheses. The dominant gap
is transformation candidate recall, not output ranking or primitive proof
strictness.

### Limitation

This replay does not test the neural NVARC/Nemotron candidates and does not
prove that a richer symbolic compiler has zero value. It does establish a hard
negative for the current finite libraries on the labeled evaluation split.
Future symbolic work must show nonzero exact output-class coverage before
receiving GPU or submission budget.

## Iteration 130 — Correspondence relaxation does not repair the object DSL

### Observation

The zero exact-recall result could still have been a search-cap artifact: the
closed executor initially retained only minimum-cost first-demo assignments and
rejected shape-changing effects. I therefore replayed it with `k` equal to 4,
16, and 64, both with and without shape transforms, and then repeated the
measurement over all retained correspondences rather than tied minima.

### Result

Every setting verified 0/120 evaluation tasks and emitted 0 exact programs.
The other symbolic profiles were similarly sparse: aligned graph traces
appeared on 9/120 tasks, only 3 had unique role hypotheses, and no aligned
proof passed the current grounded-effect gate. The relation-equation profile
found one task with a relation hypothesis and seven unique relation equations,
but no unique placement equation.

The object-delta profile explains why relaxing matching is insufficient. On the
same 120 tasks, 100 consistently contained object deletion across their demos
and 99 consistently contained object addition; these labels overlap heavily
with 37 consistent grid resizes, 12 recolors, 12 moves, and 6 transforms.
The label counts are therefore not a claim that the task is literally an
add/delete program: they show that the current connected-component extractor
frequently sees a global re-encoding as replacement of objects.

### Theoretical consequence

The closed object DSL is not merely under-searching correspondences. Its
inductive bias assumes that an output is a sparse edit of the input scene with
stable object identities. Many ARC-AGI-2 examples instead require a latent
rendering rule: the input is evidence about a generator, and the output is a
new grid rendered from that generator. A useful symbolic layer must therefore
support at least three coupled levels:

1. a scene parser that can retain grids, rows, columns, tiles, and objects as
   competing views rather than choosing connected components once;
2. a latent-rule compiler whose primitives include crop/split, recolor,
   repetition, projection, symmetry, and object composition; and
3. a renderer whose proof checks the complete output grid, including changed
   dimensions, instead of proving only object-local edits.

This is a representation theorem for the current codebase: increasing the
assignment beam or allowing shape transforms cannot recover a target class
that the executor cannot express. Search should be spent on alternative scene
factorizations and renderers before more role predicates or correspondence
ranking.

### Deployment rule

Do not allocate GPU or Kaggle runtime to the current closed-object family as a
standalone proposal lane. Keep its exact executor as a verifier for candidates
that arrive from a richer neural or program-induction generator. The next
CPU-only theory test is a small renderer grammar evaluated by complete-grid
replay, with candidate recall measured before any selector or leaderboard run.

## Iteration 131 — Finite complete-grid renderer basis is still insufficient

### Hypothesis

If the object-edit executor failed because it could not represent resizing and
global re-encoding, a small renderer grammar should recover at least a few
exact evaluation candidates. The probe used complete-grid transforms: dynamic
content crop, empty-border/line removal, adjacent-line deduplication, the D8
geometric orbit, a demo-fitted palette map, and one renderer-then-palette
composition. Every retained program had to replay every visible demo exactly.

### Result

The grammar verified 0/120 evaluation tasks, emitted 0 candidate records, and
covered 0/172 labeled evaluation outputs. Its four unit tests pass, so this is
not an implementation failure. Together with Iteration 130, the result rules
out both of the following as sufficient explanations: “the object matcher is
too narrow” and “the output merely needs a basic crop/resize/global palette
renderer.”

### Theoretical update

The useful abstraction is conditional rendering from a latent description, not
a fixed transform of the raw pixel array. ARC-AGI-2 candidate programs need to
be able to infer an intermediate representation (for example a symbol grammar,
ordered relation graph, or repeated motif), then render that representation
under input-dependent dimensions and colors. A finite unary operator list has
no place to store the inferred relation, so adding more unary operators would
mostly increase correlated false hypotheses.

The next symbolic experiment should therefore be a *typed two-stage grammar*:

1. infer a latent structure from each training pair (motif, sequence, object
   relation, or grid partition),
2. anti-unify those structures across demos, and
3. execute the shared renderer on each test input with exact shape and cell
   checks.

The acceptance metric remains exact candidate recall before any ranking. If a
typed latent grammar does not produce nonzero recall on a held-out shadow fold,
the effort should return to model-generated programs and use symbolic code only
for validation.

## Iteration 132 — Renderer grammar has training recall but zero transfer

### Measurement

To distinguish an incapable grammar from a split-specific failure, I replayed
the same finite renderer grammar against the 1,000-task training split, whose
test outputs are locally labeled, without changing its code or parameters.
It verified 14 tasks and recalled 13/1,076 outputs; the resulting exact
selector replay was 1.30%. On the 120-task evaluation split it verified 0
tasks and recalled 0/172 outputs.

### Consequence

The positive training result is not evidence for a deployable renderer. It is a
small, direct example of ARC distribution shift: a grammar can fit a handful
of visible transformations while having no support on the held-out task
families. This also validates the project's shadow-fold rule. Any new symbolic
operator or latent representation needs an untouched transfer gate, not just
demo-fit counts or training pass@2.

The practical target is now a *family-complete proposal distribution*: the
renderer grammar may remain a low-cost specialist, but the main 70%-regime
system must combine independent latent representations (grid grammar, object
relations, sequence/count algebra, and model-generated code). The proof layer
should reject candidates after complete-grid execution; it cannot create
coverage for a family absent from the proposal distribution.

### Deployment rule

Do not promote the finite renderer grammar as a scored submission leg. Keep it
as a cheap specialist only if it adds distinct output classes on a shadow fold;
otherwise its training recall is treated as overfit evidence.

## Iteration 133 — Target-visible structural shift is too large for pooled rates

### Measurement

Using only challenge-visible inputs, I compared the structural feature
distributions of the 1,000-task training pool and the 120-task evaluation pool.
Total variation distance was 0.4557 for test-grid area, 0.3487 for test
palette size, 0.2417 for component count, and 0.3403 for the number of test
outputs. Train/evaluation square-grid distance was only 0.0057. Evaluation
therefore concentrates on larger, more colorful, more multi-component, and
more frequently multi-test tasks, while preserving roughly the same square
versus non-square balance.

### Theoretical consequence

A pooled family success rate is not a stable estimate of target value. Let
`x` be the visible structural bucket and `l` a candidate lane. The correct
target estimate is

`P_target(success | l) = sum_x P_target(x) P(success | l, x)`,

not the source-weighted `sum_x P_train(x) P(success | l, x)`. The difference
is material whenever the lane is sensitive to grid size, palette, or scene
complexity, which is exactly expected for token-budgeted code generation and
object-based search.

The existing position-conditioned router is therefore necessary but not
sufficient: its calibration counts must be stratified by visible target
features, with shrinkage to pooled rates for sparse buckets and an explicit
unknown reserve for unseen buckets. Large/high-palette/multi-component
positions should receive more time only when the candidate lane has measured
conditional recovery there; otherwise they should trigger complementary
proposal families rather than blindly increasing the same model's budget.

### Deployment rule

Do not use unweighted training averages to allocate the four L4 workers. Fit
lane rates on source buckets, reweight by target-visible bucket frequencies,
and require an untouched shadow fold to show that the reweighted policy does
not increase unresolved positions. This is a calibration change, not evidence
that any current symbolic lane should be promoted.

## Iteration 134 — Covariate reweighting needs a zero-support safeguard

### Measurement

I conditioned the finite renderer's labeled training recall on the same
challenge-visible buckets used for target shift. Its output recall was 7.03%
for test area `le_25`, 0.93% for `le_100`, 0.22% for `le_400`, and 0% for
`gt_400`. By test count it was 1.40% on one-test tasks and 0% on two-, three-,
and four-test tasks. Component-conditioned recall was 3.17% for `le_1` and
0% for `gt_6`. The evaluation pool is concentrated in exactly the weak
regions: 74/120 tasks have `gt_400` test area and 49/120 have multiple test
outputs.

### Theoretical consequence

Importance weighting is valid only under conditional stability:

`P_target(success | x, lane) = P_train(success | x, lane)`.

The measured zero-support strata make that assumption unusable for the
renderer. Reweighting a small positive source rate into a target region where
the lane has no demonstrated support creates false confidence. A safe target
estimate needs a gated form:

`q_target(l,x) = q_pool(l)` when support(`l,x`) is below a minimum;
`q_target(l,x) = BetaPosterior(l,x)` otherwise.

Even the pooled fallback should not authorize promotion when a lane's target
mass is mostly outside its supported strata. Such positions should be routed
to an independent proposal family or receive an explicit unknown reserve in
the pass@2 posterior. This is the allocation analogue of candidate-support
coverage: no amount of selector calibration can turn an unsupported lane into
coverage.

### Deployment rule

For each candidate family, report target-visible mass in supported, weak, and
unseen strata before allocating GPU seconds. Require a positive shadow-fold
gain after this support gate. A family with zero target support remains a
diagnostic specialist, never a production fallback merely because its pooled
training rate is nonzero.

## Iteration 135 — The labeled evaluation fold is not the hidden target prior

### Measurement

I compared the challenge-visible feature distributions of all three available
partitions, without reading hidden outputs. The hidden queue contains 240 tasks
and 259 test outputs. Training-to-hidden total variation is low: 0.0492 for
test area, 0.0478 for component count, 0.0445 for palette size, and 0.0033 for
test count. Evaluation-to-hidden variation is much larger: 0.4500 for area,
0.1958 for components, 0.3500 for palette, and 0.3375 for test count.

The visible composition makes the distinction concrete. Evaluation has 74/120
tasks with test area `gt_400` and 46 two-test tasks; hidden has 40/240
`gt_400` tasks and 15 two-test tasks. Hidden is therefore structurally much
closer to the training pool, although it remains unlabeled and must not be
used for supervised tuning.

### Theoretical consequence

The 120-task labeled evaluation set should serve as an adversarial stress fold
for proposal-family transfer, not as an unbiased estimate of hidden-task lane
rates. Let `S`, `E`, and `H` denote source training, labeled evaluation, and
hidden target. A scheduler that estimates `q(l | E)` and deploys it on `H`
incurs a transport error proportional to the feature discrepancy between `E`
and `H`; in this data that discrepancy is large. Conversely, source-to-hidden
covariate reweighting is closer to the correct deployment prior, but its
conditional-stability assumption still needs a shadow fold.

The resulting experimental protocol has two axes:

- use training folds and the labeled evaluation set to falsify transfer and
  measure candidate recall under hard shift;
- use hidden inputs only for prior estimation and queue allocation, never for
  output-dependent selection or operator tuning.

This separates “does the idea generalize to difficult public tasks?” from “is
the final four-L4 schedule spending time according to the actual queue?” A
family that wins only on the evaluation stress fold may still be useful for
robustness, but it should not displace a source-supported family on hidden
allocation without evidence from held-out training folds.

### Deployment rule

Maintain separate calibration reports for source-fold, evaluation-stress, and
hidden-input priors. The production scheduler should use hidden-input
frequencies for cost allocation, while promotion requires non-regression on the
evaluation stress fold and an untouched source shadow fold. No hidden solution
labels are needed for this policy.

## Iteration 136 — Typed motif panels recover a small hidden proposal family

### Construction

I implemented a bounded latent-structure probe in which an input grid is a
motif and the output is partitioned into equal-sized motif panels. The compiler
infers the panel transform matrix, then retains only shared constant,
row/column-template, parity, or checker rules that replay every demo exactly.
The representation is typed (`motif -> panel`) and the renderer checks the
complete output grid, including dimensions.

### Measurement

On the labeled training split the probe verified 14 tasks, emitted 31 test
records, and recalled 12/1,076 outputs (1.21% exact replay). On the labeled
evaluation split it verified 0/120 tasks and recalled 0/172 outputs. On the
unlabeled hidden challenge inputs it produced 5 demo-verified records across
2/240 tasks: `00576224` has a 3x3 row-template with alternating horizontal
mirrors, and `3c9b0459` has a verified 1x1-panel 180-degree rotation. The
hidden outputs were not consulted.

### Theoretical consequence

This is the first symbolic lane whose hypothesis is both more expressive than a
unary renderer and directly aligned with a visible hidden-task construction.
It still has no evidence of broad transfer, so its value is not a standalone
score claim. Its proper role is a high-precision, low-cost candidate source:

`hidden input -> motif-panel program -> exact test render -> candidate class`

The 3x3 alternating-panel rule also illustrates why ordinary D8/tile search
misses a valid program: the latent object is not one transformed grid but a
panel-valued function whose transform depends on panel row. Typed structure
allows that dependency to be represented and verified without allowing
arbitrary code.

### Deployment rule

Add motif-panel candidates to the proof-carrying pool only when all demos replay
exactly. Preserve distinct output classes and provenance, but never promote
them solely because they are hidden-input verified; require shadow-fold
non-regression and a candidate-class gain. Keep the generator CPU-only and
bounded, using it as a specialist alongside the neural proposal families.

## Iteration 137 — Local cellular rules verify often but abstain on target tests

### Construction

I added a fail-closed `grid -> grid` cellular transducer. It learns a finite
mapping from a local cross or square neighborhood to an output color, with an
explicit boundary sentinel. The rule is accepted only after exact replay of
all demonstrations; a test grid is rejected if even one context was unseen.
This is a bounded local-rewrite hypothesis, distinct from object identity and
motif-panel rendering.

### Measurement

For maximum radius 1, the cross/square family verified 97/1,000 training tasks,
4/120 labeled evaluation tasks, and 22/240 hidden-input tasks. Exact test
execution was much narrower: it emitted 12 training records, 0 evaluation
records, and 4 hidden records. Labeled evaluation recall remained 0/172. A
radius-2 search increased verification to 253 training, 19 evaluation, and 56
hidden tasks, but emitted no additional exact evaluation records and only the
same 4 hidden records.

Combining radius-1 cellular records with motif-panel records produced 18/1,076
training recall, 0/172 evaluation recall, and 9 hidden-input records across 4
tasks with no task overlap between the two specialist families. The hidden
records are retained without reading hidden solutions.

### Theoretical consequence

Demo verification alone is a weak signal. A local rule can be perfectly
consistent on training grids while having no defined action on the test grid;
the relevant quantity is executable candidate support, not the number of
verified programs. Increasing neighborhood radius expands the apparent
version space faster than it expands test support, so it is a poor standalone
use of search budget.

The correct interpretation is a two-stage uncertainty decomposition:

`P(correct) = P(context coverage) * P(rule correct | covered contexts)`.

The proof gate controls the second factor on demonstrations, while fail-closed
execution exposes the first factor on each test input. A production scheduler
should measure both and route low-coverage positions to a different proposal
family. This formalizes why “more exact-fit symbolic rules” did not approach
the top-8 target.

### Deployment rule

Keep the cellular transducer as an optional specialist and append only
complete, executable records. Do not spend additional radius/search budget
unless a shadow fold shows a positive gain in test-context coverage and exact
output classes. Preserve its records as a complementary family candidate, not
as a replacement for neural generation.

## Iteration 138 — Color-role quotienting increases local-rule support

### Construction

I extended the cellular transducer with an opt-in color-role quotient. Each
input palette is canonicalized by background, frequency, first occurrence, and
color tie-break; the learned rule maps local contexts to canonical roles, which
are lifted back to the test grid's actual colors. Training examples that use
different numeric colors can therefore share one structural rule. New output
colors not present in the source palette remain unsupported and fail closed.

### Measurement

With radius 1 and both cross/square neighborhoods, raw plus role-normalized
rules verified 105 training tasks, 4 evaluation tasks, and 24 hidden-input
tasks. The more important executable counts were 35 training records, 0
evaluation records, and 8 hidden records; labeled evaluation recall stayed
0/172. Compared with the raw-only radius-1 run, training recall improved from
6 to 14 outputs and hidden records from 4 to 8. Radius 2 increased verified
tasks but did not increase executable evaluation or hidden records. Some raw
and role variants produce identical output classes, so provenance-level
diversity is not the same as semantic diversity.

### Theoretical consequence

Color IDs are nuisance variables for a substantial class of ARC rules. The
quotient operation reduces hypothesis fragmentation by identifying grids under
task-local color permutations, while the lift step preserves exact output
colors. Formally, if `pi` is a palette permutation and `f` is equivariant,

`f(pi(x)) = pi(f(x))`.

Learning `f` on the quotient space can increase cross-demo support without
changing the rendered answer. But quotienting is safe only when the task rule
is color-equivariant; tasks whose semantics depend on a distinguished color
role require a separate conditioned family. The candidate pool must therefore
retain both hypotheses, deduplicate by output class, and use a class-level
coverage metric.

The experiment also reinforces the coverage factorization from Iteration 137:
verification confidence improved substantially, yet evaluation test support
remained zero. Nuisance-invariant representation helps only after the latent
rule family reaches the target input's context manifold.

### Deployment rule

Keep role-normalized cellular rules as an independent, low-cost specialist.
Emit only executable records, collapse duplicate output classes, and attach the
color-equivariance assumption to their proof metadata. Do not let improved
demo verification or additional radius consume GPU time unless a shadow fold
shows new test-context and exact-class coverage.

## Iteration 139 — Consolidated four-L4 design after support falsifications

### Design objective

The current evidence does not justify claiming a 70% solver. It does identify
the only architecture compatible with the pass@2 metric and the observed
failure modes: maximize independent exact candidate support first, then spend
the final two slots on output classes rather than correlated samples.

### Proposed Kaggle schedule

The CPU phase runs all bounded proof-gated specialists before GPU work: fixed
primitives, closed object actions, motif panels, and cellular rules. Their
records are appended only when they execute on every requested test position.
They can save model time on solved positions but never displace a baseline
candidate merely because their demo proof is shorter.

The four L4 workers then operate as independent proposal channels under the
12-hour wall clock:

1. the reproduced NVARC/Qwen grid decoder with its established TTT and DFS
   baseline;
2. an independent NVARC seed/augmentation/decode lineage, preserving separate
   correlation metadata;
3. a cheap recursive/TRM proposal channel, used only where held-out marginal
   candidate coverage justifies its cost; and
4. bounded verified program induction/evolution, using the available coder
   model or cached proposals, with exact sandbox replay and one repair budget.

The scheduler uses hidden-input structural frequencies only to estimate queue
mass. For lane `l`, visible position `j`, and already-spent count `n`, its
priority is

`support(l,j) * q(l | x_j) * (1-q(l | x_j))^n / cost(l,j)`.

Here `support` is an explicit 0/weak/strong gate derived from held-out folds;
it prevents a high pooled rate from being applied to an unsupported target
stratum. The plan is greedy and anytime, so an interrupted run still leaves a
valid candidate pool.

### Proof and selection boundary

Every record carries: exact demo replay status, complete-grid execution status,
output hash, family, lineage/correlation group, MDL, and bounded diagnostics.
Hard-invalid records are removed before aggregation. Classes are deduplicated
within correlated lineages; family mass is normalized before class mass is
computed. The selector independently chooses the top two output classes for
each test position, while shared program vectors may inform the posterior.

The core upper bound is immediate. If `C_j` is the candidate class set for
position `j` and `y_j` is the truth, then for any selector `S_j` with at most
two elements,

`1[y_j in S_j] <= 1[y_j in C_j]`.

Therefore output candidate coverage is a hard ceiling on pass@2. Selection
work is justified only after measuring the same-pool coverage ceiling and the
conditional recovery of the selector. This prevents another round of
optimizing a selector over empty or unsupported symbolic families.

### Promotion gates

A candidate family may enter a scored notebook only if it satisfies all of:

- nonzero exact output-class recall on an untouched source shadow fold;
- positive marginal coverage on the labeled evaluation stress fold, or a
  documented hidden-input specialist with no regression to the incumbent;
- no increase in unresolved positions or hard-invalid vectors;
- measured marginal gain per second under the four-L4/12-hour manifest; and
- deterministic packaging, offline execution, and official two-attempt schema
  validation.

The current motif-panel and cellular families fail the stress-fold recall gate,
so they remain optional CPU specialists. The highest-value untested experiment
is now a fixed-candidate-cache ablation of independent model lineages and the
holistic class selector; it directly measures the route to the 70s without
spending Kaggle quota on a speculative symbolic rewrite.

## Iteration 140 — Bounded arbitrary panel templates recover one more hidden task

### Construction

The hidden-input audit found one shared 2×2 D4 panel whose transform matrix was
not row-, column-, or checker-parity separable. I extended the motif grammar
with an explicit `matrix_template` rule, capped at 4×4 panels. The matrix is
inferred from the first demo, must be identical across every other demo, and
still requires exact complete-grid replay. This is a finite latent template,
not arbitrary program memorization.

### Measurement

The extended grammar verifies 20 training tasks, emits 47 records, and recalls
17/1,076 training outputs (1.67% exact replay). It still verifies 0/120
labeled evaluation tasks and recalls 0/172 evaluation outputs. On hidden inputs
it verifies 3/240 tasks and emits 8 executable records, adding `0c786b71` to
the earlier `00576224` and `3c9b0459` panel candidates. Hidden outputs remain
unread.

### Theoretical consequence

The gain is from representing the latent relation “panel index → motif group
element,” not from adding another pixel transform. The explicit matrix rule
shows a useful boundary: panel transformations form a small finite algebra for
some tasks, but arbitrary matrices overfit single demonstrations unless their
matrix and panel factor repeat across examples. A production compiler should
therefore attach a template-complexity cost and retain the generalization
certificate (shared factorization plus shared matrix), rather than treating
every observed panel as a reusable program.

The persistent 0/172 evaluation result is equally important. The evaluation
stress fold contains no task in this panel-factor family, so its value is a
hidden-target specialist only. It cannot be used to argue that expanding a
single symbolic family moves the system toward 70%.

### Deployment rule

Keep `matrix_template` CPU-only, bounded to 4×4, and proof-gated. Append its
records as a separate motif family with class-level deduplication. Do not
increase the cap or promote the family without a shadow-fold candidate-class
gain and a no-regression check against the incumbent pool.

## Iteration 141 — Stabilizer-aware D4 names recover the fourth panel family

### Observation

The hidden task `3af2c5a8` has a shared 2×2 panel transform rule, but one demo's
motif is vertically symmetric. Deduplicating D4 transformations by rendered
grid caused the name `flip_v` learned from an asymmetric demo to disappear on
the symmetric one, falsely rejecting the shared rule.

### Correction

The motif grammar now retains all D4 group names and quotients only by rendered
equality at the output-candidate boundary. A program learned with a named group
element is therefore executable on inputs with a nontrivial stabilizer. The
matrix-template rule remains capped at 4×4 and must replay every demo exactly.

### Measurement

After the correction, motif-panel replay verifies 25 training tasks, emits 62
records, and recalls 25/1,076 training outputs (2.42%). It still verifies 0/120
labeled evaluation tasks and recalls 0/172 evaluation outputs. On hidden inputs
it verifies 4/240 tasks and emits 9 executable records, adding `3af2c5a8` to
the three previously identified panel tasks. Hidden outputs remain unread.

### Theoretical consequence

Latent transformation induction must quotient by the stabilizer of the current
input, not erase group elements before cross-example anti-unification. If `G`
is the transformation group and `H_x = {g in G : g(x)=x}`, observations are
equivalence classes in `G/H_x`; program names may differ while their rendered
actions agree. This is a small but general lesson for ARC: nuisance symmetry
should reduce semantic duplicates at selection time, while preserving the
generative vocabulary needed for transfer across examples.

### Deployment rule

Keep the full named D4 vocabulary inside motif induction, attach stabilizer
metadata to the proof, and deduplicate only complete output classes. The lane
remains hidden-specialist/CPU-only because its evaluation stress-fold support is
zero.

## Rule revalidation context — Official submission constraints

### Rule check

The current official Kaggle pages confirm that a scored submission must be a
notebook producing `submission.json`, with CPU or GPU runtime no longer than 12
hours and internet disabled. One submission is allowed per day and up to two
final submissions may be selected. The competition exposes L4×4 machines with
96 GB pooled GPU memory, but they consume quota at twice the older accelerator
rate. External data and pretrained models are allowed when freely and publicly
available and otherwise compliant with the competition's open-source rules.

The public leaderboard is explicitly calculated from approximately half of the
test data; final standings use the other half. The current public head remains
72.08 and 70.42, but those values are not private-score evidence.

### Algorithmic consequence

The four-L4 design must be an anytime notebook, not a research loop that relies
on later interaction or network calls. All CPU specialists, model weights,
candidate records, verifiers, and selector code must be attached before the
rerun. The 12-hour boundary is a hard optimization constraint, while the
twice-quota rule makes uncontrolled submissions especially expensive.

The two-final-submission limit also changes the endgame: maintain two
independently selected output portfolios only after fixed-cache replay shows
complementary private-risk profiles. A public-score improvement alone is not a
reason to replace the primary portfolio, because the public board is a partial
sample and can reward evaluation-fold overfit.

### Deployment rule

Treat the official rules as a release gate: offline packaging, runtime envelope,
schema validation, and open-source/model-license audit must pass before any
Kaggle commit. Use the public evaluation fold for stress testing and the hidden
challenge inputs only for unlabeled prior/cost allocation.

## Iteration 142 — Hidden specialist union is complementary but very small

### Measurement

With the stabilizer-aware motif grammar and radius-1 raw/role cellular rules,
the hidden-input-only specialist pool contains 17 raw records across 7 tasks:
9 motif records on 4 tasks and 8 cellular records on 3 tasks. The task sets are
disjoint, and semantic output-class deduplication leaves 7 classes across 7
test positions (one class per position). No hidden solution was read.

### Theoretical consequence

The two representations are genuinely complementary at the proposal level, but
their union covers only 7/259 hidden test positions at the executable-record
boundary, and correctness is not observable offline. This is a useful upper
bound on their standalone value: even if every emitted hidden candidate were
correct, the specialist pool could not approach the 70s. Their value is
therefore strictly additive and opportunistic—free CPU candidates that may
rescue positions before the main model, not a replacement for high-throughput
neural proposal coverage.

The distinction between raw records and semantic classes also matters for
pass@2. The role/raw cellular variants and multiple motif derivations can be
different proofs of the same grid; counting them as independent votes would
overstate confidence without increasing the official success event. The
correct merge unit is `(task, test position, output hash)` with provenance
retained underneath.

### Deployment rule

Package the union as an optional CPU pre-pass with a hard cost cap and class
deduplication. If a future scored notebook uses it, report its marginal gain
against the incumbent candidate cache and never credit hidden-input verification
as correctness. GPU time remains reserved for proposal families capable of
covering the remaining 252 hidden positions.

## Iteration 143 — Official submission constraints revalidated

The rule audit above is now recorded as release-gate context rather than as a
chronologically misplaced experiment. The actionable result is unchanged:
package the solver as an offline notebook, enforce the 12-hour runtime and
submission schema, and treat public-board movement as weak evidence until a
fixed-cache private-risk analysis exists.

## Iteration 144 — Finite row/column sequence transducers

### Hypothesis

Some ARC tasks are generated by treating a grid as an ordered sequence of
rows or columns, then applying a global order statistic: lexicographic order,
non-background density, within-line value sorting, or alternating reversal.
These operations are cheap, shape-preserving, and compositional candidates for
an exact verifier.

### Measurement

The bounded family contains 14 deterministic programs and rejects any shape
changing demo. Exact replay found 2/1,076 training output positions and
0/172 labeled evaluation positions. It emitted 2 training records and 1
hidden-input record on task `1e0a9b12`; the surviving training task is a
column-value ordering rule, while the other is alternating column reversal.
The hidden record is a duplicate of the same visible task family, not evidence
of hidden-label correctness.

### Theoretical consequence

Sequence factorization is a valid low-description-length primitive, but its
observed support is nearly disjoint from the stress fold. The result also
illustrates a subtle implementation requirement: density must be recomputed
from each current input's background color at execution time. Capturing the
demo background in the program would silently break color-role transfer.

### Deployment rule

Retain the sequence lane as a CPU specialist with semantic output deduplication
and no standalone promotion. Its cost is negligible, but its measured
evaluation support is zero; any value must come from complementary candidate
coverage in a fixed cache.

## Iteration 145 — Totalizing partial cellular rules

### Hypothesis

A demo-verified local rule is a partial function on neighborhood contexts. On
the test grid, unseen contexts might be completed by a conservative default:
preserve the center cell, paint the current background, or emit the learned
output-majority color. This could provide a constrained scaffold for a later
neural or symbolic completion stage.

### Measurement

For radius-1 raw and color-role cellular programs, the three fallback policies
produced 786 training records across 110 positions and 24 evaluation records
across 7 positions. At least one completed candidate matched 27/1,076 training
outputs but 0/172 labeled evaluation outputs. On hidden inputs the completion
lane emitted 162 records across 25 positions, versus 8 records for the
fail-closed lane; hidden correctness was not inspected.

### Theoretical consequence

Totalizing a partial proof increases proposal coverage but does not preserve
the proof's correctness guarantee. The relevant decomposition becomes

`P(correct completed rule) = P(rule correct | covered) * P(default correct | unseen)`.

The second factor is not controlled by demo exactness and collapsed completely
on the evaluation stress fold. A fallback should therefore be treated as a
candidate scaffold or a neural conditioning signal, never as a verified answer.

### Deployment rule

Do not promote completed cellular outputs as standalone pass@2 attempts. They
may be retained in an internal research cache for conditional fusion, but only
if a fixed shadow fold demonstrates positive marginal class recall after
selection and no regression in the exact specialist pool.

## Iteration 146 — Dataset identity overlap is a contamination hazard

### Audit

All 240 task IDs in `arc-agi_test_challenges.json` are also present in the
local 1,000-task training challenge and solution files, with identical train
demos and identical test inputs. This was established by comparing task
structure and input grids only; no hidden test output value was used to select
or score a candidate.

### Interpretation

The overlap is not safe evidence for the competition solver. If a solution file
is joined by task ID, its labeled test outputs become an answer oracle for the
nominal hidden file. The research harness must therefore keep hidden inputs
strictly label-free and reject any hidden-vs-solution join, even when the IDs
match. This audit also explains why hidden-input specialist counts can be
reproduced from the training demonstrations without implying hidden accuracy.

### Deployment rule

Use the 120-task labeled evaluation fold for all correctness measurements. Use
the 240-task hidden file only for unlabeled feature, runtime, and candidate-cost
allocation. Add an explicit overlap/solution-join guard before any future
hidden analysis or notebook packaging.

## Iteration 147 — Algebraic cross-cellular rules are too restrictive

### Hypothesis

The poor transfer of finite neighborhood lookup might be caused by sparse
context coverage rather than by locality itself. A more mathematical family
could map binary foreground occupancy in the five-cell cross neighborhood to
two output colors using center, parity, threshold, or exact-count predicates.
Such predicates generalize to unseen raw color contexts by construction.

### Measurement

The bounded family contained center/inversion, parity, five thresholds, and six
exact-count predicates, with a strict two-output-color fit and complete-grid
demo replay. It verified 0/1,000 training tasks, 0/120 evaluation tasks, and
0/240 hidden-input tasks; it emitted no records. The replay harness therefore
returned 0/172 evaluation recall and no nontrivial candidate support.

### Theoretical consequence

The result separates two failure modes: lookup tables can fail because their
context domain is under-covered, while algebraic occupancy rules can fail
because the chosen sufficient statistic discards object identity, color role,
or geometry. Increasing local algebra without restoring those latent variables
is not a path to broad ARC transfer.

### Deployment rule

Retain the family only as a falsified baseline for future representation
comparisons. Do not spend GPU or search budget on more thresholds, radii, or
Boolean combinations until a task-level abstraction identifies when occupancy
is a sufficient statistic.

## Iteration 148 — Hidden-label join guard

The label-free protocol is now executable: `experiments/label_guard.py`
rejects any solution mapping whose task IDs overlap a hidden challenge mapping,
while allowing unlabeled analysis and disjoint evaluation labels. Three unit
tests cover the safe, disjoint, and overlapping cases. The full harness passes
378 tests. This guard is deliberately conservative because the local bundle's
hidden IDs overlap the labeled training IDs exactly.

The guard does not alter candidate generation or claim a score. Its purpose is
to make the research boundary mechanically enforceable before future hidden
feature audits, cache construction, or notebook packaging.

## Iteration 149 — Candidate-support ceiling and four-L4 allocation theorem

### Measurement

The current hidden-input-only symbolic union—motif panels, fail-closed
cellular rules, totalized cellular completions, sequence transducers, and the
algebraic cellular family—covers 29/259 test positions with at least one
candidate class and contains 78 semantic classes. The Boolean family adds
zero. The union therefore has an optimistic upper bound of 11.20% even if
every emitted class were correct. At the current public-score scale, 70.42%
requires at least 183/259 correct outputs and 72.08% requires at least 187/259.

### Theorem-shaped deployment rule

Let `C_j` be the set of output classes generated for position `j`, and let
`Z_j` be its hidden truth. The pass@2 objective is

`sum_j 1[Z_j in {attempt_1(j), attempt_2(j)}]`.

For a new proposal channel `f`, its only guaranteed benefit over an incumbent
pool is the marginal class event `Z_j in C_f(j) minus C_old(j)`. Therefore the
correct anytime allocation statistic is expected unique-class gain per unit
cost,

`rho_f = sum_j P(Z_j in C_f(j) \ C_old(j) | x_j) / cost_f(j)`,

with probabilities calibrated on task-disjoint source folds and reweighted by
hidden-input features only for cost allocation. Raw sample count, number of
verified programs, and repeated copies of an existing output class are not
valid substitutes for `rho_f`.

### Four-L4 consequence

Run the CPU proof-gated specialists first, then shard GPU generation by the
highest remaining `rho_f` tranche. Keep the incumbent best class in
`attempt_1`; choose `attempt_2` greedily from the highest expected marginal
class, subject to a different-lineage constraint when the alternatives are
correlated. This makes selection monotone in candidate coverage and prevents
an attractive but redundant symbolic family from displacing a neural class.
The theorem does not manufacture probabilities; it identifies the sufficient
statistics that the missing fixed-cache ablation must estimate.

### Release gate

No Kaggle run is justified by the current symbolic union alone. The next
high-value artifact is a fixed-cache replay of the four proposal lineages,
with per-position output hashes, lineage correlation, runtime, and unique
class recall recorded before any new GPU allocation or scored submission.

## Iteration 150 — Independent modalities, then holistic judging

### External evidence

The ARC Prize technical report identifies per-task refinement loops as the
defining successful pattern on ARC-AGI-2-era systems. A newer modality-driven
study reports a stronger decomposition: independent text, image, and code
proposal channels followed by a context-preserving holistic judge, with the
explicit warning that prescriptive iterative refinement can reduce hypothesis
diversity. These claims are method evidence, not a Kaggle score guarantee.

### Theoretical design

Let each proposal lineage `l` emit a set of complete output classes and an
execution trace `T_l`. Generation should be conditionally independent as far
as the hardware permits: do not feed a failed candidate back into every
lineage, because shared error feedback increases correlation and lowers the
effective sample size. The judge should receive the *joint* shortlist

`J = {(class, lineage, trace, demo-diff, shape, palette)}`

and choose two output classes, not two textual explanations. This allows a
minority class with a globally coherent trace to defeat a locally popular but
correlated class.

### Four-L4 implementation mapping

1. Run CPU proof specialists and normalize all outputs by semantic hash.
2. Use separate GPU channels for baseline numeric TTT, independently seeded or
   augmented TTT, coder/DSL synthesis, and a structurally different proposal
   mode. Freeze each channel's raw candidate cache before judging.
3. Build a compact evidence card per output class: all-demo exactness or
   partial-cell score, shape consistency, palette roles, augmentation
   agreement, lineage ID, and first-divergence coordinates.
4. Run one joint judge pass over the top classes per position or task. Preserve
   the incumbent class in attempt 1 unless calibrated posterior mass falls;
   select attempt 2 by marginal class gain and lineage decorrelation.
5. Permit refinement only for candidates whose execution diff localizes a
   repair. A failed trace is a diagnostic input, not a global instruction to
   collapse every branch toward the same patch.

### Release gate

This architecture is compatible with the offline notebook and 12-hour
anytime constraint documented in the official rules, but it requires the
missing fixed-cache ablation before implementation choices can be promoted.
The ablation must report unique output-class recall, pairwise lineage
correlation, judge ranking recovery, and wall-clock cost on task-disjoint
folds. A higher raw candidate count without those measurements is not evidence
for top-8 progress.

## Iteration 151 — Fixed-cache ablation seam implemented

The missing empirical gate is now executable in
`experiments/fixed_cache_ablation.py`. Given a frozen candidate cache, it
reports raw records, positions with support, semantic output classes,
per-family counts, same-position cross-family class Jaccard overlap, and—only
when a disjoint labeled solution mapping is explicitly supplied—candidate
recall and selected pass@2 score. Three tests cover duplicate-class collapse,
labeled recall, and label-free inventory. The full suite now passes 381 tests.

No neural cache was fabricated and no hidden solution was supplied. This is a
measurement seam, not an accuracy claim; it makes the next four-lineage run
auditable and prevents raw sample volume from being mistaken for useful
coverage. Running it on the current symbolic hidden-input inventory reports
180 raw records, 29 supported positions, and 78 output classes; cellular
completion overlaps the fail-closed cellular classes at only 0.041 Jaccard,
while the other family pairs have zero same-position class overlap.

## Iteration 152 — Output-value task scheduling

### Hypothesis

The current four-worker notebook queue spends time in task order rather than
in expected score order. Because the official metric is additive per test
output, a task with multiple unresolved outputs can be more valuable than a
cheaper one-output task. The correct task utility for a proposal lane is the
sum of position-level unresolved mass, novelty rate, and selector recovery,
divided by measured task cost.

### Construction

`experiments/task_value_scheduler.py` now implements this policy. It sorts
tasks by calibrated expected output gain per second, then packs them onto four
workers using the shortest current load while respecting each worker's wall
clock. Three tests cover additive output utility, density ordering with worker
balancing, and probability validation. The full suite passes 384 tests.

### Proof obligation

For a task `t` with positions `j`, the lane's expected marginal value is

`V(t) = sum_j u_j * n_j * r_j`,

where `u_j` is unresolved pass@2 mass, `n_j` is the chance of discovering a
new correct class, and `r_j` is selector recovery. For a non-preemptive task
with cost `c_t`, the greedy density `V(t)/c_t` is the correct first-order
ordering; bin packing handles the four-worker wall-clock constraint. The
statistics must be learned from task-disjoint folds, and the planner must
retain an unknown reserve when estimates are weak.

### Deployment rule

Replace key-order or cost-only queueing only after a fixed-cache replay shows
that value-density scheduling increases completed correct outputs at equal
wall-clock. A schedule that merely completes more tasks is not sufficient;
the measured target is output-level score per 12-hour run.

## Iteration 153 — Literal object deletion is not the observed delete family

### Hypothesis

Because the evaluation delta profile contains 100 consistently classified
`object_delete` tasks, a strict component-erasure renderer might recover a
large high-precision lane. The compiler derived changed cells directly from
each demo, required unchanged dimensions/background, and anti-unified
color-blind shape, area, bounding-box, and border guards.

### Falsification

The renderer compiled 0/1,000 training tasks, 0/120 evaluation tasks, and
0/240 hidden-input tasks. A broader pair-level audit found only 6 pure single-
component deletions among 3,232 training demos and 1 among 358 evaluation
demos; even allowing unions of deleted components found only 17 and 2 pairs,
respectively. The `object_delete` label is therefore a correspondence-level
diagnostic, not evidence that the output is a literal subset of the input.

### Theoretical consequence

ARC object counts are not sufficient statistics for object actions. A task can
delete a source correspondence while simultaneously transforming, merging,
or regenerating pixels, causing a naive delete renderer to erase the wrong
scene factor. The next object-centric compiler must model a complete relation
graph and render all changed objects jointly; adding more deletion guards would
only optimize a mislabeled subfamily.

### Deployment rule

Discard the literal-delete lane from promotion and keep it only as a negative
control. Do not broaden it to arbitrary multi-delete subsets. Return to
relation-graph synthesis with explicit source/target object correspondence,
joint rendering, and a candidate-support measurement on the evaluation fold.

## Iteration 154 — Scene-graph grammar as the next symbolic boundary

### Representation

Represent each parsed scene as a labeled graph `G=(V,E)`. A node stores
color-role, shape, area, bounding box, and anchor; edges store bounded
relations such as left/right, above/below, same-row, same-column, touching,
and relative displacement. Keep multiple parsers alive—objects, rows,
columns, and equal panels—instead of committing to connected components before
the task type is known.

For a source/target correspondence `pi`, derive a graph rewrite containing
node actions (`identity`, `move`, `recolor`, `transform`, `add`, `delete`) and
edge changes. Anti-unify these rewrites across demos by graph isomorphism and
role predicates, not by raw object index. A candidate program is then

`parse -> role assignment -> graph rewrite -> complete-grid render`.

### Proof obligation

For each surviving role assignment, the renderer must prove all of the
following: every selected role is unique, every target node is generated or
accounted for, every source node is preserved or erased, all edge equations
are satisfied, writes are in bounds and collision-free, and the rendered grid
equals every demonstration target exactly. A program that proves only one
object delta is not a certificate for the task.

### Bounded search

Use the existing top-`k` correspondence beam only as a proposal generator, then
enumerate a small role vocabulary and relation terms. With at most `k`
correspondences, `R` role predicates, `Q` edge/action terms, and fixed caps
`A,B` on clauses and relations, the frontier is bounded by a finite sum of
terms of the form `k * R^A * Q^B` before exact execution; graph-isomorphic
rewrites are quotient representatives. The fixed caps, not an informal linear
bound, are the safety guarantee. This is materially cheaper than enumerating
arbitrary programs while allowing the joint compositions that literal object
deletion missed.

### Four-L4 mapping

The CPU lane enumerates and verifies graph rewrites. GPU coder proposals emit
typed graph programs rather than unconstrained Python. A separate numeric or
image-conditioned lineage proposes alternate scene factorizations. The final
judge sees output classes plus graph traces and first-divergence diagnostics;
it never gets permission to invent an unverified grid. Candidate classes are
deduplicated after rendering, preserving multiple proofs underneath.

### Promotion gate

Implement only the smallest graph grammar that produces nonzero exact output
class recall on the 120-task evaluation fold. Measure correspondence ambiguity,
role uniqueness, renderer completion rate, and output recall separately. If
the grammar again yields zero recall, stop expanding the executor and route
graph proposals to the neural/refinement lineage instead.

## Iteration 155 — Structured execution diffs for localized refinement

### Hypothesis

The refinement loop should receive a causal counterexample, not a prose
judgment that a candidate is wrong. If the verifier exposes the first
divergent intermediate state and the exact changed coordinates, a coder model
can mutate the responsible operation while preserving already-correct demos;
this should retain proposal diversity and reduce repair search.

### Construction

`experiments/trace_repair.py` now provides a bounded `GridDiff` with observed
and expected shapes, changed-cell tuples, exception state, and truncation at
128 cells. `build_repair_prompt` includes this structured diff alongside the
bounded candidate source and demo grids. Five repair tests cover coordinates,
shape bounds, truncation, exception reporting, and prompt safety. The full
suite passes 389 tests.

### Proof obligation

The diff is diagnostic only: it does not authorize a hidden-test read or a
partial candidate promotion. Every repaired program must replay all demos
exactly, and a trace mismatch at state `s_j` may target only operation `o_j`
unless the trace length or input itself is invalid. This preserves the CEGIS
invariant while making refinement more informative.

### Deployment rule

Attach structured diffs to the coder/DSL lineage only after a real fixed-cache
ablation measures repair success, unique-class discovery, and wall-clock cost.
Do not run a global “repair everything toward the modal answer” loop; that
would trade away the independent hypotheses required by pass@2.

## Iteration 156 — Equivariant cross-task analogy retrieval

### Hypothesis

Training demonstrations might act as a reusable transformation library. If an
unlabeled test input matches a labeled training-demo input under a witnessed
D8 geometric transform and a bijective color-role map, applying the same group
action to that demo's output is a principled analogical proposal:

`q = pi(g(x))  =>  candidate(q) = pi(g(y))`.

This is a stricter and more general memory prior than exact task-ID lookup,
and it uses no hidden solution labels.

### Measurement

The library contained 3,232 labeled training demonstrations. On the disjoint
120-task evaluation fold, it emitted 0 records and recalled 0/172 outputs. On
hidden inputs it emitted 37 records across 9/259 positions; hidden correctness
was not inspected. The full D8 plus bijective color matching pass required
about 35 seconds on CPU for the two challenge files.

### Theoretical consequence

Equivariance alone does not establish task-family equivalence. Matching one
input pattern under a group action is insufficient evidence that the source
demo's generator applies to the target task; the latent rule may depend on
object relations, dimensions, or color semantics outside the matched grid.
The zero evaluation support falsifies this memory prior as a standalone lane.

### Deployment rule

Keep the analogy index only as an optional low-priority proposal source when a
future cache shows positive marginal class recall. Never use raw nearest-pattern
frequency as a judge vote, and never use task-ID joins to labeled solutions in
hidden analysis.

## Iteration 157 — Connectivity as a latent scene parser

### Hypothesis

The object-centric lanes have silently assumed 4-connected non-background
components. ARC scenes can instead treat diagonal contact as object continuity,
so connectivity should be a typed latent variable in the parser version space:

`parse_4(grid)` and `parse_8(grid)` are competing scene factorizations.

### Construction and measurement

`experiments/object_deltas.py` now accepts an explicit 4- or 8-connectivity
parameter, and `experiments/object_correspondence.py` plus the graph LGG profile
carry that parameter through correspondence and role extraction. The separate
`experiments/connectivity_ablation.py` reports structural disagreement without
reading hidden solutions. Three regression tests cover diagonal merging,
orthogonal agreement, and invalid connectivity; the full suite passes 399
tests.

On the labeled training challenges, 1,446/4,308 parsed grids changed object
count under 8-connectivity, with 8,943 component merges. On the evaluation
challenges, 192/530 grids changed. More importantly for the existing graph
anti-unification probe, evaluation first-choice LGG tasks rose from 9 to 16,
fully typed LGGs from 7 to 10, and over-cap tasks fell from 50 to 26. This is
not a hidden accuracy measurement and no test solutions were consulted.

### Theoretical consequence

Connectivity is part of the latent representation `h`, not merely an
implementation detail. A principled solver should maintain

`P(rule, h | demos) ∝ exp(-L(rule,h)) · 1[exact_replay(rule,h)]`

and marginalize `h ∈ {4,8}` until the complete-grid renderer disambiguates it.
Choosing 4-connectivity globally creates a structural blind spot; choosing
8-connectivity globally can incorrectly fuse independent diagonal objects.
The correct move is parser competition with an exact demo gate and
position-level output-class deduplication after rendering.

### Deployment rule

Promote 8-connectivity only as a second CPU parser feeding the scene-graph
proposal lane. Do not treat the higher LGG count as a score gain. The next
falsification is a complete joint graph-rewrite renderer on both connectivity
choices, reporting exact evaluation output recall, correspondence ambiguity,
and runtime. If 8-connectivity improves traces but produces no complete-grid
class recall, stop expanding connectivity and route its traces as neural
conditioning evidence instead.

## Iteration 158 — Connectivity-aware complete-grid replay

### Integration test

The 4/8 parser choice is now carried through the guarded-role and frame
executor. `FrameProgram` stores the connectivity used for compilation, so a
program cannot silently parse the test grid under a different scene model.
The exact-demo gate and collision/bounds checks are unchanged.

### Measurement

On training data, both 4- and 8-connectivity compiled 1/1,000 frame programs
and produced no test candidate outputs. On the labeled evaluation fold, both
compiled 0/120 tasks and recalled 0/172 output positions. The graph-LGG trace
gain from Iteration 157 therefore does not transfer through this strict
move/recolor/delete frame renderer.

### Theoretical consequence

Changing the segmentation topology can improve latent trace identifiability
without improving the executable hypothesis class. The bottleneck is now
joint rewrite expressivity—additions, transformations, merges, and generated
geometry—not merely object connectivity. A solver should retain parser
uncertainty as evidence, but must not promote a representation whose complete
renderer has zero held-out support.

### Deployment rule

Keep 4- and 8-connectivity as cheap competing parsers only for a future
scene-graph rewrite lane. Stop expanding the strict frame executor. The next
symbolic moonshot must render complete graph rewrites with target-grounded
add/merge/transform clauses, then be judged on exact evaluation output-class
recall; trace counts alone are insufficient.

## Iteration 159 — Sound shape/palette pruning for token search

### Hypothesis

The decoder needs an explicit, auditable contract for structural pruning when
demonstrations support an invariant output shape or palette. These constraints
are not learned guesses: under a calibrated invariant hypothesis, a violating
completion has zero support. Applying them per augmentation view can reduce
DFS branching without changing model weights, TTT, or candidate ranking.

### Construction

`experiments/decode_constraints.py` defines `GridConstraints`, infers only
shape/palette invariants shared by every labeled training output, validates
complete grids, computes a conservative digit/newline/EOS token bound, and
transforms the constraints through `transpose`, `rot90`, and bijective color
permutation view operations. Unknown shape or palette information remains
unconstrained. The existing notebook already contains
`SIZE_CAP_TOKENS=True` and the fold-validated `symbolic_size.py` paranoid
predictor; this module formalizes and tests the same deployment seam rather
than claiming a second independent speedup. The module has three focused
regression tests and the full suite passes 399 tests.

### Measurement

Exact shared output shape was available for 433/1,000 training tasks,
37/120 evaluation tasks, and 117/240 hidden-input tasks. Exact shared palette
was available for 332, 16, and 76 tasks respectively; both invariants held for
162, 3, and 43 tasks. On hidden tasks with known shape, the safe token bound
averaged 159 tokens versus the generic 932-token 30×30 bound, with the latter
retained whenever shape was unknown. No hidden solution values were read.

### Theoretical consequence

Conditional on a calibrated invariant hypothesis `h` proving
`shape(y)=H×W` and `palette(y)⊆P`, token-search states outside that language
have posterior mass exactly zero under `h`. The decoder may then prune them
without a recall tradeoff *conditional on h*. Demonstrations alone do not
prove that the test output preserves a shared shape or palette, so the
inference helper is a candidate prior until task-disjoint calibration confirms
its recall. The invariant must also be transformed with the same group action
as the input view; failing to permute palette roles or swap dimensions under
transpose/odd rotation would make pruning unsound.

This remains a compute theorem, not an accuracy theorem. It improves the
number of search branches reachable within 12 hours only when the invariant
posterior is reliable; it cannot create a missing semantic rule.

### Deployment rule

Keep the existing validated size-cap seam as the production implementation;
use this helper as a contract test/reference if the notebook is regenerated.
Default to no restriction when an invariant is not exact or its fold-calibrated
recall is weak. Log per-task pruning ratio, candidate recall, and wall-clock at
equal seeds on the labeled evaluation fold. Promote any stricter token mask
only if exact candidate recall is unchanged and saved time is spent on
additional independent views or localized repair; never infer shape/palette
constraints from hidden labels.

## Iteration 160 — Submission-path release-gate audit

### Audit

The 14-cell notebook was parsed without executing a model. The starter launch
cell precedes the submission-construction cell, which precedes the Leg-C merge
cell. The submission cell loads evaluation challenges plus labels only outside
competition rerun mode, loads hidden challenges without solutions during the
rerun, writes `submission.json`, and emits `attempt_1` and `attempt_2` through
the original schema builder. The global deadline retains the 600-second
write/submission buffer. All checks passed.

### Theoretical consequence

The merge is monotone at the pass@2 set level: when a verified Leg-C output
replaces the base attempt, the base attempt is demoted into the second slot.
Thus a verified candidate can add coverage but cannot remove the incumbent
from that output position. The guarantee is conditional on the verifier's
proof boundary; it does not equate demo verification with hidden correctness.

### Deployment rule

Treat cell order, schema completeness, rerun label isolation, and deadline
buffer as release invariants. Any future selector or symbolic lane must feed
the existing normalized submission/merge seam and preserve the incumbent on
promotion. No scored Kaggle submission was made in this audit.

## Iteration 161 — Bounded joint scene-graph rewrite renderer

### Construction

`experiments/scene_graph_rewrite.py` implements the smallest complete-grid
rewrite extension suggested by Iteration 154: exact top-1 object
correspondence, id-free role guards, identity/move/recolor/delete clauses,
target-grounded shape transforms, and additions whose anchor is a fixed offset
from a uniquely selected preserved reference object. Every source role and
reference role must be unique; writes are bounds-checked, collision-free, and
replayed against every demonstration before a program is admitted. The
selected 4/8 connectivity parser is stored in the program. Three synthetic
tests cover anchored addition, grounded transform, and ambiguous-reference
rejection.

### Measurement

At full scale, 4-connectivity compiled 1/1,000 training tasks and 0/120
evaluation tasks. 8-connectivity compiled 2/1,000 training tasks and 0/120
evaluation tasks. Evaluation candidate recall was 0/172 for both. The hidden
input-only pass compiled 1/240 tasks for each parser and emitted 0 candidate
outputs; no hidden solution values were read. The full suite passes 402 tests.

### Theoretical consequence

Joint rendering is strictly stronger than a single object-delta label, but
top-1 one-to-one correspondence is still a hard ceiling. A real merge can
make several source objects correspond to one generated target object, while
the current assignment must either delete a source or attach the target to one
source as a false transform. Thus zero transfer does not falsify scene graphs
as such; it falsifies this correspondence/effect quotient as a standalone
solver.

### Deployment rule

Do not spend GPU time on this strict renderer and do not promote its training
candidate. Keep it as a proof substrate and route future graph proposals to a
many-to-one/many-to-many correspondence lane. The next symbolic test is a
bounded merge correspondence that preserves all source footprints, derives a
single target geometry from the union or a reference role, and still requires
complete-grid exact replay on the evaluation fold.

## Iteration 162 — Many-to-one bridge falsification

### Hypothesis

The strict one-to-one correspondence ceiling might be hiding a common merge
primitive: two singleton source objects become one connected target by painting
the horizontal or vertical segment between their anchors. A bounded renderer
can learn the orientation and bridge color from demo deltas, retain both
source footprints, and select the two endpoints by unique relational guards.

### Measurement

`experiments/merge_bridge.py` implements this exact many-to-one subfamily under
both 4- and 8-connectivity. It rejects non-singleton endpoints, diagonal
placements, nonuniform orientations/colors, occupied interiors, ambiguous role
guards, and any demo mismatch. Three synthetic tests pass. At full scale it
compiled 0/1,000 training tasks, 0/120 evaluation tasks, and 0/240
hidden-input tasks for each connectivity; evaluation recall was therefore
0/172 and hidden candidate support was zero. No hidden solution values were
read.

### Theoretical consequence

Many-to-one correspondence is necessary for some scene rewrites but not
sufficient: a line bridge is only one geometric realization of a merge, and
its exact singleton/axis assumptions are absent from this dataset's strict
support. The result rules out spending search budget on more hand-coded
bridge-color or distance variants without a task-family router.

### Deployment rule

Keep the bridge renderer as a negative control and proof substrate. Do not
promote it or expand it blindly. Richer merge/transform proposals should come
from an independent neural or typed-code lineage, then be checked by a
general many-to-many renderer with object-union, relation, and collision
certificates. The graph lane's value is now chiefly verification and
structured evidence for the holistic selector.

## Iteration 163 — Long-horizon proposal lineages under offline constraints

### Hypothesis

The path to the 70s is more likely to come from allocating the 12-hour
window to independent, executable proposal lineages than from adding another
isolated symbolic primitive. Recent public solver reports converge on this
shape: multiple modality-specific solvers propose complete grids, a program
or sandbox executes them, iterative repair is driven by localized failures,
and a judge compares whole output classes. These reports are evidence about
search structure, not a Kaggle-legal implementation recipe: their strongest
claims rely on external APIs, hosted models, or sandbox services.

### Evidence and translation

The beetree report describes four complementary solvers (multimodal image
prompting, hint extraction, staged object/transform labeling, and deep
search) followed by logic and consistency judges. The Confluence report uses
12 agents, repeated refinement, a 12-hour watchdog, and a top-frequency vote
over all generated grids. Both are consistent with the earlier MDS/ARCANA
finding that verification and refinement matter more than a single monolithic
decoder. The direct offline translation is four GPU lineages, each with a
different representation or prompt contract, plus CPU-side execution,
normalization, proof checks, and class-level judging.

The critical correction is correlation control. A raw global frequency vote
can amplify one shared misconception when agents inherit the same TTT state
or prompt. Therefore each output class should first be normalized within a
lineage, then weighted by lineage reliability and evidence quality. The second
submission attempt should maximize marginal class coverage relative to the
incumbent, not merely be the second-highest raw score.

### Budget model

For a task with independent-lineage success probability (p_i), the idealized
coverage of (m) independent lineages is

\[
1-\prod_{i=1}^{m}(1-p_i).
\]

If a repair pass succeeds conditionally with probability (q_i), its value is
not (q_i) alone: it is (q_i) times the probability that the lineage's
failure is repairable, divided by the additional wall-clock cost. Repair is
worthwhile only after an execution diff localizes a small failure surface;
otherwise a fresh lineage has higher expected unique-class gain. This gives a
testable scheduler: spend each GPU-minute on the largest estimated marginal
unique output-class gain per minute, with an explicit reserve for a fresh
lineage and final packaging.

### Deployment rule

Preserve the current base launch and Leg-C reserve. Add only an offline-safe
proposal contract: each lineage emits a complete candidate, a compact trace,
its exact-demo replay score, its failure diff if rejected, and a semantic
output hash. Permit at most two localized repairs before branching to a fresh
lineage. Keep the CPU specialists as cheap proposals and proof filters, never
as unconditional answers. Before any scored run, perform a frozen-cache
replay comparing raw voting, lineage-normalized voting, and incumbent-relative
pass@2 selection under identical seeds and wall-clock. This is the highest
value experiment still available without consuming Kaggle attempts or
fabricating a neural cache.

## Iteration 164 — Task-local query equivariance as a soft posterior gate

### Hypothesis

The current decoder already creates multiple transformed views of each test
input, but it mostly treats their agreement as raw support. A stronger and
still offline-safe signal is task-local query equivariance: after inverting
each view, a genuine output class should recur across independently sampled
legal nuisance views. This is not a correctness proof—an incorrect rule can
be consistently wrong—but it is evidence against view-specific decoding
artifacts and should reduce the chance that one accidental view wins attempt
2.

### Mathematical form

Let `G_t` be the subset of geometric/color transforms justified by the
task's synthetic demo behavior, and let `y_g` be the decoded output from
view `g`, mapped back to the original frame. For an output class `c`, use

`K_t(c) = sum(g in G_t) w_g * 1[y_g = c]`, with `sum_g w_g = 1`.

after deduplicating repeated samples within each view. The weights should be
calibrated from held-out demo transformations (or remain uniform when no
calibration exists), with a cap on any single view. Combine this with the
candidate's base likelihood and family/lineage evidence; do not replace them
with `K_t`. In particular, a class supported by two independent lineages
and three views should outrank a class supported by ten views of one lineage,
unless held-out calibration says otherwise.

### Falsification and deployment

The gate is falsified if it lowers exact candidate recall on a disjoint
shadow fold, or if it hard-rejects orientation-sensitive tasks whose
transformed views are not legal nuisance instances. Therefore the transform
set is a soft, task-local hypothesis, never a global hard constraint. The
replay should compare raw view count, uniform view consensus, calibrated view
consensus, and the existing log-evidence scorer on the same frozen cache.

The production contract is simple: preserve the decoded view id and inverse
transform in each candidate record, collapse within-view duplicates before
computing `K_t`, and expose the consistency term to the class selector. This
is a selector-only change, so it cannot rescue a missing correct candidate;
the coverage ceiling remains the primary gate. No cache was available here,
so no accuracy claim is made and no neural output is fabricated.

## Iteration 165 — The controller is the learned component

### New evidence

Poetiq's public analysis describes a model-agnostic meta-system that chooses
among direct reasoning, code generation, repeated feedback, and termination;
the reported ARC-AGI-2 result is officially verified for its hosted-model
configuration, but the implementation requires model APIs and is not a
drop-in offline Kaggle path. Its most transferable claim is architectural:
the prompt is only an interface, while an adaptive controller extracts and
assembles partial answers over several calls. This agrees with the ARC Prize
technical report's broader refinement-loop observation and with the
independent ARCANA/ABPR proposals.

### Theoretical translation

Let a solver state be `s = (D, t, C, E, b)`: demonstrations, query input,
current candidate classes, execution evidence, and remaining budget. An
action `a` is one of fresh proposal, targeted repair, alternate
representation, proof check, or stop. The correct policy is not

`argmax_h P(h | D, t)`,

but a finite-horizon value-of-information policy

`pi*(s) = argmax_a E[delta unique pass@2 coverage | s, a] /
                 E[seconds | s, a]`.

The controller should stop when the upper confidence bound on every
remaining action's marginal gain falls below the value of preserving the
final packaging reserve. This formalizes self-auditing without requiring a
second large judge model.

### Offline implementation rule

Use deterministic state transitions around the existing solvers. After each
proposal, record exact-demo replay, first-divergence diagnostics, new output
classes, lineage overlap, and elapsed time. Repair only if the diff is local
and the calibrated repair rate exceeds the best fresh-lineage rate; otherwise
branch. Route alternate representations only when the current lane has zero
or weak support for the visible structural bucket. Stop solved positions
early, but never infer correctness from agreement alone. This makes the
controller auditable, preserves the four-L4 breadth requirement, and turns
the 12-hour run into a data-collection process whose raw cache can be
replayed under better selectors.

No controller ablation was run because the repository has no neural candidate
cache. The claim remains a design hypothesis, not a leaderboard result.

## Iteration 166 — Whole-grid recursive correction versus token-tree search

### Hypothesis

The current NVARC lane searches an autoregressive token tree. That is efficient
when the correct continuation is locally likely, but ARC errors are often
global: one wrong object count, alignment, or output dimension can invalidate
an otherwise plausible prefix. A complementary recursive lane should maintain
an answer state and a latent task state, then repeatedly revise the whole
grid. The 2025 technical report identifies this answer/latent-state recursion
as the defining mechanism of the Tiny Recursive Model (TRM), alongside the
separate evidence for recursive self-refinement in the ARChitects line.

### Mathematical target

For a candidate answer or executable hypothesis `y`, define a demo energy

`E_D(y) = mismatch(y(D), D.outputs) + lambda * invariant_violations(y)`.

A recursive solver maintains `(y_k, z_k)` and applies

`z_(k+1) = F(D, t, y_k, z_k)` and
`y_(k+1) = G(D, t, y_k, z_(k+1))`.

The desired training/fitting property for a task family is a contraction on
the demo energy: `E_D(y_(k+1)) <= rho * E_D(y_k)` for `rho < 1` until an exact
demo solution is reached. Under that condition, after `N` correction steps
the residual is bounded by `rho^N * E_D(y_0)`. This is a real stopping rule,
unlike decoding until a token budget is exhausted. The contraction is a
family hypothesis that must be measured; it is not assumed for all ARC tasks.

### Offline translation

Keep NVARC/DFS as the incumbent and use a small recursive model or recursive
head as an independent proposal family. Give it a bounded number of full-grid
updates, an explicit latent state, and a proof gate that rejects any update
which worsens exact demonstration replay or violates hard shape/palette
constraints. Preserve every intermediate grid as an output class because a
later correction can be wrong even when an earlier state was useful. If the
energy plateaus or oscillates, branch to a fresh representation rather than
continuing the same recursion. This lane is especially attractive on large
grids where token-tree branching consumes the 12-hour budget, but it must
clear the same marginal candidate-coverage and runtime gates as every other
proposal family.

No recursive model was loaded and no ablation was run. The result is a
mathematical architecture hypothesis supported by the report's mechanism,
not a local score claim.

## Iteration 167 — Stochastic latent trajectories for basin escape

### Hypothesis

A deterministic whole-grid recurrence can converge quickly to one attractive
but wrong answer basin. Probabilistic TRM work proposes injecting noise into
latent recursion so multiple trajectories explore distinct basins, while
Recursive Inference Machines provide a related inference-oriented view of
recursive neural reasoning. The ARC-specific opportunity is not to create
more correlated token samples; it is to turn latent noise into a controlled
mode search over exact output classes.

### Mathematical construction

For a fixed task, let the recurrent state update be

`z_(k+1) = F(D, t, y_k, z_k) + sigma_k * epsilon_k`,

`y_(k+1) = G(D, t, y_k, z_(k+1))`, with `epsilon_k` drawn from a zero-mean
distribution. Each trajectory `r` produces a final class `c_r` and a proof
energy on the demonstrations. If the noise is a valid proposal kernel, the
trajectory histogram estimates mode mass,

`P_hat(c | D, t) = sum_r w_r * 1[c_r = c] / sum_r w_r`,

where `w_r` is based on proof energy and calibrated early-stop confidence.
The two-attempt action is the two largest distinct classes under this
histogram, after collapsing trajectories sharing a lineage and a final
class. This prevents a single basin with many near-identical trajectories
from overwhelming a minority basin.

### Offline rule and falsification

Use a small number of noise scales and stop a trajectory when its demo energy
plateaus; reserve at least one deterministic trajectory as an anchor. Reject
noise schedules that increase invalid grids, demo mismatch, or runtime. The
critical replay compares deterministic recursion, independent latent seeds,
and input-view perturbations at equal compute, reporting unique-class gain,
candidate coverage, selector recovery, and effective sample size. A positive
trajectory count with zero new output classes is no gain. If the model has no
knowledge coverage for a task family, basin escape cannot create the missing
rule, so the lane remains subordinate to cross-family proposal diversity.

No recursive checkpoint is present in the current local notebook and no
trajectory experiment was run. This is a falsifiable architectural hypothesis,
not a score claim.

## Iteration 168 — Remove the hidden long-grid prior from DFS

### Observation

The current decoder prunes a partial sequence when cumulative NLL reaches a
fixed `max_score = -log(0.2)`, then recursively applies that same bound to the
whole completion. For a completed output with `L` generated tokens, this
accepts only when its geometric-mean token likelihood exceeds

`exp(-max_score / L)`.

Thus the same threshold demands roughly 98.4% mean token likelihood at
`L=100` and 99.82% at `L=900`. This is not merely a confidence threshold; it
is a strong length prior that can delete globally correct, larger-grid
outputs before output-class selection. The structural size-cap work fixes
maximum length, but it does not remove this cumulative-score bias.

### Moonshot formulation

Replace the fixed probability threshold with a calibrated description score

`S(y) = -log P(y | D, t) + lambda * L(y)`,

where `lambda` is a code-length prior learned on a disjoint source fold. The
DFS frontier can then be budgeted by a fixed number of best partial states or
by an adaptive `S` quantile, while exact shape/palette constraints still
prune impossible continuations. The special case `lambda=0` is pure
length-normalized evidence; positive `lambda` restores a principled MDL prior
instead of an accidental probability cutoff. The output selector remains
class-based and lineage-aware after this frontier expansion.

The safe implementation is an anytime frontier: retain the incumbent best
candidate, open new branches only while their optimistic completion bound can
enter the top-K class budget, and stop when the marginal unique-class gain per
second falls below the reserve threshold. This couples score normalization to
the controller rather than allowing a larger beam to consume the entire
12-hour run.

### Falsification gate

On one frozen decoder cache, compare the current cumulative threshold against
length-normalized and MDL-normalized frontiers at equal wall-clock. Measure
candidate recall, unique output classes, large-grid versus small-grid recall,
selector recovery, and runtime. Promote only if long-grid coverage rises
without reducing the incumbent pass@2 set. No neural cache exists locally, so
this remains a static decoder-theory result; `experiments/decode_score_policy.py`
formalizes the replay seam and its five tests pass. No candidate or hidden
label was fabricated.

## Iteration 169 — Row-boundary diverse frontier search

### Hypothesis

When the output shape is known, the decoder can treat a grid as a sequence of
rows rather than an undifferentiated character stream. The current token DFS
branches at every cell and applies one global cumulative threshold; it can
spend its budget on many near-duplicate prefixes while losing a globally
useful alternative. A row-boundary frontier can preserve diversity at the
same wall-clock cost.

### Construction

For each row boundary `r`, maintain states

`s_r = (prefix, palette_used, object_summary, relation_summary)`

with score `-log P(prefix | D, t)` plus optional calibrated structural
penalties. Within a fixed summary signature, a higher-score state dominates a
lower-score state, so the latter can be discarded without changing the best
completion under any constraint that depends only on that signature. Keep a
bounded number of states per distinct output-class/structural signature, not
just the global top-K token prefixes. At the final boundary, deduplicate exact
grids and pass the class set to the lineage-aware selector.

This is exact only for constraints expressible through the retained summary;
component connectivity, long-range symmetry, and object correspondence need
larger state or a proof pass. The safe version therefore treats summaries as
search heuristics and keeps the incumbent raw beam as a fallback. Known
shape/palette constraints remain hard only when justified by the existing
decode contract.

### Falsification gate

Replay a fixed cache or deterministic decode trace at equal time with (1) the
current cumulative DFS, (2) row-boundary global beam, and (3) row-boundary
diverse beam. Report candidate coverage, unique semantic classes, selector
recovery, output-size strata, and peak memory. Promote only if the diverse
frontier adds useful classes without displacing the incumbent class or
exceeding the 12-hour budget. No model run or hidden label access occurred;
this is a search-design hypothesis.

## Iteration 170 — Soft-masked 2D refinement as a global correction lane

### New evidence

The ARChitects technical report explains why their masked-diffusion line
escaped limitations of their earlier autoregressive solver: a predicted grid
can be fed back into the model, uncertain positions can be remasked, and
recursive refinement can operate on global structure. The report also
describes continuous mixtures of token embeddings and adding a mask direction
to a predicted token, rather than forcing every intermediate state to be a
hard discrete grid. This is a different mechanism from widening the current
Qwen DFS.

### Mathematical design

Assume a known output shape with cell states (u_i) in an embedding space and
a mask vector (m). A soft-remask step is

`u_i' = e(y_i) + alpha_i * m`,

where `alpha_i` is large for high entropy, cross-view disagreement, or a
localized execution-diff cell, and small for stable cells. The denoiser maps
the full 2D state to new cell distributions. Iteration is therefore a
coordinate-wise trust-region method: preserve low-uncertainty cells while
reopening only the uncertain region. A structurally informed schedule can
remask an entire object, row, or relation neighborhood when the error signal
indicates a coupled mistake; cell-independent remasking is insufficient for
ARC transformations that move or resize objects.

### Kaggle-constrained translation

Use a public, license-cleared 2D masked model as a separate proposal lineage,
fit or adapt it only on visible demonstrations, and run a bounded number of
recursive denoising steps. The known shape/palette contract supplies the
output lattice and hard color mask. Preserve the incumbent NVARC candidate,
all intermediate grids, uncertainty maps, and remask masks. A CPU proof gate
checks exact demo replay, dimensions, palette, and collisions; a class-level
selector then compares the diffusion lineage with NVARC, TRM, and program
proposals. Memory must be audited for one model per L4, and model/source
licenses must pass the same release gate as the current notebook.

The most promising hybrid is not to replace NVARC globally: route large-grid,
high-entropy, or diagonal-structure tasks to soft-masked refinement while
keeping token DFS for compact, high-confidence tasks. This is a structural
router hypothesis, not permission to spend the entire 12-hour budget on a
second model without measured marginal coverage.

### Falsification gate

On a disjoint fold, compare autoregressive DFS, hard masked diffusion, and
soft-remask recursion at equal forward-pass and wall-clock budgets. Measure
large-grid recall, coupled-object corrections, unique output classes, exact
demo replay, and memory. Reject if soft remasking merely reproduces the same
class, destabilizes already-correct cells, or has no positive marginal class
coverage. No masked checkpoint was loaded locally and no hidden labels were
used.

## Iteration 171 — Chronological demo truncation is an edge-case risk

### Measurement

The baseline's `cut_to_len` removes the earliest demonstrations when a task
exceeds the 8,192-token context. A tokenizer-compatible approximation that
counts every grid cell and row separator found 1/1,000 training tasks above
8,192 tokens after including train inputs/outputs and test inputs, and 36/1,000
above 4,096. Exact special-token accounting can move the boundary, so the
single-task figure is a lower-bound audit rather than a proof that truncation
never fires. The same task structure is used by the hidden input bundle, but
no hidden labels were inspected.

### Theoretical consequence

Chronological deletion is a poor approximation to the information-selection
problem: the examples jointly expose the latent generator, and an early
example may contain the only instance of a shape, palette change, or relation.
A future context adapter should preserve the mandatory challenge example and
select the remaining demos by marginal structural-feature coverage per token,
with a last-example/chronological fallback for exact baseline compatibility.
This is a budgeted coverage heuristic, not a correctness theorem, because
features can omit the decisive relation.

### Deployment rule

Keep the current path unchanged by default. Add an optional information-aware
selector only after measuring exact-token overflow on the real checkpoint and
replaying held-out examples. Its promotion gate is strict: no demo-inclusion
loss on disjoint fold replay, no candidate-coverage regression, and a positive
time or large-task recall gain. Given the measured rarity of 8k overflow,
context selection is lower priority than widening the candidate class set.

## Iteration 172 — On-policy denoising curriculum for self-generated states

### Problem

The baseline adapts on clean demonstration pairs and then decodes the hidden
query in one autoregressive pass. A recursive refiner trained only on clean
targets sees a distribution shift when its own imperfect output is fed back
as the next state. The denoising-recursion literature identifies precisely
this mismatch: supervising recovery over multiple corrupted intermediate
states creates a tractable curriculum and encourages non-greedy correction.

### Leakage-safe algorithm

For each visible demonstration `(x, y)`, generate a bounded corruption chain
`y_0, y_1, ..., y_K = y` using cell replacement, object-local masking, row or
column erasure, and small geometric perturbations that remain within the
known output shape. Train/adapt a separate LoRA branch to map `(D, x, y_k)` to
`y_(k+1)` or directly to `y`, with loss weights increasing toward exact
recovery. After adaptation, generate a query candidate, feed it through the
same repair interface, and retain every intermediate output class.

The chain must be generated only from labeled demonstrations; the hidden
query is used only as an input to inference. A corruption is admissible only
when its target is known and its shape/palette semantics are explicit. This
avoids treating an arbitrary model hallucination as a training label.

### Stability condition

Let `E_k` be exact demo error after refinement step `k`. A branch is eligible
for another step only if held-out demonstration corruption experiments show
`E_(k+1) <= E_k` in expectation and the step adds nonzero output-class
coverage. If the update improves demo error but collapses all query
trajectories to one class, it is over-regularized; if it increases query
diversity without demo improvement, it is noise. The controller should retain
the deterministic NVARC branch and spend recursion budget only while the
measured conditional gain beats a fresh proposal.

### Deployment and falsification

This is a separate GPU lineage, not a mutation of the incumbent. Compare
clean-only adaptation, one-step corruption repair, and multi-step denoising at
equal adaptation steps and wall-clock. Report exact demonstration replay,
first-divergence repair rate, unique query classes, large-grid recall,
selector recovery, and memory. Reject any schedule that uses query outputs as
labels, loses the incumbent pass@2 class, or creates only correlated copies.
No model or candidate cache is available locally; the result is a design
hypothesis grounded in the denoising-recursion mechanism.

## Iteration 173 — Shift-robust routing with declared uncertainty

### Problem

The hidden-input structural audit found substantial evaluation-to-hidden drift
in area and palette distributions. A point estimate of a lane's held-out hit
rate can therefore route too much of the 12-hour budget to a specialist that
does not transfer. The existing hierarchical calibrator shrinks sparse groups,
but it does not expose a conservative lower bound for routing.

### Bound

For a lane with `s` successes in `n` task-position trials, compute a Wilson
lower confidence bound `L_source(s,n; z)`. If a target distribution is known
to lie within total-variation distance `delta` of the source distribution,
then for the lane-success event `A`,

`P_target(A) >= max(0, L_source - delta)`.

This follows from the defining TV event inequality. It is only a guarantee
when `delta` is a declared valid radius; an observed train/evaluation TV
distance is merely a stress estimate and must not be mislabeled as a proof.
Multiply the resulting target lower rate by the selector-recovery lower rate
and divide by expected lane seconds to obtain a conservative routing score.

### Implementation

`experiments/shift_robust_router.py` implements the Wilson bound, explicit
shift radius, selector-recovery factor, and deterministic lane ranking. Five
tests cover zero evidence, finite-sample conservatism, monotone shift penalty,
cost-aware ranking, and invalid inputs. The router is intentionally separate
from production defaults: it can be fed task-position rates from
`fold_calibration.py` and target-visible structural buckets without accessing
hidden outputs.

### Deployment rule

For each candidate lane, maintain both a point estimate and a lower-bound
score. Route a lane only when its lower unique-class gain per second beats the
incumbent's conservative alternative, reserving an unknown-mass bucket for
unsupported target strata. If no defensible shift radius exists, widen the
interval or use the pooled prior; do not invent confidence from the hidden
input distribution. This deliberately sacrifices some speculative upside to
avoid wasting a GPU tranche on a brittle family.

No model calls or GPU runs were made. The implementation is a calibration and
budgeting tool, not evidence of candidate recall or leaderboard performance.

## Iteration 174 — Partial-identification routing under heterogeneous shift

### Problem

Iteration 173 used one declared total-variation penalty for an entire lane.
That is safe but can erase useful evidence when the hidden mix is visibly
heterogeneous: a lane may be stable on compact two-color objects while having
no support for large multi-component scenes. A single global penalty cannot
express that distinction.

### Conditional lower bound

Partition task positions by a query-visible structural bucket `b` (for
example, area band, palette cardinality, component count, or a coarse scene
signature). Let `l_b` be a leakage-safe lower bound on the lane's success rate
conditional on bucket `b`, and let `pi_b` be the unlabeled target mass of that
bucket. Then the target success rate has the partial-identification lower
bound

`P_target(success) >= sum_b pi_b l_b`.

The proof is direct: condition on the observable bucket, apply the lower bound
inside each bucket, and take the target-mixture expectation. A bucket absent
from source evidence has no identified success probability, so the safe lower
bound is zero. This is less pessimistic than subtracting one global shift
radius only when the conditional-stability assumptions are defensible; it is
not a license to turn train/evaluation frequency differences into a theorem.

### Implementation and deployment rule

`experiments/shift_robust_router.py` now includes
`groupwise_target_lower_rate`. It validates a target-mass distribution,
weights each supported bucket's Wilson-minus-shift lower bound, and assigns
zero lower mass to unseen buckets. Eight CPU tests cover the original router
plus unseen-bucket mass, the single-bucket identity case, and invalid mass
distributions.

Use this score for structural lanes only after defining buckets from query
inputs and estimating their masses without labels. Sparse buckets should be
pooled into a predeclared conservative superclass or left at zero; they must
not inherit a favorable neighboring rate by convenience. The router should
compare conditional lower unique-class gain per second with the incumbent,
reserve an explicit unknown bucket, and spend GPU time on a lane only when the
bound survives a shadow-fold sensitivity sweep over bucket definitions and
shift radii. The incumbent answer and exact verifier remain authoritative.

No model calls, GPU runs, Kaggle submission, or hidden labels were used. This
iteration improves budget allocation theory, not candidate recall evidence.

## Iteration 175 — Distributionally robust two-attempt selection

### Problem

The official objective is set-valued: one exact output is enough, but two
attempts can rescue a position. Selecting the two highest global classes can
still be brittle when the visible task mix contains subfamilies with different
class frequencies. The selector should optimize joint pass@2 coverage under
the same structural-shift uncertainty used by the router.

### Bound and objective

For visible bucket `b`, let `p_b(C)` be the source-fold probability mass of an
output class `C`, let `S` be a one- or two-class attempt set, and let `delta_b`
be a declared total-variation radius. The bucket event bound is

`P_target(output in S | b) >= max(0, sum_(C in S) p_b(C) - delta_b)`.

If the target-visible bucket mixture is `pi_b`, then the full pass@2 lower
bound is

`LB(S) = sum_b pi_b max(0, p_b(S) - delta_b)`.

This is a lower bound on the probability that either submitted attempt is
correct, not a claim about exact class posterior calibration. It preserves the
competition's two-output action while making overlap and complementarity
explicit. The max is outside the subtraction per bucket: a weak bucket cannot
make another bucket's evidence negative, and an unseen target bucket contributes
zero rather than borrowing support from a different family.

### Implementation and falsification

`experiments/robust_pass2_buckets.py` implements the bound and exhaustive
selection over observed classes, with deterministic tie-breaking. Four tests
cover conditional mixture arithmetic, unseen-bucket conservatism,
complementary pair selection, and invalid inputs. This is a CPU-only theorem
and selector contract; it has not been run against a candidate cache.

The deployment gate is strict. Build `p_b` only from disjoint labeled folds,
define buckets from query-visible features, estimate `pi_b` without hidden
labels, and sensitivity-sweep the result over bucket coarsenings and
`delta_b`. Promote only if the robust pair improves held-out pass@2 recovery
or unique output-class rescue without losing the incumbent pair. If a pair's
advantage disappears under a plausible shift radius, keep it as a diversity
candidate but do not let it replace the base answer.

## Iteration 176 — Multiple-comparison correction for pair search

### Problem

Pass@2 selection is itself a search problem. If `K` output classes are
available, the selector can examine `K + choose(K, 2)` one- or two-class
actions and then report the best calibration score. The winning score is
optimistically biased even when every individual estimate is unbiased. This
can make a diversity lane look like a 72% path while it is only the luckiest
pair among many correlated alternatives.

### Finite-class bound

For a fixed class set and an exchangeable calibration sample of `n` output
positions, let `p_hat(S)` be the empirical pass@2 rate of action `S`. A
one-sided Hoeffding union bound over all actions gives, simultaneously,

`p(S) >= p_hat(S) - sqrt(log((K + choose(K,2))/delta) / (2n))`.

The correction grows only logarithmically in the pair search space, but it is
large when `n` is small—the usual ARC regime. The guarantee is conditional on
the candidate set being fixed before calibration and on the exchangeability
assumption; it does not survive repeatedly redefining the class set after
looking at the same labels.

### Implementation and promotion gate

`experiments/selection_bias_guard.py` implements the action count and the
uniform lower bound. Four tests verify singleton-plus-pair counting,
search-size monotonicity, the zero-safe floor, and input validation. The
selector may still use the raw score for ordering, but a candidate pair cannot
replace the incumbent unless its corrected lower bound beats the incumbent's
corrected bound on a disjoint shadow fold.

For production, freeze the candidate generator, bucket definitions, and
correlation-collapse policy before opening the calibration fold. If the search
is adaptive, treat each adaptation round as a new family and pay an additional
alpha-spending or code-length penalty. This is a cheap safeguard against
winner's curse and is more credible than interpreting a single best-of-many
public or held-out score as a generalization proof.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 185 — Representation-closed PoE view calibration

### Audit finding

The official competition currently permits a GPU notebook of at most 12 hours,
with internet disabled, on the L4x4 pool; at most one submission is allowed per
day and at most two final submissions may be selected. These constraints make
view generation and pass@2 selection part of the same mathematical budget,
not independent post-processing. The rules also permit freely/publicly
available external data and pretrained models, subject to the open-source
requirements for a winning solution.

The strongest transferable warning from the recent multi-perspective ARC-AGI-2
study is that 40–80 geometric/color views are not automatically independent
experts. Their PoE failed when the model had been trained mainly on row-major
serialization, and task-local TTT could overfit badly. A valid ARC symmetry can
therefore be semantically correct but statistically miscalibrated for the
checkpoint.

### Mathematical correction

Let `g` index a reversible view and let `nll_(g,i)` be the teacher-forced NLL
of visible demonstration `i` in that view. Estimate a representation offset

`b_g = mean_i [nll_(g,i) - median_h nll_(h,i)]`.

The median reference makes the estimator invariant to a task-wide difficulty
shift and avoids selecting one view as an absolute oracle. Shrink the offset
toward zero when the task has too few demonstrations:

`b_g(lambda) = (1-lambda) b_g`, `0 <= lambda <= 1`.

For a candidate output with view scores `nll_g(y)`, use the calibrated PoE

`S(y) = - sum_g w_g [nll_g(y)-b_g] / sum_g w_g`,

where the sum is over decoded views. If a candidate lacks a view, do not invent
its likelihood; charge an explicit coverage penalty
`eta * (1 - sum_present w_g / sum_all w_g)`. This separates two effects that
raw PoE conflates: a view being systematically hard for the checkpoint and a
candidate being unsupported by the view family.

The correction is not an assertion that the views are independent. It is a
calibration layer before any correlation-aware aggregation. Independent
lineages still require provenance normalization; view calibration only removes
an additive representation bias.

### Executable seam

`experiments/view_calibration.py` implements offset estimation, calibrated
weighted geometric-mean scoring, explicit missing-view penalties, and stable
ranking. Four CPU tests cover bias removal, shrinkage, coverage penalties, and
deterministic ties. The test inventory now has 442 tests. In this shell the
bundled Python lacks `pytest` and the isolated runner cannot access its managed
Python directory; the new module self-check, direct assertions, and bytecode
compilation pass, while the previous 438-test run remains the last full-suite
execution. No production decoder default was changed.

### Promotion and falsification

On one frozen visible-task cache, compute offsets using leave-one-demo-out
teacher-forced NLLs where possible, then compare raw mean-NLL, raw PoE,
calibrated PoE, and calibrated PoE plus coverage penalty. Stratify by view
family and grid area. Reject the method if it merely increases demo replay
while reducing unique query output classes, or if the selected offset is
unstable under one-demo deletion. The production gate is positive held-out
query pass@2 gain per second with no exact-coverage regression.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 177 — Refine the latent program state, not only the output grid

### Hypothesis

Autoregressive grid decoding commits too early to a surface form. A better
moonshot is to let the model maintain a distribution over typed transformation
states—object correspondence, operation family, guards, parameters, and only
then rendered cells. This is compatible with the recurring refinement pattern
in recent ARC systems, but changes the latent being refined from “next token”
to “next program constraint.” The grid renderer becomes a deterministic
terminal, not the sole reasoning medium.

Represent a state as

`z = (skeleton, correspondence, parameters, guards, residual-grid-mask)`.

For visible demonstrations `D`, score it with

`E(z;D) = exact_mismatch(z,D) + lambda * MDL(z) + mu * invariant_violations(z,D) + nu * unresolved_cells(z)`.

The critical distinction is that `exact_mismatch` is a hard verifier at the
promotion boundary. The other terms guide search among partial states but may
never turn a demo failure into a label.

### Refinement operator and proof boundary

Alternate three operators for a bounded number of steps:

1. a neural proposal expands or edits one typed AST frontier;
2. an exact constraint projector removes parameter/correspondence values that
   fail any labeled demo or proven shape/palette invariant;
3. a deterministic renderer executes surviving complete states on the query.

If `V_k` is the finite version space after step `k`, exact projection gives the
monotonic invariant `V_(k+1) subseteq V_k`. If every surviving state renders the
same query output, the output is certified relative to the enumerated DSL. If
two output classes survive, the selector must retain both or pass their proof
cards to the existing robust pass@2 selector. Neural energy, trajectory
agreement, and partial-cell accuracy are search signals only; none is a
correctness certificate.

### Why this could move the score

The current verified symbolic union is tiny because complete object programs
are sparse, while the neural branch often has the right visual hypothesis but
loses it during long-grid decoding or ranking. Program-state refinement can
reuse a partial correspondence or operation prefix across multiple output
renderings, expose first-divergence repairs, and delay expensive cell-level
decoding until shape, palette, and object count are fixed. It also creates
genuinely different candidates from NVARC's token beam, rather than flooding
the same output basin with temperature copies.

### Kaggle deployment design

Use the existing one-hour Leg-C slot with the local coder to propose bounded
typed AST fragments, not arbitrary executable Python. Run the AST whitelist,
demo replay, and resource limits after every expansion; persist only proof
cards and rendered outputs. A high-value implementation is a two-stage beam:
`8` skeletons × `8` parameter/correspondence completions, capped by a static
node budget. Feed only verified outputs into attempt selection. The NVARC
incumbent remains available for every unverified position, and the global
12-hour guard reserves the final serialization buffer.

The first shadow-fold experiment should compare whole-program Leg-C against
fragment refinement at equal model calls and CPU seconds, measuring complete
verified candidate coverage, first-divergence repair rate, unique query output
classes, pass@2 recovery, and large-grid timeout rate. Reject the idea if
partial states merely produce more syntactic programs without increasing
verified output-class coverage. No production default changes are authorized
by the theory alone.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 178 — Diversity-preserving program refinement

### Evidence update

The newly checked MDS technical report is unusually relevant to the current
design: its reported gain comes from independent text/image/code search plus
context-preserving holistic judging, while its negative results say that
prescriptive prompting and iterative refinement can systematically reduce
hypothesis diversity. This is a warning against turning Iteration 177 into one
shared chain of increasingly similar program edits.

### Revised architecture

Maintain `B` isolated refinement branches. Each branch owns its initial
modality, model seed, typed-program state, and local repair history. Branches
may share only:

- exact hard constraints extracted from labeled demonstrations;
- proven shape/palette/resource invariants;
- canonical output hashes and correlation metadata at the merge boundary.

They must not share an unverified natural-language hypothesis, preferred
object correspondence, or partial AST. A later refinement step either stays
inside its branch or creates a fresh branch from the original task. This
preserves the possibility that a minority interpretation is correct.

### Diversity objective

Let `q_b(C)` be branch `b`'s mass on output class `C`, and let `w_b` be its
lineage weight. The merge objective is not raw sample count. Track

`U(S) = sum_C 1[C in S] * (1 - product_b (1 - q_b(C)))`

for the unique class coverage of a proposed action set `S`, together with
`H(q)` or effective class count `exp(H(q))` as a collapse diagnostic. A
refinement is admitted only when it raises verified candidate coverage or
conditional selector recovery, or when it buys a demonstrable increase in
unique class mass per second. Higher agreement among branches is evidence
only after correlation and independent generation are established.

### Kaggle consequence

Leg-C should use several short isolated proposal lineages rather than one
long self-repair conversation. The code verifier can broadcast a failed
constraint as a fact, but each branch must independently choose whether and how
to repair. NVARC/TRM/typed-program branches should remain separate until exact
rendering. The final holistic analogue is the existing robust pair selector,
which sees all proof cards and output classes together without allowing one
lineage to flood the vote.

Falsification is straightforward on a frozen shadow fold: compare shared-loop
refinement, isolated-branch refinement, and independent one-shot proposals at
equal model calls and wall-clock. Record unique output classes, verified
coverage, selector recovery, and correlation-adjusted effective sample size.
Reject isolated branching if its diversity rises but verified coverage and
pass@2 recovery do not. This evidence changes the deployment design but does
not authorize a production default change.

The external result is method evidence, not a prize-eligible Kaggle recipe:
its reported system uses hosted frontier models and API-scale search. No model
calls, GPU runs, Kaggle submission, or hidden labels were used locally.

## Iteration 179 — Provenance must survive the notebook merge boundary

### Audit finding

The current notebook still has a material gap between the research selector
theory and the shipped path. `ArcDecoder.load_decoded_results` stores only
`beam_score`, augmented NLLs, and the rendered solution. `score_kgmon` then
uses `len(guesses)` as positive evidence after grouping equal grids. The
production decoder therefore cannot distinguish:

- eight D8/color views of one TTT lineage;
- several beams from one decoding view;
- genuinely independent model seeds or program families.

The first two cases are correlated copies, not independent votes. A class that
occupies one model basin can win by flooding even when a minority class has
stronger independent support. This is exactly the failure mode that the
diversity-preserving refinement and lineage-ESS iterations were intended to
prevent.

### Safe seam

Persist a bounded provenance record with each decoded sample:
`source_file`, base test position, geometric/color view, TTT seed or branch,
decoder lineage, and candidate index. At semantic-class aggregation, use one
normalized contribution per `(independent_lineage, view-family)` rather than
raw count. Within a lineage, retain the best likelihood/MDL witness for a
class; across lineages, average or weight only by a predeclared family prior.
The group-normalized class mass then feeds the existing robust pair selector.

Do not immediately replace the known-good attempt-1 ranking. Add a shadow
`score_lineage_kgmon` beside `score_kgmon`, benchmark both on the local
evaluation cache, and promote only if lineage normalization improves
conditional selector recovery or minority-class rescue without reducing exact
candidate coverage. If provenance is missing, the selector must fail closed to
the baseline or mark the samples as one correlated unknown group; it must not
infer independence from filenames or output multiplicity.

### Expected value under the 12-hour constraint

This is a high-leverage, low-compute change: it alters no model weights and
adds only metadata plus CPU aggregation. It can expose output classes already
generated by the four-GPU run, whereas additional sampling may only create
more copies of the current mode. The decisive shadow metrics are output-class
coverage, lineage-adjusted effective sample size, selector recovery, and
pass@2 score. A raw score increase without minority recovery is not evidence.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 180 — Shadow decoder aggregation with explicit provenance

### Implementation

The provenance seam is now executable in
`experiments/decoder_provenance.py`. `DecodedCandidate` carries an output
class, weight, lineage, and view family. The aggregator keeps the maximum
class witness within each `(lineage, view_family)` group, normalizes one class
distribution per group, averages those group distributions, and returns the
top two classes. Missing lineage or view metadata is mapped to one explicit
`__unknown__` group, so absent provenance cannot manufacture independent
evidence.

Four tests cover same-lineage vote flooding, the conservative unknown-group
fallback, best-witness deduplication, and invalid candidates. The full suite
now passes 430 tests. This is a selector-only shadow component; the Kaggle
notebook's baseline `score_kgmon` and attempt-1 behavior are unchanged.

### Mathematical interpretation

For provenance groups `g`, let `p_g(C)` be the normalized within-group mass
of class `C`. The shadow score is

`p(C) = (1 / |G|) sum_g p_g(C)`.

This is a conservative exchangeability model: repeated beams inside one group
cannot increase its total mass, while genuinely independent groups receive
separate evidence. It is not automatically optimal—if groups have different
reliability, replace the uniform weight only with a fold-calibrated prior.
The exact output hash remains the merge key; provenance is evidence metadata,
never a correctness proof.

### Promotion protocol

Before wiring this into the notebook, add provenance at sample creation in the
worker and preserve it through `load_decoded_results`. Run a frozen local
evaluation replay with three selectors: current `score_kgmon`, the shadow
group-normalized selector, and the existing diverse-attempt fallback. Require
no loss in exact candidate coverage, positive conditional selector recovery or
minority-class rescue, and stability across seed/view ablations. A cache record
without trustworthy provenance must remain baseline-ranked or be treated as a
single unknown lineage. No public score or hidden label is a substitute for
this replay.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 181 — Log-evidence is an intermediate, not a lineage proof

### Artifact reconciliation

The repository contains two notebook lines. The current baseline notebook
uses `score_kgmon` by default; the separate Nemotron probe builder already
contains an opt-in `score_log_evidence` selector. The latter is a meaningful
improvement over raw support counting: it combines beam and augmentation
likelihoods with stable log-mean-exp operations, and its tests show that a
twenty-copy weak class no longer defeats one strong candidate solely by count.

It still cannot prove independence. `score_sum` groups by exact rendered grid
but passes only a list of candidate dictionaries to the getter. Provenance is
absent, so eight transformed views, same-view beams, and independent lineages
are all exchangeable. Log-mean-exp prevents linear flooding, but it does not
assign separate prior mass to genuinely independent sources or discount
correlated ones differently.

### Correct ablation matrix

The next frozen-cache replay should compare, on identical decoded records:

1. `score_kgmon` — known control, support-count plus NLL;
2. `score_log_evidence` — probability-space nuisance marginal, no provenance;
3. provenance-normalized log evidence — within-group best witness, one
   normalized distribution per `(lineage, view-family)`, then group averaging;
4. the existing diverse-attempt policy applied after each ranking.

Compare exact candidate coverage separately from selector recovery and
minority rescue. The third arm should win only if it recovers correct classes
that the first two miss without losing correct candidates that the baseline
already emits. A score difference on only one output class is not enough: the
official action is two attempts per output position.

### Runtime implication

The Nemotron builder currently defaults to one Leg-C lineage and disables
lineage-aware mode, so provenance normalization alone cannot create
independent evidence for that leg. The high-value configuration is to spend a
small fixed Leg-C budget on multiple isolated prompt/seed lineages, persist
their IDs, and let the selector give each lineage bounded mass. This must be
balanced against the base queue refund: more lineages are useful only when
their verified output-class gain per second exceeds the tasks they displace.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 182 — Optimize lineage count jointly with sample count

### Missing decision variable

The existing lineage planner balances a fixed sample total across a fixed
number of groups. The runnable Nemotron path exposes both `--lineages` and
per-lineage sampling, but its budget gate does not select the lineage count
from measured setup cost. With a fixed one-hour Leg-C window, spending all
samples in one correlated batch can be statistically wasteful; opening too
many branches can instead lose coverage to model/request overhead.

### Calculus

Let `B` be the Leg-C budget, `s` the setup cost per lineage, `c` the cost per
sample, and `L` the number of lineages. The feasible sample count is

`N(L) = floor((B - Ls) / c)`, with `N(L) >= L`.

For balanced counts `n_i` and exchangeable within-lineage correlation `rho`,
use the planning ESS

`ESS(L) = N(L)^2 / sum_i [n_i + rho n_i(n_i - 1)]`.

When `rho=0`, setup cost favors fewer lineages because ESS is just total
sample count. As `rho` rises, balanced independent groups become more valuable
and the optimum moves toward larger `L`, until setup cost or the one-sample
per-lineage floor dominates. This is a scheduling theorem for evidence
diversity, not a correctness guarantee; `rho` must be estimated from duplicate
output behavior on a shadow fold.

### Implementation

`experiments/lineage_budget.py` enumerates feasible lineage counts, constructs
balanced allocations, and chooses the maximum-ESS plan with deterministic ties.
Four tests cover correlation-sensitive lineage choice, balanced allocation,
zero-setup behavior, and invalid budgets. The full suite now passes 434 tests.

### Deployment rule

Before the Leg-C run, estimate setup/sample seconds from a short local probe
and estimate a conservative range for `rho`. Choose the plan that maximizes
the *lower* ESS across that range, not the plan that wins at one optimistic
point. Cap lineages so every group still receives enough samples to emit a
verified program. Then feed the explicit lineage IDs into the provenance-aware
selector; do not use ESS as a vote or promote a branch because it is merely
independent.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 183 — Balance TTT by demonstration, not output-token count

### Audit finding

The completion-only collator masks user text and leaves assistant grid spans
as labels. With the default cross-entropy reduction, a training episode is
effectively weighted by its total number of supervised output tokens. A large
demonstration therefore supplies more gradient than a small demonstration
even when both are equally informative examples of the task rule. The
augmentation loop repeats this size bias 128 times.

### Objective correction

For an episode with `m` supervised demonstration outputs, let `T_i` be the
number of labeled tokens in output `i` (including its structural terminator),
and let `ell_(i,t)` be token cross-entropy. The current token-balanced loss is

`L_token = [sum_i sum_t ell_(i,t)] / [sum_i T_i]`.

The rule-learning objective is better approximated by the demo-balanced loss

`L_demo = (1/m) sum_i [sum_t ell_(i,t) / T_i]`.

Equivalently, each token in demo `i` receives weight `1/(m T_i)`. This makes
loss invariant to grid area when the number of demonstrations is fixed and
prevents a 30×30 output from overwhelming a 3×3 output solely by length.
The weighting must be computed before any random geometry/color augmentation;
augmentation copies of the same episode inherit the same per-demo weights.

### Implementation plan

Extend the collator to emit a bounded `loss_weights` vector by identifying
assistant spans and their EOS/row boundaries. Override `compute_loss` with
unreduced token cross-entropy, multiply only supervised positions by those
weights, and normalize by the weight sum. If the Unsloth trainer cannot safely
accept unreduced loss tensors, use an episode-level sampling weight as a
fallback and preserve the exact baseline path behind a flag. Do not infer
weights from raw character counts after tokenization.

Use an interpolation
`L_lambda = lambda L_demo + (1-lambda) L_token` as the shadow family,
because larger outputs may legitimately carry more relational evidence. Tune
`lambda` only on disjoint training folds and select by demo-replay held-out
recovery, candidate-class coverage, and query pass@2—not training loss.

### Falsification and expected value

Replay the same task episodes, augmentations, learning rate, and optimizer
steps under token-balanced, demo-balanced, and interpolated loss. Measure
exact held-out demonstration recovery by size stratum, first-divergence repair,
unique query output classes, selector recovery, and wall-clock/memory. Reject
the correction if it improves small-grid replay but loses large-grid query
coverage or collapses output diversity. This is a cheap, high-leverage TTT
ablation because it changes gradient weighting rather than model architecture;
it remains entirely label-safe, using only visible demonstration outputs.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 184 — Executable demo/token loss interpolation

### Implementation

`experiments/demo_loss_weights.py` now encodes the TTT weighting proposal.
Given supervised answer-span lengths `T_i`, every token in demonstration `i`
receives

`w_(i,t) = lambda/(m T_i) + (1-lambda)/(sum_j T_j)`.

The weights sum to one. At `lambda=0` the loss is token-balanced; at
`lambda=1` every demonstration span has equal total mass. Four tests verify
equal span mass, token-proportional mass, interpolation normalization, and
input rejection. The full suite now passes 438 tests.

### Integration boundary

The helper is deliberately detached from the GPU trainer. A future collator
must identify exact supervised spans, including their EOS/row terminators,
and pass the weight tensor to unreduced cross-entropy. It must not approximate
span length from characters or assume that augmentation copies are
independent demonstrations. If trainer internals make weighted loss unsafe,
the fallback is a deterministic episode sampler with the same effective
weights; do not silently retain token imbalance while claiming the correction.

### Promotion test

On a fixed fold, compare `lambda` in `{0, 0.25, 0.5, 0.75, 1}` at identical
optimizer steps, augmentations, and wall-clock. Stratify held-out recovery by
output area and number of demonstrations. The correction is only useful if it
improves query candidate coverage or pass@2 recovery after controlling for
large-grid recall; lower training loss alone is not a promotion signal.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 186 — Median calibration proof and order-preservation gate

### Identifiability model

Assume the visible-demo NLL for view `g` and demo `i` decomposes as

`nll_(g,i) = q_i + b_g + epsilon_(g,i)`,

where `q_i` is task/demo difficulty, `b_g` is a view-specific representation
bias, and the residual is bounded by `|epsilon_(g,i)| <= epsilon` for the
honest views. The absolute values of `q_i` and `b_g` are not separately
identifiable: adding a constant to all `q_i` and subtracting it from all
`b_g` changes nothing. The median-reference estimator fixes this gauge by
recovering offsets relative to `median_g b_g`.

If strictly more than half of the views are honest on every demo, the per-demo
median remains inside the honest residual envelope. For an honest view, the
estimated offset error is therefore bounded by `2 epsilon` (one residual from
the view and one from the median reference). Averaging demos cannot worsen
that deterministic bound. This is the reason for using a median reference,
not a mean reference: a minority of pathological serializations cannot move
the center arbitrarily.

### Ranking theorem

For candidates with the same complete view support and normalized weights, the
calibrated score error is at most `2 epsilon` per candidate. Thus a candidate
pair whose raw calibrated score margin exceeds `4 epsilon` keeps its order
under every admissible honest-view residual. If view support differs, that
guarantee disappears; the explicit coverage penalty becomes part of the
decision and must be sensitivity-tested over a declared interval.

This yields a release gate: estimate a bootstrap/leave-one-demo stability
interval for `epsilon`, then only allow calibration to replace the incumbent
when the selected margin exceeds the corresponding worst-case bound. Small
margin cases should retain both output classes for pass@2 rather than forcing a
point ranking. Large offsets with unstable signs are evidence of a broken
representation family, not permission to amplify it.

### Falsification

Construct synthetic additive-bias caches with controlled honest-view fraction,
residual radius, and candidate margins. Verify the `4 epsilon` threshold and
measure failure once the honest fraction falls to one half or the residuals are
non-additive. On real visible tasks, compare offset signs under demo deletion
and view-family deletion. Reject the calibration lane if it fails its bound,
has unstable signs, or changes only the first output while reducing minority
class retention.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 187 — Enforce group-equivariant TTT, not only view augmentation

### Commutator failure

Let `G` be the finite group generated by D8 geometry and bijective color
relabeling, `T_g` its action on a complete task, and `S` the serializer. The
desired property is not merely to train on many transformed examples; it is

`decode(f_theta(S(T_g D))) = T_g decode(f_theta(S(D)))`.

The left and right sides can disagree because the serializer, causal prefixes,
and checkpoint are not equivariant. The discrepancy is a commutator error.
When it is large, a product-of-experts multiplies incompatible distributions
and can suppress the correct candidate. This explains why raw view count is a
poor proxy for independent evidence.

### Architecture proposal

Keep the existing answer loss on transformed visible demonstrations, but add a
masked orbit-consistency term over canonicalized cell logits:

`L_TTT = L_answer + kappa * (1/|G|) sum_g JS(P_theta^g, mean_h P_theta^h)`.

Here `P_theta^g` is the distribution over abstract color values at each cell
after inverse-mapping view `g`; row/column delimiters and shape tokens are
excluded from this term. Use a small `kappa` at first and ramp it only after
the answer loss has reached an exact-demo plateau. The canonicalized cell
distributions are then coupled across views while the answer loss remains the
hard semantic constraint.

A cheaper approximation is pairwise stop-gradient distillation from the
canonical view to transformed views, with a held-out transformed demo used as
the consistency check. Do not force consistency across views whose geometry or
palette transform changed a task invariant; the transform must act on the
whole input/output pair and preserve shape legality.

### Quotient-space argument

If the orbit term reaches zero for every `g` and the canonicalizer is exact,
the predictor descends to the quotient of task representations by `G`: all
members of an augmentation orbit produce the same canonical output. Thus the
view ensemble ceases to be a vote over representation artifacts. In the
finite-error case, the expected disagreement of the ensemble is bounded by
the average orbit loss; a candidate whose calibrated score margin exceeds the
combined orbit and NLL uncertainty can be promoted, while low-margin cases
retain both pass@2 classes.

This is a stronger condition than repeating augmented examples. It also gives
a diagnostic: measure canonicalized cell disagreement before and after TTT;
if answer replay improves while orbit disagreement rises, TTT is memorizing
serialization quirks and must be rejected.

### Falsification and budget

On a fixed visible fold, compare answer-only TTT, orbit-consistent TTT, and
frozen weights at equal optimizer steps and equal 12-hour schedule allocation.
Report exact transformed-demo replay, canonicalized cell KL/JS, unique query
output classes, and pass@2 recovery. Route the regularizer only to tasks where
view disagreement is high and enough cell logits fit in memory. The lane is
promotable only if it adds query-class coverage per second without reducing
the incumbent’s exact coverage; low disagreement tasks should bypass it.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 188 — Reynolds projection onto the equivariant predictor space

### Theorem

Let `G` be a finite group acting on task representations and on output-cell
probability vectors. For a predictor `p`, define its canonicalized Reynolds
projection

`P_G p(x) = (1/|G|) sum_g U_g^(-1) p(T_g x)`,

where `U_g` maps output cells/colors back to canonical coordinates. If the
task distribution is `G`-invariant and the true conditional output `p*` is
equivariant, then `P_G p` is equivariant and is an orthogonal projection in
the squared-error geometry. Jensen gives

`E ||P_G p - p*||^2 <= E ||p - p*||^2`.

For probabilistic cell predictions, cross-entropy has the same direction of
improvement because `-log` is convex: the likelihood of the symmetrized
prediction is no worse than the average likelihood of the aligned view
predictions, under the same invariant distribution. This is a variance
reduction theorem, not a claim that every individual task benefits from every
candidate view.

### Executable reference

`experiments/reynolds_projection.py` provides a dependency-free permutation
reference for inverse alignment, Reynolds averaging, and maximum orbit
residual. Four CPU tests cover exact equivariance, non-equivariant averaging,
idempotence, and invalid alignments. The test inventory is now 446
definitions; the previous 438-test full run remains the last complete suite in
this shell.

### ARC consequence

The projection should operate on aligned soft cell/color logits or
probabilities, after inverting the view transform. It should not average
already-decoded grids: a cellwise average can produce a non-integer palette,
break shape constraints, or erase a minority discrete hypothesis. Decode the
projected soft field into one candidate, retain the best exact-verified
unprojected candidate as a second candidate, and let the calibrated/provenance
selector decide when the projection is low-margin.

This gives a clean separation:

1. Reynolds projection reduces representation variance in the continuous
   predictor;
2. exact rendering and invariants enforce the discrete ARC grammar;
3. pass@2 preserves a valid minority mode when projection is ambiguous.

The theorem fails if the chosen augmentation is not a symmetry of the whole
input/output task, if the task distribution is strongly non-invariant, or if
the output alignment is wrong. Those are measurable failure modes, so the
projection must be gated by shape legality, palette bijection, and observed
canonicalized disagreement.

### Promotion experiment

On a frozen cache, compare raw per-view logits, calibrated PoE, orbit-loss TTT,
and Reynolds-projected logits at equal decode budget. Measure soft cell
cross-entropy on visible demos, exact demo replay after rendering, canonical
orbit disagreement, unique query classes, and official pass@2 recovery. A
projection lane is promoted only if its uncertainty-adjusted query-class gain
is positive and its exact-coverage lower bound is no worse than the incumbent.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 189 — Monte Carlo orbit projection budget

### Sampling bound

The full color-permutation orbit is too large to enumerate. If each sampled
view contributes an aligned coordinate in `[0,1]`, the Monte Carlo Reynolds
estimate has the Hoeffding-union bound

`P(max_u |P_hat(u)-P_G(u)| > epsilon) <= 2 M exp(-2 m epsilon^2)`,

where `M` is the number of cell/color coordinates and `m` is the number of
independent sampled views. Therefore a sufficient view budget is

`m >= ceil(log(2 M / delta) / (2 epsilon^2))`.

This is a planning bound, not an accuracy claim: it assumes uniform samples
from the valid symmetry orbit, bounded coordinates, and independent samples.
Inverse-paired views can reduce variance but should be counted as one bounded
pair when independence is not established. A sampled subset that is not
uniform over the orbit estimates a different projector.

### Scheduling consequence

Choose `epsilon` from the candidate score margin and `delta` from the global
release risk, then calculate `m` per task from its output area. Small grids
need few views; large grids should not blindly receive the same 128-way
augmentation. Spend the saved budget on an independent program/representation
lineage or a proof check, because more correlated orbit samples cannot expand
the hypothesis class. Conversely, if the bound exceeds the remaining task
budget, skip projection and preserve the incumbent rather than use a noisy
“equivariant” average.

`experiments/orbit_sample_budget.py` implements the bound and minimum-sample
calculation with four CPU tests. The test inventory is now 450 definitions;
the module self-check and bytecode compilation pass. No production decoder
default was changed.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 190 — Behavioral quotienting and submodular probe design

### Why output hashes are insufficient

Two programs can produce the same query grid while encoding different latent
rules, and two samples from one rule can differ only because the decoder
failed. Counting exact query-output classes therefore confounds hypothesis
coverage with surface noise. For a finite demo-verified program set `H`, use a
label-free probe battery `P` made from visible inputs and legal metamorphic
transformations. Define

`h ~_P h'` iff `h(p) = h'(p)` for every `p in P`.

The quotient `H / ~_P` is a behavioral coverage certificate relative to `P`.
It is stronger than beam-count diversity: members in different classes are
provably different on at least one executed counterfactual, while members in
one class can be safely collapsed for probe-driven search (but not necessarily
for final output ranking if their provenance differs).

### Probe selection theorem

For each probe `p`, let `E_p` be the set of program pairs it separates. The
coverage objective

`F(S) = |union_(p in S) E_p|`

is a monotone submodular set function: adding a probe cannot un-separate a pair,
and the marginal gain decreases as `S` grows. Greedy selection of `k` probes
therefore achieves the standard `1 - 1/e` approximation to the best fixed-size
probe battery. This gives a principled CPU budget: spend probes where they
split the most unresolved program pairs, instead of running every probe on
every candidate or assuming every output sample is independent.

The probe battery must remain label-free. D8 transforms and valid color
relabelings are safe when applied to the complete input/output task; translations,
object deletions, or duplications are only soft stress tests because their true
outputs are unknown. A probe separates hypotheses, but does not certify which
side is correct. Exact visible-demo verification remains the proof boundary.

### Selector consequence

Allocate one representative from each high-mass behavioral class to the
independent solver/repair budget. If two classes render the same query output,
they need only one pass@2 slot but should retain separate provenance for
calibration. If one class contains many raw beams from one lineage, collapse
them before estimating coverage. Feed behavioral-class count and unresolved
pair mass into the lane router; a high unresolved-pair mass is a reason for a
new representation or counterexample search, not another sample from the same
lineage.

`experiments/behavioral_partition.py` implements exact probe signatures,
separated-pair coverage, behavioral partitions, and greedy probe selection.
Four CPU tests were added; the test inventory is now 454 definitions. The
module self-check and bytecode compilation pass. No production selector was
changed.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 191 — Four-output score-delta budget

### Current target audit

The current public leaderboard snapshot shows 72.08 for first place, 70.42
for second, and 40.83 for third; the former planning note that treated 34.44
as the top-eight cutoff is therefore historical, not the present moonshot
target. The official board warns that the public leaderboard is only an
approximately 50% representative sample and that final placement uses the
private half. Treat 70.42 as the demonstrated competitive floor and 72.08 as
the target band, not as a guarantee about the final private ordering.

### Exact decomposition

For `N` output positions, let `c` be the fraction whose correct output is
present in the candidate set and `r` the fraction of those covered positions
whose correct class survives selection. The expected output score is

`s = c r`.

For a baseline `(c0,r0)` and candidate `(c1,r1)`, the exact change is

`Delta s = (c1-c0)r0 + c0(r1-r0) + (c1-c0)(r1-r0)`.

The three terms are coverage, conditional selector recovery, and their
interaction. With roughly 259 output positions in the current rerun shape,
the 70.42-to-72.08 gap is only about four to five additional exact outputs.
This makes selector preservation and candidate recall equally concrete: a
method that adds one candidate class but displaces a covered class can have a
negative delta even when its raw diversity rises.

### Operational gate

Every future lane must report `(coverage, selector_recovery)` on a disjoint
shadow fold and convert the result to this decomposition. If the lane changes
only `r`, solve the inverse threshold for the required recovery at fixed `c`;
if it changes only `c`, do the symmetric calculation. Do not promote a lane
because its task-weighted accuracy or demo replay improves while its output
delta is below the four-output target. Preserve the incumbent pair whenever
the confidence interval for `Delta s` crosses zero.

`experiments/score_delta_budget.py` implements exact score calculation,
component decomposition, and fixed-coverage inversion with four CPU tests.
The test inventory is now 458 definitions. The module self-check and bytecode
compilation pass. No production selector was changed.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 192 — Two whole-submission finals are a separate hedge

### Protocol distinction

The competition overview specifies exactly two predictions for every test
output in each `submission.json`, with an OR between those two attempts. The
rules separately say that a participant may select up to two Final Submissions
for judging. The latter is a whole-file portfolio budget; it must not be
silently converted into four per-output guesses. Unless the final judging
protocol explicitly combines files position-wise, every final file must be
treated as independently scored and schema-valid.

### Portfolio objective

Let `s_(k,f)` be the score of whole submission `k` on labeled fold/bootstrap
scenario `f`. If the platform selects the better whole-file final, the
appropriate validation utility for a pair is

`U(A,B) = mean_f max(s_(A,f), s_(B,f))`.

This differs from the invalid per-output union
`mean_j 1[y_Aj = y*_j or y_Bj = y*_j]`, which assumes a scoring rule not stated
for separate final files. Select the pair with the largest `U`, then use the
minimum scenario-best score as a robustness tie-breaker. Scenarios must be
disjoint folds or paired bootstrap replicates; public leaderboard score alone
cannot estimate private complementarity because the board is only a
representative sample.

The likely final pair is a conservative known-good notebook and one genuinely
complementary exploratory notebook (for example, provenance-aware NVARC plus
an independent verified-program/recursive lane). Two cosmetic selector changes
from the same cache are one correlated submission and should not consume the
second final slot. If Kaggle ultimately selects only one final file, this
portfolio analysis becomes a release-risk diagnostic rather than a score gain.

`experiments/final_portfolio.py` implements the conservative whole-file
scenario utility and deterministic pair selection with four CPU tests. The
test inventory is now 462 definitions. The module self-check and bytecode
compilation pass. No production submission path was changed.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 193 — Nemotron license status is an unresolved release gate

### Local artifact audit

The local `data/models/nemotron-lightning/` mirror identifies itself as
NVIDIA Nemotron 3.5 Lightning 30B A3B NVFP4 and carries `OpenMDW-1.1`. Its
metadata also marks the mirror private. The probe README already restricts
this path to unscored public-evaluation experimentation and asks for a private
Kaggle model attachment.

### Eligibility finding

The competition rules require a prize submission to have an open-source
system, model, and weights/parameters as defined by the OSI Open Source AI
checklist, and require the winner to grant the specified CC-BY-4.0 license.
The OSI's current approved-license list does not visibly include OpenMDW-1.1;
the OSI license-review record shows OpenMDW-1.1 was submitted for approval and
explicitly discusses unresolved concerns about its defensive termination
scope. This is not a legal determination, but it is sufficient evidence that
the repository cannot mark the Nemotron lane prize-eligible today.

### Release policy

Keep Nemotron as a potentially valuable accuracy probe, isolated from the
final notebook and not treated as the only route to the 70s. For a final
submission, prefer a checkpoint with a clearly documented OSI-approved or
otherwise rule-compatible license and publicly reproducible weights. Before
reconsidering Nemotron, obtain a current approval/status record, verify that
the exact quantized weights and attached runtime may be redistributed under
the winner obligation, and preserve all upstream notices. If any of those
checks remain uncertain, the model is excluded from the prize path even if it
raises the local score.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 194 — Sound prefix grammar for DFS constraint injection

### Lossless-pruning theorem

Let `L(H,W,P)` be the language of exactly `H x W` grids whose cells belong to
palette `P`, with forced row separators and a final EOS. If the true hidden
output is known to satisfy the demo-inferred shape `(H,W)` and palette subset
`P`, then the true token sequence lies in `L`. A DFS that masks every token
outside the current prefix automaton is therefore recall-preserving under the
constraint's soundness assumption. This is stronger than generating a grid
and rejecting it after the fact: invalid branches are never expanded.

The baseline's unconstrained 16-token vocabulary allows cell, newline, and EOS
choices at every step and uses a conservative maximum of roughly 932 output
tokens for a 30x30 reply. The exact grammar fixes the sequence length to
`H*W + H` tokens (cells, `H-1` separators, EOS), and the cell branching factor
to `|P|`. Its leaf count is `|P|^(H*W)` rather than a variable-length mixture
of delimiter paths. The cell search is still exponential, but the structural
part of the search is eliminated and the model's probability cutoff can be
spent on legal content candidates.

### Safety conditions

Shape equality and palette equality across all visible outputs are high-
precision gates in the repository, but they are not universal ARC laws. If
shape or palette is uncertain, leave that field unconstrained; a false hard
constraint destroys candidate coverage. Under a color-permutation view, map
the palette through the same permutation before constructing the grammar.
Under transpose/rotation, swap dimensions as appropriate. Never infer a view's
constraint from an unparsed key suffix.

### Executable seam and promotion

`experiments/grid_prefix_grammar.py` provides an exact state machine with
forced delimiters, palette masking, completion counts, and deterministic
token budgets; four CPU tests cover legal traversal, branching counts, forced
structure, and invalid inputs. The test inventory is now 466 definitions.
The module self-check and bytecode compilation pass. The production DFS is
unchanged pending a replay on a frozen cache.

Promote only if constrained DFS has identical correct-candidate recall on a
shadow fold, lower decode time, and more unique query output classes per
second. Measure the constraint gate itself: any false-negative shape/palette
case blocks the production default even if the average runtime improves.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 195 — Rerun-file prevalence of sound structural constraints

### Input-only audit

Using only `arc-agi_test_challenges.json` and the labeled demonstration outputs
inside each task, `infer_task_constraints` fires an identical output shape on
117 of 240 rerun tasks, an identical output palette on 76, and both constraints
on 43. No test outputs or hidden solutions were read.

For the 117 shape-constrained tasks, the exact prefix grammar's token count has
mean 158.95, median 112, and maximum 932. The current decoder's conservative
unconstrained maximum is 932. This is not a claim of a 83% speedup—actual
runtime depends on branching, batching, and model overhead—but it identifies a
large and measurable subset where delimiter/shape search can be removed.

### Allocation implication

The safe order is: infer constraints from visible demos, transform them through
the parsed view operations, build the prefix automaton, then mask the model's
ARC vocabulary before expanding DFS. Merely lowering `max_new_tokens` without
forcing row boundaries does not obtain the grammar's guarantee. When the
constraint gate is unavailable, preserve the current search.

If a replay shows near-identical correct-candidate recall, the reclaimed time
should be spent on independent candidate classes or a proof check, not on more
copies of the same view. For a task with output area `A`, palette size `p`, and
known shape, the legal content leaf count is `p^A`; this can still be enormous,
so grammar pruning is a structural accelerator, not a complete solver.

### Promotion gate

Run the grammar on a disjoint labeled fold and check false-negative rate first,
then measure decode seconds, number of valid candidates, unique output classes,
and correct-candidate recall. A single false-negative task blocks enabling the
hard palette/shape field globally; fields can be enabled independently. The
input-only prevalence result supports prioritizing implementation, but does
not justify changing production defaults by itself.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 196 — Naive fixed-demo shape grammar is a negative result

### Public-evaluation falsification

The Iteration 195 prevalence count was only a firing count. On the labeled
public evaluation split, compare each ground-truth test output with
`infer_task_constraints(task)`, which copies a single shape/palette from the
visible demonstration outputs. Among the 56 output positions where fixed shape
fired, 19 violate the inferred shape. Among 22 where fixed palette fired, 1
violates it; among 5 where both fired, 2 violate at least one. Example failures
include 221dfab4, 38007db0, 4a21e3da, 7ed72f31, 8b9c3697, and a25697e4 for
shape, plus 53fb4810 for palette.

This is not a model failure; it is an invalid invariant. Demo outputs can share
a shape while the test transformation changes shape. A hard prefix grammar
built from `infer_task_constraints` would remove the correct answer on roughly
one-third of its shape-fired public-evaluation positions. That source is
blocked for hard pruning.

### Stricter predictor audit

The separate paranoid input-to-output size predictor is materially safer. Its
combined size rule fired on 882/1076 training outputs with precision 1.000 and
109/172 public-evaluation outputs with precision 1.000. The combined palette
rule fired on 824/1076 training outputs with precision 0.9951 and 108/172
evaluation outputs with precision 1.000. Its conservative palette superset
bound fired on every evaluation output with precision 1.000. The result is
consistent with a rule-learning gate that abstains on disagreement, not with
copying a demo statistic.

### Salvage and promotion gate

`grid_prefix_grammar.py` remains a valid exact automaton once its constraints
are sound, but it must not call `infer_task_constraints` as a production hard
gate. The paranoid predictor can propose a hard shape only after a disjoint
labeled-fold audit with zero false negatives; palette should default to the
conservative superset or remain unconstrained unless exact precision is
demonstrated. If a predictor is useful but not zero-false-negative, retain the
unconstrained branch and use the prediction only as a soft EOS/row-length
prior. Measure fired coverage, false-negative rate, decode time, valid class
count, and correct-candidate recall before promotion.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 197 — Preserve recall with a mixture of constraint branches

### Hypothesis

An uncertain shape/palette prediction should influence search without becoming
an invalid support restriction. The target is a decoder whose candidate support
contains the incumbent support, while still harvesting the runtime reduction
from high-confidence structural hypotheses.

### Construction

Let `H0` be the existing unconstrained decoder and let `H1...Hm` be constrained
grammars inferred by the paranoid predictor or other independently calibrated
rules. Decode each branch with prior `pi_k`, retaining at least one beam for
`H0`. The returned candidate support is

`S = union_k S(Hk)`,

so `S(H0) subseteq S`; adding a branch cannot remove an incumbent candidate
unless the scheduler silently steals all of `H0`'s budget. Integer branch
budgets use largest-remainder allocation and an explicit baseline minimum.

For a duplicate grid `y` generated by several branches, rank with the Bayesian
model average

`log P(y|x,D) = logsumexp_k [log(pi_k) + log P(y|x,D,Hk)]`,

over witnesses that produced `y`. This is strictly safer than taking the best
witness: max aggregation treats correlated explanations as independent
evidence and can over-rank a single accidental agreement. A soft length,
row-boundary, or palette bonus may be applied inside `Hk`, but it must not be a
global hard mask unless the disjoint-fold zero-false-negative gate passes.

### Why this matters for ARC-AGI-2

The public audit showed the copied-demo shape rule had a 19/56 false-negative
rate when it fired. A hard grammar therefore has negative expected value even
if it makes decoding much faster on those cases. The mixture policy converts
the same signal into an optional accelerator: the 109/172 evaluation cases
covered by the paranoid combined size rule can receive constrained budget,
while the baseline branch protects the 63 uncovered cases and any future
shape-changing transformation. Palette can use its all-color superset bound
without sacrificing support.

### Executable artifact and gate

`experiments/soft_constraint_policy.py` implements normalization, hard-gate
eligibility, largest-remainder allocation, stable log-sum-exp, and duplicate
output mixture scoring. CPU self-check passes; production decoder is unchanged.
The next replay should compare baseline-only versus mixture branches on a
frozen candidate cache, holding total model calls and wall-clock constant.
Promotion requires non-decreasing correct-candidate recall, positive unique
output-class gain per second, and no loss on large or shape-changing buckets.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 198 — Leave-one-demo stability is a useful confidence tier

### Audit

For every public-evaluation test output, refit the paranoid size predictor after
deleting each one of that task's demonstrations. Of the 109 outputs with a full
task prediction, 81 kept exactly the same prediction under every deletion, and
all 81 were correct. The remaining 28 full predictions were also correct, but
at least one deleted demonstration changed the prediction or caused abstention.
On the training split the corresponding counts were 743 stable/correct out of
882 fired.

### Interpretation

Leave-one-demo stability is not a proof of correctness: the same wrong rule can
be stable if every demonstration supports the same accidental extrapolation.
It is nevertheless a cheap, task-local estimate of how much the prediction
depends on one example. It separates the zero-disagreement high-confidence
tier from a fragile tier without consulting a query label.

### Decoder policy

Use the stable tier for the constrained branch's highest prior and decode
fragile predictions only as mixture branches alongside the unconstrained
incumbent. If a production budget cannot afford both, fragile predictions must
abstain from hard pruning. Calibrate the prior mapping from deletion agreement
on training tasks, and audit by shape-changing versus shape-preserving buckets;
do not mistake the 81/109 public result for a future guarantee.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 199 — Rule-family allowlisting recovers a safe shape gate

### Falsification by rule family

The paranoid predictor's combined precision hides unsafe individual rules. On
public evaluation, the rule-level results were: `same_as_input` 117/117,
`transpose` 2/2, `largest_obj_4` 1/1, and `largest_obj_8` 1/1. The excluded
families were unsafe: `constant` 37/56, `ratio` 2/4, and `affine_offset` 3/5.

Restricting the shape candidate set to
`{same_as_input, transpose, largest_obj_4, largest_obj_8,
smallest_obj_4, smallest_obj_8}` produced 119 fired evaluation outputs, all
119 correct, and 727/727 correct fired training outputs. Every fired evaluation
case had a singleton size; only two training cases had a two-size conflict.
This is 119/172 = 69.2% evaluation coverage with no observed false negative,
versus the unsafe demo-copy rule's 56 fired positions with 19 errors.

### Claim and limit

The actionable result is not “the shape gate is proven for hidden test.” It is a
proof-oriented allowlist: transformation classes with a direct input relation
or an explicitly selected object relation are retained; constant, ratio, and
affine extrapolations are not allowed to hard-mask. The zero-error result is
only an empirical release gate on the available labeled folds.

For a decoder, apply the prefix grammar only when every surviving allowlisted
rule agrees, or decode the union of its size languages when they conflict. The
unconstrained branch remains mandatory for any fragile/unsupported case. This
gives a concrete high-value runtime seam with substantially better expected
recall than a monolithic “any rule that fits demos” constraint. The allowlist
and agreement contract are encoded in `experiments/safe_shape_family.py` and
replayed against the predictor output with 727/727 training and 119/119
evaluation containment.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 200 — Exact palette masking fails the cross-fold gate

### Rule-level audit

The public-evaluation palette result is deceptively clean. Under the paranoid
palette battery, `pal_same_as_input` was 89/89 on evaluation but only 487/489
on training; `pal_add_remove` was 27/28 on evaluation and 387/389 on training;
`pal_constant` was 21/22 and 359/363. The conservative `input_palette` bound
was 153/153 contained on evaluation but had a training violation. Therefore no
exact palette rule currently passes the cross-fold zero-false-negative gate.

### Decision

Do not mask the model's color vocabulary with an exact inferred palette in the
production decoder. The safest palette constraint is the full color universe;
the `input_palette` or `input_plus_added` bound can be used as a soft prior,
candidate-ranking feature, or an additional mixture branch only after a
disjoint calibration audit. This sacrifices some token pruning but avoids a
single missing color invalidating the entire grid candidate.

The current hard-grammar candidate is thus narrower: shape may use the
allowlisted direct/object rules when they agree, while palette remains
unconstrained. This is an intentionally asymmetric policy because shape and
palette have different empirical error profiles; applying one confidence rule
to both fields is unjustified.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 201 — Prefix grammar token semantics are compatible, conditionally

### Interface audit

The notebook does not use arbitrary token IDs for grid decoding. Its embedded
`ARC_VOCAB` maps digits `0..9` to IDs `0..9`, newline to `10`, and
`<|im_end|>` to `15`; `ARC_TOKENS` is the exact list used by `turbo_dfs`. The
formatter serializes an `h x w` grid as `h*w` cell tokens, `h-1` row separators,
and one end marker, giving `h*w+h` tokens. The grammar's
`exact_token_count = h*w+h` and its accepting transitions match this
serialization on representative 1x1, 2x3, and 3x2 grids.

### Boundary

This clears a mechanical compatibility risk, not a correctness risk. The
grammar can exactly forbid malformed row lengths and out-of-vocabulary colors
for this checkpoint, but it still must receive a sound predicted shape and it
must be transformed through every view operation before decoding. A tokenizer
or checkpoint change invalidates this precondition and requires a fresh
vocabulary audit; no generic “digit token” assumption is portable.

### Decision

Keep the grammar as a notebook-compatible runtime primitive. The production
promotion order remains: safe shape allowlist first; palette unmasked; fragile
predictions on mixture/unconstrained branches; exact same-token serialization
and transformed-view tests before enabling any DFS mask. The corresponding
`kaggle_notebook/BASELINE_ANALYSIS.md` injection seam was updated to encode this
asymmetric release contract, preventing the old broad palette/shape-mask plan
from being copied into the notebook.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 202 — Existing notebook cap is safe-but-conservative

### Static seam audit

The notebook already enables `SIZE_CAP_TOKENS` and calls the embedded
`predict_size_paranoid` before each task's augmented DFS batches. That embedded
predictor is not the naive demo-copy helper: it fits input-to-output rules,
requires all surviving rules to agree, and abstains on conflict. The measured
combined result is 882/882 correct fired training outputs and 109/109 correct
fired public-evaluation outputs.

### Opportunity

The unanimous battery loses coverage when an unsafe rule conflicts with a safe
direct/object rule. The allowlisted family from Iteration 199 fires 727 versus
882 total training outputs and 119 versus 109 evaluation outputs, with zero
observed false negatives on both splits. Thus a shadow replacement could
filter out unsafe rule families before agreement, recovering ten evaluation
shape caps without weakening the observed containment gate.

This is still not hidden-test proof. The safe allowlist must be embedded in the
Kaggle copy, replayed with exact view-key behavior, and compared against the
current cap at equal seeds. Until that replay exists, leave the scored notebook
unchanged: the current unanimous predictor is conservative and empirically
sound on the available labeled folds. The updated baseline analysis marks the
allowlist as a shadow replacement rather than a silently applied change.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 203 — More safe caps do not imply more useful runtime

### Cap-budget comparison

Using the notebook's cap proxy `h*w+h+2` against the 932-token generic cap,
compare the current unanimous paranoid predictor with the allowlisted shadow
family. On public evaluation, the broad predictor fired on 109 outputs and
saved 47,833 token slots in aggregate; the allowlist fired on 119 and saved
48,447, an incremental 614 slots. On training, broad saved 629,557 slots over
882 fires, while the allowlist saved 485,532 over 725 fires.

### Interpretation

The ten additional allowlist fires are not uniformly cheap: they are larger
outputs, so coverage gain overstates runtime gain. Conversely, aggregate token
slots are only a proxy—the DFS cost also depends on branching, batch padding,
KV-cache behavior, and the per-task timeout. The result rules out a simplistic
“maximize constraint coverage” objective.

### Promotion gate

The allowlist remains attractive for recall-preserving correctness, but its
runtime value must be measured at the batch/task level with exact tokenizer
lengths and wall-clock. A shadow replay should report (1) correct-candidate
recall, (2) unique output classes, (3) total and p90 decode seconds, and (4)
unfinished-task count under identical seeds. Keep the current unanimous cap if
the allowlist fails to improve completed-task coverage per second; use the
allowlist only where it adds a safe cap without increasing the batch maximum.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 204 — Monotone primary-or-fallback shape caps dominate both alone

### Overlap audit

Compare the current broad unanimous predictor (`primary`) with the allowlisted
shape family (`fallback`) on each labeled output. On training, both fired on
706 outputs with no disagreement; broad-only fired on 176 and allowlist-only on
19. On evaluation, both fired on 105 with no disagreement; broad-only fired on
4 and allowlist-only on 14.

### Policy

Return the broad prediction whenever it exists; consult the allowlist only when
the broad predictor abstains. This is a monotone fallback: it never replaces a
current cap, and it adds only candidates from a separately zero-error observed
family. In the labeled audit it would fire on 901/1076 training outputs and
123/172 evaluation outputs, with no observed shape misses. On the evaluation
cap proxy it preserves the broad 47,833 saved token slots and adds 3,070 from
allowlist-only cases, for 50,903 total—better than broad-only (47,833) or
allowlist-only (48,447) without requiring a model call.

The guarantee is conditional: the two predictors must never disagree, and the
fallback family must retain its cross-fold containment. If they disagree in a
future fold, fail closed to the primary/unconstrained branch rather than
choosing one by confidence rhetoric. Exact batch padding and wall-clock replay
remain mandatory because token slots are only a cost proxy.

`experiments/shape_cap_fallback.py` encodes this primary-preserving contract
and the tokenizer-compatible cap formula. The scored notebook remains
unchanged; the next implementation experiment should be a shadow replay with
the fallback inserted only at the existing `size_caps` seam.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 205 — Runtime savings must be evaluated at the transformed-batch level

### Exact proxy

The notebook decodes four transformed views in a batch and sets one
`batch_max_new_tokens` to the maximum cap among them. Since rotations and
transposes preserve area and may swap dimensions, a predicted original size
`(h,w)` has batch cap `h*w + max(h,w) + 2`, not simply `h*w+h+2`.

Recomputing the comparison with this view-aware proxy gives public-evaluation
counts of 109 broad fires and 123 primary-or-fallback fires. Broad-only saves
47,662 token slots against the 932-token generic cap; the monotone fallback
saves 50,728, an incremental 3,066 slots. The sign and conclusion survive
batch padding, although the magnitude is still only a proxy for wall-clock.

### Decision

Any future cap ablation must report task/batch-level caps after the complete
view schedule, not average original-grid caps. The fallback remains a plausible
shadow optimization because it never replaces a primary prediction, but its
promotion gate must use exact tokenizer lengths, batch padding, candidate
recall, and completed-task count under identical seeds.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 206 — Shape-only grammar separates syntax safety from palette risk

### Theorem

For a validated output shape `(h,w)`, define the language

`L(h,w) = { row_1 newline ... row_h EOS : row_i in {0,...,9}^w }`.

Every valid ARC grid of that shape has exactly one serialization in `L(h,w)`
under this notebook's `grids15` tokenizer. Therefore a decoder that masks
non-cell tokens at cell positions, forces newline after each completed row, and
forces EOS after the final cell preserves the complete support of all valid
shape-`(h,w)` answers. It does not need an exact palette assumption: allowing
all ten colors gives `|L(h,w)| = 10^(h*w)` while still eliminating malformed
row/EOS continuations.

The theorem is conditional only on (1) the shape being correct, (2) the token
serialization staying fixed, and (3) the view transform being applied to the
shape. It is stronger than a mere `max_new_tokens` cap, which leaves illegal
row boundaries in the search and can spend budget on prefixes that cannot
become a valid target grid.

### Fail-closed architecture

Use the primary unanimous predictor; if it abstains, use the allowlisted
fallback; if both abstain, run the original unconstrained decoder. Feed only a
selected shape into the grammar and always use the full color set. On any
predictor disagreement, tokenizer mismatch, unsupported view operation, or
grammar state error, disable the grammar for that batch rather than guessing.
This localizes correctness risk to the measured shape contract and keeps the
palette failure from Iteration 200 out of the hard path.

### Promotion gate

The shadow replay must demonstrate exact support containment on a labeled fold,
identical candidate recall for the incumbent, lower p90 batch time, and fewer
unfinished tasks. Compare cap-only, shape-only grammar, and grammar plus any
soft palette feature at equal seeds and total wall-clock; the latter cannot be
credited for syntax gains that belong to shape-only masking.

A CPU property check enumerated all `10^4` two-by-two full-color grids and the
grammar accepted all 10,000 canonical serializations. This validates the
automaton implementation for that finite case; it does not substitute for the
end-to-end decoder replay.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 207 — Runtime is valuable only through marginal output-class gain

### Budget theorem

Let `N` be scored test outputs, `T` the remaining wall-clock budget, and
`g_l` the expected number of *new correct output classes per second* produced by
lane `l` after incumbent candidates are accounted for. If a constraint change
reclaims `Delta T` seconds and routes it to lane `l`, its expected score gain is

`Delta score = g_l * Delta T / N`.

The cap audit supplies token-slot differences but not `Delta T` or `g_l`. The
primary-or-fallback view-aware proxy adds 3,066 evaluation token slots over the
broad cap, while the current leaderboard gap from roughly 70.42 to 72.08 is
about four to five of 259 output positions. Without a measured conversion from
slots to completed independent proposals, claiming that the grammar closes the
gap is unjustified.

### Allocation consequence

Promote shape grammar only if its wall-clock savings are converted into
verified, non-correlated output classes—e.g. a fresh representation lineage,
first-divergence repair, or proof-check lane. Spending the reclaimed time on
more copies of the same view has near-zero `g_l` once effective sample size
saturates. If no downstream lane can demonstrate positive unique-class gain per
second, retain the existing cap and avoid integration risk.

This connects the runtime work to the competition objective: the measurable
unit is not tokens removed, but incumbent-relative pass@2 recovery per second.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 208 — Union frontier relaxes long-grid evidence without deleting incumbents

### Observation

The current DFS admits a branch only while cumulative NLL is below
`b = -log(0.2) = 1.609`. For a target serialization of length `L`, this is an
average-NLL requirement of `b/L`; at `L=900` the mean must be below about
`0.00179` per token. This couples a fixed sequence-probability threshold to
grid area and can erase a structurally correct long answer.

### Construction

For a shape-known target length `L = h*w+h`, define a calibrated per-token
threshold `tau` and accept a complete path when

`NLL < b OR NLL/L < tau`.

Equivalently, the absolute frontier budget is
`max(b, tau*L)`. Every path accepted by the current decoder remains accepted;
the normalized branch adds only long-grid candidates whose average evidence is
acceptable. For example, `tau=0.02` gives a union budget of `18.0` at
`L=900`, versus `1.609` currently. A partial path can be admitted only when
its NLL is below this union budget, because future NLL is non-negative; grammar
validity and the exact EOS position remain separate obligations.

### Calibration and risk

`tau` must be estimated from teacher-forced demonstration/query-like spans on
a disjoint fold, stratified by output area and model/view family. It is not
valid to choose `tau` because it produces more candidates. The normalized
branch should run as an additional frontier with a fixed class/beam budget,
while the absolute branch and incumbent candidate are retained. If compute is
fixed, reserve baseline beams first and allocate only the remainder to the
relaxed branch.

`experiments/length_frontier.py` implements the union budget and optimistic
partial-path contract; its CPU self-check passes. This is a high-upside
coverage hypothesis because it targets the decoder's length prior directly,
but it remains untested without a frozen neural candidate cache. Promotion
requires long-grid candidate recall gain, no incumbent pass@2 loss, and
positive unique-class gain per second.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 209 — Long-grid coverage is common enough to justify a frontier replay

### Target population

Using labeled output geometry only to identify the bucket, the serialized
length `L=h*w+h` exceeds 200 on 379/1076 training outputs and 122/172
evaluation outputs. It exceeds 400 on 160/1076 training and 85/172 evaluation
outputs, and exceeds 800 on 40/1076 training and 28/172 evaluation outputs.

The current broad predictor supplies a correct observed shape cap for 141/160
training and 63/85 evaluation outputs in the `L>400` bucket, and for 35/40
training and 22/28 evaluation outputs in `L>800`. The primary-or-fallback
policy raises the corresponding counts to 147/160 and 77/85, then 39/40 and
27/28. Thus the length-normalized frontier can be targeted at a substantial
population rather than a pathological edge case.

### Boundary

These counts prove only that the target length is available for many labeled
cases; they do not prove the model's correct path is absent under the absolute
cutoff. That requires a frozen neural cache with beam-level NLLs. The proper
first replay is therefore stratified: compare candidate recall for `L<=200`,
`200<L<=400`, `400<L<=800`, and `L>800`, keeping the current absolute frontier
as an always-on control.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 210 — Condition logits on the grammar, but preserve shape evidence

### Derivation

At grammar state `s`, let `A_s` be the legal token set and let the model's
full-vocabulary logits define `p(t|s)`. The conditional grammar distribution is

`p_G(t|s) = p(t|s) / Z_s`, where `Z_s = sum_{u in A_s} p(u|s)`.

For a legal target token, the conditional NLL is

`-log p_G(t|s) = -log p(t|s) + log Z_s`.

Since `Z_s <= 1`, conditioning removes exactly
`-log Z_s` of invalid-vocabulary NLL. At forced newline or EOS states
`|A_s|=1`, the conditional NLL is zero; at cell states it renormalizes over
the ten valid colors. `experiments/grammar_score.py` implements this identity
and its CPU self-check passes.

### Critical ranking caveat

Grammar conditioning is safe for content ranking *inside one validated shape*,
but it deliberately discards the model's probability of entering that grammar.
If different shape hypotheses are compared after separate conditioning, the
shape with many forced syntax tokens can be over-ranked because those tokens
have been normalized to probability one. The joint score must therefore retain
a shape prior/evidence term, for example

`log P(H|x,D) + log P_G(y|x,H,D)`,

or compare conditioned branches only within a fixed shape and keep the
unconstrained full-vocabulary candidate as the cross-shape control. The
primary-or-fallback policy supplies that control; a grammar branch must never
replace it solely because conditional NLL became smaller.

### Promotion gate

Run three frozen-cache variants: current full-vocabulary scoring, conditional
grammar scoring within fixed shape, and conditional scoring plus calibrated
shape prior. Measure long-grid recall, cross-shape ranking, unique classes, and
pass@2 recovery. Promote only the variant that improves incumbent-relative
coverage without turning grammar normalization into an artificial shape prior.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 211 — Re-anchor the deployment target to the current official state

### Rules refresh

The official rules page was re-read on 2026-09-02. The competition permits one
submission per day and up to two final submissions; prize eligibility requires
the winning system, model, and weights to satisfy the stated open-source
requirements and the winner license is CC-BY 4.0. The official leaderboard
states that the visible board uses approximately half of the test data and
that final standings use the other half.

The current public snapshot is 72.08 in first, 70.42 in second, then 40.83,
37.22, 34.86, 34.86, 34.44, and 34.44. Therefore “top eight in the 70s” is a
forward-looking moonshot target, not the current observed top-eight cutoff. The
research must optimize private-half robustness and release eligibility rather
than tune to a public rank threshold.

### Algorithmic consequence

The two-final portfolio from Iteration 192 remains justified, but only as two
independently valid whole submissions with genuinely complementary candidate
sets. Every proposed decoder change must be evaluated by incumbent-relative
pass@2 recovery on disjoint local folds; public-board imitation is not evidence
of private improvement. The license gate from Iteration 193 remains active for
the Nemotron path.

Sources: [official rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules),
[official leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard),
and [official data description](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data).

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 212 — Actual notebook artifact is a 14-cell hybrid, not the historical 9-cell baseline

### Static inventory

Parsing the current `kaggle_notebook/notebook.ipynb` gives 14 cells, 12 code
cells, and 6 `%%writefile` cells. The active flags are
`SIZE_CAP_TOKENS=True`, `CHEAP_FIRST_ORDER=True`, `LEGC_ENABLED=True`, and
`DIVERSE_ATTEMPT_2=True`. The notebook contains the embedded unanimous size
predictor and two-attempt submission path, but it does not yet contain the
shape grammar, conditional grammar scoring, provenance selector, or
length-normalized frontier from the latest shadow work.

### Consequence

The active scored artifact already includes two prior interventions—cheap-first
task ordering and a conservative size cap—that must be treated as controls in
future ablations. New theories must be inserted at their actual seams and
compared against this hybrid, not against the historical 9-cell description.
The shadow priority order is now: (1) frozen-cache candidate recall, (2)
length/grammar frontier, (3) provenance-aware selection, and only then (4)
additional TTT or program-induction compute. No upgrade is promoted merely
because it beats the original public 33.89 baseline.

The baseline analysis header and queue description were corrected to reflect
the 14-cell artifact and `CHEAP_FIRST_ORDER=True`; the scored notebook itself
was not changed.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 213 — Quantify the long-grid probability cliff before relaxing DFS

### Exact thresholds

With the current absolute budget `b=-log(0.2)=1.609438`, a length-`L`
sequence must have geometric-mean token probability at least
`exp(-b/L)`. The thresholds are 0.984034 at `L=100`, 0.991985 at `L=200`,
0.995984 at `L=400`, 0.997990 at `L=800`, and 0.998213 at `L=900`.

A path whose mean NLL is only 0.01005 (mean token probability 0.99) therefore
passes the absolute rule only through length 160. It fails solely because of
length at the long-grid sizes common in Iteration 209. A calibrated normalized
threshold `tau=0.02` instead accepts mean token probabilities above 0.980199,
so it can recover such a path at any known target length.

### Search guardrail

Relaxing the threshold can admit exponentially many cell sequences. The
normalized branch must not become an unbounded beam: use the exact shape
grammar, retain only the top `K` partial states per row/length bucket, dedup
complete grids before scoring, and reserve a fixed baseline beam for the
absolute frontier. The union policy is a support guarantee, not a claim that
all newly admitted paths are useful.

### Promotion experiment

On a frozen cache, stratify by target length and sweep `tau` from the empirical
90th/95th percentile of held-out mean NLL rather than selecting it by query
score. For each `tau`, record the number of newly admitted correct classes,
newly admitted total classes, p90 decode time, and incumbent pass@2 retention.
The useful regime is where the first quantity grows faster than cost; if the
normalized branch mostly adds duplicate or low-quality grids, stop at the
current absolute frontier.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 214 — Calibrate the normalized frontier with a finite-sample conformal gate

### Threshold construction

Let `a_i` be one non-negative mean-NLL nonconformity score from a held-out
task/output, computed under the same tokenizer, view family, and TTT protocol
as the query decoder. For calibration size `n` and desired miscoverage
`alpha`, choose the order statistic

`tau = a_(ceil((n+1)(1-alpha)))`.

Under exchangeability, a future score is at most `tau` with probability at
least `1-alpha`. If the requested rank exceeds `n`, the finite-sample honest
answer is `tau=+infinity`, which disables the relaxed branch. This prevents a
small bucket from manufacturing a precise-looking length threshold.

Feed this `tau` into the union frontier from Iteration 208:

`budget(L) = max(b_absolute, tau*L)`.

Use task-level scores rather than treating multiple views or beams as separate
calibration examples; otherwise correlated copies falsely increase the sample
size. Stratify calibration by output-area bucket, view family, and whether the
shape cap came from the primary or fallback rule. Unseen or underpopulated
target buckets fall back to the absolute frontier.

### Limits and promotion gate

Conformal coverage is marginal under the stated exchangeability assumption; it
does not prove ARC tasks are iid, nor does it guarantee the NLL calibration
survives task-local TTT shift. Use a conservative miscoverage level and a
declared shift penalty if the target bucket differs from calibration. On a
frozen cache, measure actual long-grid recall and pass@2 recovery, not just
calibration coverage.

`experiments/conformal_threshold.py` implements the finite-sample rank and
union-budget contract; CPU self-check passes. The score frontier remains
shadow-only until a labeled fold shows positive unique-class gain per second
with the absolute baseline retained.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 215 — Cheap-first scheduling is value-blind

### Input-only stress test

The current notebook orders tasks by estimated serialized cost, but the official
metric awards one point per test output and a task may contain multiple test
inputs. On the 240-task challenge file there are 259 output positions, with
task proxy costs ranging from 102 to 6,510 tokens. Simulate four workers under
normalized capacities from 55% to 95% of total proxy cost and compare
cost-first, output-count-first, and output-count/cost ordering.

Output-count/cost ordering completes 210, 225, 238, 247, and 254 output
positions at the five capacities, versus cost-first's 209, 224, 236, 245, and
253. It is consistently better by 1–2 positions in this stress model; pure
value-first is less stable. The estimate is label-free but not a wall-clock
measurement.

### Scheduler design

Use `value/cost` as the primary queue priority, where value is calibrated
unresolved output mass rather than merely the number of test inputs. Use the
current cheap-first order as a control and reserve a small tail for high-value
tasks whose expected gain is large but whose cost estimate is uncertain. If
tasks can produce partial outputs, score each output position independently in
the value estimate; if a task is all-or-nothing, retain a task-level value.

The exact optimization is a bounded multiple-knapsack/scheduling problem, so a
greedy ratio is only an approximation. A safe implementation must preserve
deterministic tie breaks, never exceed the global deadline buffer, and report
completed output positions—not just completed task IDs. Promotion requires
same-seed wall-clock replay and non-decreasing pass@2 coverage; the 1–2 output
proxy advantage is too small to justify a blind scored run.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 216 — Calibrated unknown output cost beats worst-case padding

### Audit

The notebook's current `task_cost` counts train inputs/outputs and test inputs,
but cannot see test outputs. A tempting implementation adds the 932-token
maximum whenever the shape predictor abstains. On the 120-task public
evaluation fold, that worsened task-cost rank correlation: Pearson fell from
0.9916 to 0.9797 and Spearman from 0.9894 to 0.9627.

Use a training-only structural calibration instead. Bucket each test input by
area (`<=25`, `26..100`, `101..225`, `>225`) and each task by demonstration
count (`<=2`, `3..4`, `>=5`); store the mean labeled output serialization cost
for buckets with at least 20 training examples, otherwise use the pooled mean.
When an audited shape cap exists, it overrides the estimate; unknown shapes use
the bucket mean rather than the 932-token worst case.

Applied to evaluation, this calibrated estimator reduced mean absolute task
cost error from 625.43 to 81.61 proxy units, improved Pearson correlation to
0.9977 and Spearman to 0.9956, and used no evaluation labels in fitting. The
result is a strong scheduling-control improvement, but serialized cost is still
only a proxy for TTT, DFS branching, batch padding, and scoring time.

`experiments/output_cost_calibration.py` encodes the leakage-safe bucket,
minimum-sample, pooled-fallback, and known-shape-override rules; CPU self-check
passes. The next shadow replay should replace only the queue's cost estimate,
hold the candidate generator and cheap-first/value-per-cost policy fixed, and
measure completed output positions, p90 task time, and pass@2 recovery.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Iteration 217 — Calibrated cost changes the anytime queue materially

### Corrected shadow replay

The first ad hoc composition accidentally calibrated on visible training-output
sizes. That was the wrong target variable. The corrected replay fits bucket means
to labeled *test-output* serialization costs from the training challenge/solution
pairs, then estimates each hidden challenge's total cost as visible cost plus the
calibrated unknown test-output portion. No hidden labels are used for fitting.

On the 240 hidden challenge tasks, the raw visible-cost proxy totals 335,030
units, while the corrected calibrated estimate totals 390,173.7 units. Under
normalized capacity fractions of the corresponding total, deterministic cheap
first ordering completes:

| capacity | raw proxy | calibrated proxy |
|---:|---:|---:|
| 55% | 179 | 192 |
| 65% | 193 | 207 |
| 75% | 206 | 219 |
| 85% | 217 | 229 |
| 95% | 226 | 237 |

This is a queue-ordering result, not a score claim: the simulation counts
completed tasks under serialized-cost capacity and does not model GPU contention,
TTT branching, or candidate quality. It nevertheless supports replacing the
input-only cheap-first key with the leakage-safe calibrated unknown-output cost,
while retaining the current solver and deadline reserve. The next real replay
must measure output positions, wall-clock p90, and pass@2 recovery; absent that,
the change remains a shadow scheduling improvement rather than a release claim.

No model calls, GPU runs, Kaggle submission, or hidden labels were used.

## Checkpoint — 2026-09-02

The autonomous campaign was stopped at the user's request after 9h 29m 02s
(34,142 seconds; 6,708,326 goal tokens). A detailed handoff is saved in
`docs/AUTORESEARCH_CHECKPOINT.md`. It summarizes the 205 ledger entries/latest
numbered Iteration 217, the control artifact, audited positive and negative
results, release gates, and the exact resume protocol. No model calls, GPU runs,
Kaggle submission, or hidden labels were used.

## Sources consulted

- Official Kaggle rules and overview (links above).
- Verantyx public neighborhood-rule description (motivation for the algebraic
  cellular falsification):
  <https://huggingface.co/kofdai/Verantyx-arc-agi2-7.4>.
- ARC Prize announcement and evaluation protocol:
  <https://arcprize.org/blog/announcing-arc-agi-2>.
- ARC Prize 2025 technical report: <https://arxiv.org/abs/2601.10904>.
- Modality-Driven Search with Holistic Trace Judging:
  <https://arxiv.org/abs/2606.31543>.
- MDS full paper release (candidate/judge budgets and post-hoc ablations):
  <https://arc.markbarney.net/paper.pdf>.
- Confluence Labs open-source ARC-AGI-2 solver (public-eval claim, not assumed
  prize-eligible because it requires external APIs):
  <https://github.com/confluence-labs/arc-agi-2>.
- Structural Grid Descriptors Predict Within-Task Solver Success:
  <https://arxiv.org/abs/2606.09026>.
- ARC-TGI task-family generation and episode-level constraints:
  <https://arxiv.org/abs/2603.05099>.
- Kaggle discussion describing an offline library of verified transform
  functions (partial community evidence):
  <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/733501>.
- ARCANA reflective multi-agent program synthesis:
  <https://arxiv.org/abs/2607.09059>.
- Abduction-Based Procedural Refinement / proof-tree debugging:
  <https://arxiv.org/abs/2603.20334>.
- TraceViT grounded intermediate transformation traces:
  <https://arxiv.org/abs/2607.29586>.
- Public deterministic typed DSL engine used as packaging/search evidence:
  <https://github.com/Julien-Livet/aicpp/tree/dsl_engine>.
- Current official ARC-AGI-2 rules:
  <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules>.
- Current official ARC-AGI-2 leaderboard snapshot:
  <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard>.
- ARC Prize evaluation guide:
  <https://arcprize.org/guide/1>.
- Community discussion on a disputed task interpretation:
  <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/731040>.
- Public verified program-synthesis implementation (method evidence only):
  <https://huggingface.co/Interstellar007/arc-agi-2-solver/blob/main/kaggle_submission.py>.
- ARCANA reflective multi-agent program-synthesis paper (iterative proposal,
  verification, and refinement design; reported result not treated as a
  reproducible Kaggle recipe): <https://arxiv.org/abs/2607.09059>.
- Confluence Labs public ARC-AGI-2 solver (parallel agents and repeated
  refinement; API/sandbox-assisted public-eval result, not promoted to the
  offline notebook): <https://github.com/confluence-labs/arc-agi-2>.
- Imbue code-evolution ARC-AGI-2 method (partial-cell fitness, mutation
  diversity, and crossover evidence; hosted-model reproduction not assumed
  Kaggle-legal): <https://imbue.com/blog/2026-02-27-arc-agi-2-evolution>.
- beetree Multi-Model Reflective Reasoning solver (four solver modalities,
  executable search, and logic/consistency judging; API-backed result treated
  as method evidence only): <https://github.com/beetree/ARC-AGI>.
- Poetiq reproducibility repository (hosted-model iterative solver and
  two-attempt packaging; API keys required, so not promoted to the offline
  notebook): <https://github.com/poetiq-ai/poetiq-arc-agi-solver>.
- Procedural Refinement by LLM-driven Algorithmic Debugging (formalized
  failure traces and abductive repair): <https://arxiv.org/abs/2603.20334>.
- Poetiq hosted-model meta-system analysis (adaptive orchestration,
  self-auditing, and iterative feedback; not an offline notebook recipe):
  <https://poetiq.ai/posts/arcagi_announcement/>.
- Poetiq officially verified hosted-model ARC-AGI-2 result and reproducibility
  repository context: <https://poetiq.ai/posts/arcagi_verified/>.
- Tiny Recursive Model: <https://arxiv.org/abs/2510.04871>.
- Probabilistic Tiny Recursive Model (stochastic basin exploration):
  <https://arxiv.org/abs/2605.19943>.
- Recursive Inference Machines for Neural Reasoning:
  <https://arxiv.org/abs/2603.05234>.
- The ARChitects technical report on 2D-aware masked diffusion and recursive
  soft-masked refinement: <https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/>.
- Recursive Scaling in Masked Diffusion Models:
  <https://arxiv.org/abs/2606.18022>.
- Denoising Recursion Models official repository:
  <https://github.com/wwwwwwwwz/DenoisingRecursionModels>.
- One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion
  Models: <https://arxiv.org/abs/2604.18839>.
- Multi-Perspective Transformers in ARC-AGI-2 Challenge (view PoE and TTT
  ablation evidence): <https://arxiv.org/abs/2605.01154>.
- OSI approved-license list: <https://opensource.org/licenses>.
- OSI OpenMDW-1.1 license-review record: <https://lists.opensource.org/pipermail/license-review_lists.opensource.org/2026-August/006126.html>.
- OpenMDW v1.1 repository and license description: <https://github.com/OpenMDW/OpenMDW>.
