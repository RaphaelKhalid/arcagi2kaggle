# Training Track & Automated Research Loop

## Where the 12 hours actually go

The 12-hour submission window is **inference plus test-time training, not offline
training**. Timeline of one scored run (from the baseline's own mechanics):

- Model weights arrive pre-trained, attached as a Kaggle model (read-only mount).
- Per task, inside the window: LoRA test-time fine-tuning on that task's augmented demo
  pairs (~minutes), then DFS decoding + candidate rescoring. So "training" does happen in
  the 12h — but only the per-task adaptation, from the same fixed checkpoint every time.
- Offline training — the thing that separates 34% from 72% — happens on OUR time, on OUR
  compute, before submission: continued SFT of the checkpoint on synthetic corpora,
  training a coder model for induction, TRM training. Its output is a new attached model.

Compute pools for offline training:
| pool | what for | limits |
|------|----------|--------|
| Kaggle notebooks (non-submission) | SFT runs on T4x2/P100 (1× quota) or L4x4 (2× quota, competition-attached only) | ~30h GPU/week |
| Local RTX 4060 (8GB) | LoRA pilots on the 2B/4B (4-bit), symbolic/eval work, sandboxed program verification | VRAM-bound, slow |
| Colab / other free tiers | overflow SFT | session limits |
| OpenAI batch ($50 cap) | synthetic task + verified-trace generation only | ledger in SYNTHETIC_DATA.md |

## Automated research loop ("autoresearch")

The daily cadence is mechanizable. Design:

1. **Nightly experiment queue** — a ranked list of one-variable changes (flags in the
   notebook, checkpoint swaps, budget re-allocations) maintained in experiments/LEDGER.md.
2. **Scheduled agent runs** (Claude Code scheduled routines / local loops) that can:
   check the day's submission score when it lands, log it in the ledger, correlate with
   the change under test, and prepare (not launch) the next day's kernel version;
   monitor long Kaggle training notebooks and pull their artifacts.
3. **Human gates stay human**: pushing a kernel (burns quota) and submitting (burns the
   1/day slot) remain explicit user-approved actions. Everything around them — analysis,
   preparation, verification, report — automates.
4. **Failure-analysis automation**: after each local eval run, auto-categorize misses
   (perception / missing primitive / composition / ranking / ambiguity / runtime) by
   comparing candidate pools vs. ground truth; feed categories into synthetic-data
   targeting (SYNTHETIC_DATA.md).

The loop's throughput ceiling is 1 scored measurement/day + ~30h GPU/week — the scarce
resources. Automation exists to make sure neither is ever idle or wasted, not to spend
them faster.
