# arcagi2kaggle

Open-source campaign for the [ARC Prize 2026 — ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) Kaggle competition.

Read the public research site: [ARC/RESEARCH](https://raphaelkhalid.github.io/arcagi2kaggle/).

Approach: the public NVARC-lineage test-time-training baseline (Qwen3-4B, per-task LoRA
fine-tuning, DFS decoding, augmentation-consistency selection) extended with neurosymbolic
components — falsification-based symbolic predictors that constrain decoding, verified
program induction as a diverse second attempt, and improved candidate selection.

## Layout

- `arc/` — evaluation harness: task loading, submission validation, official-metric scoring,
  deterministic 5-fold splits (fold 4 is a shadow fold, held out between milestones).
- `symbolic/` — falsification-first symbolic predictors (output size, palette) used to
  prune and verify neural candidates. Rules only fire when consistent with every
  demonstration pair.
- `kaggle_notebook/` — the submission notebook, its kernel metadata, baseline analysis,
  and launch runbook.
- `scripts/` — utilities, including the secure Kaggle data downloader
  (`python scripts/secure_kaggle_download.py`; interactive, token never touches disk).
- `data/` (gitignored) — official competition data. Not redistributed, per competition
  rules; fetch it with the download script.
- `.tools/` (gitignored) — vendored Python packages (Kaggle CLI etc.), used via `PYTHONPATH`.
- `sources/` — read-only synced reference material.

## Conventions

- Credentials never touch disk: no `kaggle.json`, no tokens in files or code. Kaggle auth
  is passed via `KAGGLE_API_TOKEN` in process environment only.
- Experiments are measured on the 120-task evaluation set locally before any leaderboard
  submission. One scored submission per day — every one is a controlled measurement.

## License

MIT-0 (see `LICENSE`). Competition data is Apache 2.0, © ARC Prize Foundation, and is not
included in this repository.
