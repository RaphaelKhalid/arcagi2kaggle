# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local mirror of the ChatGPT project "arcagi2kaggle" — a workspace for the ARC Prize 2026 Kaggle competition (`arc-prize-2026-arc-agi-2`). It is not a git repository.

Rules from `AGENTS.md`:
- Everything under `sources/` is read-only reference material synced from the ChatGPT project. Do not edit, rename, move, or delete synced files — they may be replaced on the next sync.

## Layout

- `scripts/secure_kaggle_download.py` — the only project code. Prompts for a Kaggle API token with echo disabled (never written to disk), verifies competition access, downloads the official bundle into `data/raw/`, safely extracts it, and validates the JSON files. Run with `python scripts/secure_kaggle_download.py`; it is interactive (uses `getpass`), so the user must run it themselves.
- `sources/` — synced ChatGPT project files (currently empty). Read-only.
- `.tools/` — vendored Python packages (kaggle, jupytext, nbformat, requests, etc.), Python 3.12. Gitignored; treat as a dependency directory, not project code. Add it to `PYTHONPATH` (or `sys.path`) to use these packages without installing anything.
- `data/`, `artifacts/`, `submission.json` — gitignored competition data and generated outputs; `data/raw/` holds the downloaded competition bundle.

## Conventions

- Credentials never touch disk: no `kaggle.json`, no tokens in `.env` files or code. The download script's pattern (in-memory token via child-process environment) is the model to follow for anything needing Kaggle auth.
