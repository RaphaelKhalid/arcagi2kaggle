# Experiment & Submission Ledger

Every scored submission and every local milestone measurement gets a row. Local eval =
120-task evaluation set (172 test outputs) unless noted.

## Scored submissions

| # | date | notebook version | change under test | local eval | public LB | notes |
|---|------|------------------|-------------------|-----------|-----------|-------|
| 1 | 2026-08-31 (staged) | e5834e5 Launch B | 4 flags vs 33.89 baseline: DIVERSE_ATTEMPT_2 + CHEAP_FIRST_ORDER + SIZE_CAP_TOKENS + LEGC_ENABLED | local tests green (no full GPU eval possible) | pending | awaiting user permission grant → push → smoke (~35-40min) → verify_kernel_run.py → submit |

## Local measurements

| date | component | configuration | metric | value | notes |
|------|-----------|---------------|--------|-------|-------|
| 2026-08-31 | data profile | — | GPU-min/task budget | ≈12 | 240 tasks, ~260 outputs, 12h on 4×L4 |

## Failure taxonomy (running)

Categories: perception / missing primitive / composition / ranking (right candidate, wrong
rank) / ambiguity (two rules fit demos) / runtime (timeout before decode).
