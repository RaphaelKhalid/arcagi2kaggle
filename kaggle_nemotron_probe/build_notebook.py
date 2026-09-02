"""Assemble the Rawr-AGI-2 Nemo submission notebook.

Fork of the proven Launch B baseline (kaggle_notebook/, scored 29.31 on the
public LB): every base cell (env setup, arc_loader/arc_decoder/symbolic_size/
arc_solver, starter.py, submission cell) is byte-identical. The only change is
Leg C's induction engine: Qwen2.5-Coder-7B (arc_induction_v2.py, plain
transformers bf16) is replaced with Nemotron-3.5-Lightning-30B-A3B-NVFP4 via
vLLM (nemotron_induction.py, reusing probe_core.py's tested AST-sandboxed
verifier). The Leg C merge cell that promotes verified outputs into
submission.json is unmodified -- it already only cares about the shape of
/kaggle/working/induction_results.json, not which model wrote it.

vLLM is installed into an isolated target directory and given PYTHONPATH only
for the induction subprocess call (not the whole kernel's os.environ), so it
never leaks into the unsloth-based base pipeline subprocess -- avoiding the
torch-ABI conflict between vLLM's and unsloth's pinned torch versions.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def writefile_cell(filename: str) -> dict:
    content = (ROOT / filename).read_text(encoding="utf-8")
    return code_cell(f"%%writefile {filename}\n{content}")


def main() -> None:
    notebook = {
        "cells": [
            markdown_cell(
                "# Rawr-AGI-2 Nemo\n\n"
                "Fork of the proven Launch B baseline (`kaggle_notebook/`, public LB 29.31): "
                "environment setup, `arc_loader.py`, `arc_decoder.py`, `symbolic_size.py`, "
                "`arc_solver.py`, `starter.py`, the submission cell, and the Leg C merge cell "
                "are byte-identical to that run.\n\n"
                "**The only change:** Leg C's induction engine is now "
                "NVIDIA Nemotron-3.5-Lightning-30B-A3B-NVFP4 via vLLM "
                "(`nemotron_induction.py`) instead of Qwen2.5-Coder-7B. It reuses the same "
                "AST-sandboxed, demo-verified-only candidate acceptance as the tested probe "
                "core, and writes `/kaggle/working/induction_results.json` in the exact shape "
                "the (unmodified) Leg C merge cell already consumes.\n\n"
                "This **is** a submission notebook: it writes `submission.json` from the base "
                "pipeline regardless of whether Nemotron induction verifies anything, then the "
                "merge cell promotes any verified Nemotron outputs to attempt_1. If Nemotron "
                "verifies nothing, this is byte-equivalent to the Launch B baseline.\n\n"
                "vLLM installs into an isolated site-packages directory; its PYTHONPATH is "
                "passed only to the induction subprocess, never to the base (unsloth) "
                "pipeline's subprocess, to avoid a torch-ABI conflict between the two "
                "runtimes.\n"
            ),
            code_cell(
                "# Keep the original global 10-minute submission/write buffer.\n"
                "import time\n"
                "global_end_time = time.time() + 12 * 3600 - 600\n"
            ),
            code_cell(
                "# Preserve the baseline environment workaround.\n"
                "!pip uninstall -y tensorflow\n"
            ),
            writefile_cell("arc_loader.py"),
            writefile_cell("arc_decoder.py"),
            writefile_cell("symbolic_size.py"),
            writefile_cell("arc_solver.py"),
            writefile_cell("starter.py"),
            writefile_cell("probe_core.py"),
            writefile_cell("nemotron_induction.py"),
            code_cell(
                "# CPU-only safety and metric smoke test for the induction verifier.\n"
                "import subprocess, sys\n"
                "subprocess.run([sys.executable, 'probe_core.py', 'selftest'], check=True)\n"
            ),
            code_cell(
                "# Install the attached offline vLLM 0.27 CUDA wheelhouse into an isolated\n"
                "# target -- NOT the base site-packages, and NOT the kernel's os.environ, so\n"
                "# it cannot leak into the unsloth-based base pipeline's own subprocess.\n"
                "import pathlib, subprocess, sys\n"
                "wheel_candidates = [\n"
                "    pathlib.Path('/kaggle/input/vllm-027-cuda-wheels'),\n"
                "    pathlib.Path('/kaggle/input/datasets/vladimiryakunin/vllm-027-cuda-wheels'),\n"
                "]\n"
                "vllm_wheelhouse = next((p for p in wheel_candidates if (p / 'requirements.lock').is_file()), None)\n"
                "if vllm_wheelhouse is None:\n"
                "    raise FileNotFoundError(f'offline vLLM wheelhouse missing: {wheel_candidates}')\n"
                "vllm_target = pathlib.Path('/kaggle/working/vllm-site-packages')\n"
                "vllm_target.mkdir(parents=True, exist_ok=True)\n"
                "subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',\n"
                "    '--find-links', str(vllm_wheelhouse), '--requirement', str(vllm_wheelhouse / 'requirements.lock'),\n"
                "    '--target', str(vllm_target), '--upgrade', '--ignore-installed',\n"
                "    '--only-binary', ':all:', '--no-compile', '--disable-pip-version-check',\n"
                "    '--no-warn-conflicts'], check=True)\n"
                "print('[induction] isolated vLLM site-packages at', vllm_target)\n"
            ),
            code_cell(
                "# ===================== Leg C: verified program induction (pre-pass) =====================\n"
                "# Nemotron-3.5-Lightning-30B-A3B-NVFP4 (vLLM) samples candidate transform()\n"
                "# programs per task; probe_core's AST-whitelisted, subprocess-isolated sandbox\n"
                "# executes them against EVERY demonstration pair. Fully verified programs'\n"
                "# outputs pre-empt attempt_1 in the final merge and their tasks are removed\n"
                "# from the base queue (time refund). Empty results (missing model, budget too\n"
                "# small, or nothing verifies) -> starter and merge are no-ops, i.e. this\n"
                "# notebook degrades exactly to the Launch B baseline.\n"
                "# LEGC_ENABLED = False -> nothing runs, no results file -> byte-equivalent\n"
                "# baseline behavior. Must match the flag in starter.py.\n"
                "import os\n"
                "import sys\n"
                "import subprocess\n"
                "import time\n"
                "\n"
                "LEGC_ENABLED   = True\n"
                "LEGC_BUDGET_H  = 1.0   # wall budget of the whole induction phase (public-head value)\n"
                "BASE_RESERVE_H = 9.8   # wall hours guaranteed to remain for the base pipeline\n"
                "# Research controls: defaults preserve the one-batch/raw-majority contract.\n"
                "LEGC_LINEAGES = 1\n"
                "LEGC_LINEAGE_AWARE = False\n"
                "LEGC_LINEAGE_TEMPERATURES = '1.0'\n"
                "LEGC_PROMPT_STYLES = 'strict'\n"
                "LEGC_DIAGNOSTICS = False\n"
                "LEGC_DIAGNOSTIC_TRACE_CHARS = 6000\n"
                "\n"
                "# Same 4 smoke tasks as the base pipeline's eval mode (keeps the commit run short).\n"
                "SMOKE_TASKS = \"0934a4d8,36a08778,981571dc,aa4ec2a5\"\n"
                "\n"
                "if LEGC_ENABLED:\n"
                "    # Sandbox selftest already ran above (CPU-only). vLLM-dependent gate next:\n"
                "    # fail here, before model loading, if the isolated install is broken.\n"
                "    induction_env = dict(os.environ)\n"
                "    induction_env['PYTHONPATH'] = str(vllm_target) + os.pathsep + os.environ.get('PYTHONPATH', '')\n"
                "    gate = subprocess.run(\n"
                "        [sys.executable, '-c',\n"
                "         'import torch, vllm; from packaging.version import Version; '\n"
                "         \"print('torch', torch.__version__, 'cuda', torch.version.cuda); \"\n"
                "         \"print('vllm', vllm.__version__); \"\n"
                "         \"assert Version(vllm.__version__) >= Version('0.27.1'); \"\n"
                "         'assert torch.cuda.device_count() == 4, torch.cuda.device_count()'],\n"
                "        env=induction_env, check=False)\n"
                "    if gate.returncode != 0:\n"
                "        print('[LegC] vLLM dependency gate failed -- skipping Nemotron induction, '\n"
                "              'base pipeline keeps the full budget', flush=True)\n"
                "    else:\n"
                "        # Enforce both controls: the Leg-C cap and the reserved base window.\n"
                "        legc_end_ts = min(\n"
                "            time.time() + LEGC_BUDGET_H * 3600,\n"
                "            global_end_time - BASE_RESERVE_H * 3600,\n"
                "        )\n"
                "        cmd = [sys.executable, 'nemotron_induction.py',\n"
                "               '--budget-h', f'{LEGC_BUDGET_H:.3f}',\n"
                "               '--end-ts', f'{legc_end_ts:.0f}',\n"
                "               '--lineages', str(LEGC_LINEAGES),\n"
                "               '--lineage-temperatures', LEGC_LINEAGE_TEMPERATURES,\n"
                "               '--prompt-styles', LEGC_PROMPT_STYLES,\n"
                "               '--diagnostic-trace-chars', str(LEGC_DIAGNOSTIC_TRACE_CHARS)]\n"
                "        if LEGC_LINEAGE_AWARE:\n"
                "            cmd.append('--lineage-aware')\n"
                "        if LEGC_DIAGNOSTICS:\n"
                "            cmd.append('--diagnostics')\n"
                "        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):\n"
                "            cmd += ['--tasks', SMOKE_TASKS]\n"
                "        print('[LegC] launch:', ' '.join(cmd), flush=True)\n"
                "        subprocess.run(cmd, env=induction_env, check=False)\n"
                "else:\n"
                "    print('[LegC] disabled -- base pipeline keeps the full budget')\n"
            ),
            code_cell(
                "!UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 python starter.py --end-time {global_end_time}\n"
            ),
            code_cell(
                "import os\n"
                "import json\n"
                "import numpy as np\n"
                "from arc_loader import ArcDataset\n"
                "from arc_decoder import ArcDecoder, score_full_probmul_3, score_kgmon, score_log_evidence\n"
                "\n"
                "rerun_mode = os.getenv(\"KAGGLE_IS_COMPETITION_RERUN\")\n"
                "\n"
                "if rerun_mode:\n"
                "    data = ArcDataset.from_file(\"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json\")\n"
                "else:\n"
                "    data = ArcDataset.from_file(\"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json\")\n"
                "    data = data.load_replies(\"/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json\")\n"
                "\n"
                "decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)\n"
                "\n"
                "decoder.load_decoded_results(\"/kaggle/inference_outputs\")\n"
                "\n"
                "# ---- Fork v1: diverse second attempt (the only change vs. the LB33.89 baseline) ----\n"
                "# Baseline behavior: run_selection_algo() ranks candidates with score_kgmon and the\n"
                "# submission takes its top-2. Diverse mode keeps the score_kgmon top pick as attempt_1\n"
                "# (identical to the baseline's attempt_1) and uses the score_full_probmul_3 top pick as\n"
                "# attempt_2 whenever the two algorithms disagree, so the attempts test two scoring\n"
                "# hypotheses instead of ranks 1-2 of a single ranking. When both algorithms agree on the\n"
                "# winner (or probmul_3 has no differing candidate), fall back to the exact baseline top-2.\n"
                "# DIVERSE_ATTEMPT_2 = False restores baseline selection exactly.\n"
                "DIVERSE_ATTEMPT_2 = True\n"
                "# Optional cache-calibrated alternative; False keeps probmul_3 as the control.\n"
                "LOG_EVIDENCE_ATTEMPT_2 = False\n"
                "\n"
                "def select_attempts(decoder):\n"
                "    baseline = decoder.run_selection_algo()  # default = score_kgmon (baseline)\n"
                "    if not DIVERSE_ATTEMPT_2:\n"
                "        return baseline\n"
                "    secondary_algo = score_log_evidence if LOG_EVIDENCE_ATTEMPT_2 else score_full_probmul_3\n"
                "    secondary = decoder.run_selection_algo(secondary_algo)\n"
                "    results = {}\n"
                "    for bk, ranked in baseline.items():\n"
                "        second = None\n"
                "        if ranked:\n"
                "            second = next((g for g in secondary.get(bk, []) if not np.array_equal(g, ranked[0])), None)\n"
                "        results[bk] = ranked if second is None else [ranked[0], second]\n"
                "    return results\n"
                "\n"
                "submission = data.get_submission(select_attempts(decoder))\n"
                "\n"
                "with open(\"submission.json\", \"w\") as f:\n"
                "    json.dump(submission, f)\n"
                "\n"
                "if not rerun_mode:\n"
                "    decoder.benchmark_selection_algos()\n"
                "    with open(\"submission.json\", \"r\") as f:\n"
                "        reload_submission = json.load(f)\n"
                "    print(\"*** Reload score:\", data.validate_submission(reload_submission))\n"
            ),
            code_cell(
                "# ===================== Leg C merge (run AFTER the submission cell) =====================\n"
                "# Monotonic union: train-verified induction outputs pre-empt attempt_1; the base\n"
                "# candidate is demoted to attempt_2 (cross-family decorrelation). Tasks without a\n"
                "# verified program are left untouched. LEGC_ENABLED=False or a missing results file\n"
                "# makes this a no-op.\n"
                "import os\n"
                "import json\n"
                "\n"
                "if LEGC_ENABLED:\n"
                "    SUB = \"submission.json\"\n"
                "    IND = \"/kaggle/working/induction_results.json\"\n"
                "\n"
                "    with open(SUB) as f:\n"
                "        sub = json.load(f)\n"
                "    ind = {}\n"
                "    if os.path.exists(IND):\n"
                "        with open(IND) as f:\n"
                "            ind = json.load(f)\n"
                "\n"
                "    rerun_mode = os.getenv(\"KAGGLE_IS_COMPETITION_RERUN\")\n"
                "\n"
                "    def local_score(s):\n"
                "        try:\n"
                "            return data.validate_submission(s)   # `data` from the submission cell\n"
                "        except Exception:\n"
                "            return None\n"
                "\n"
                "    before = None if rerun_mode else local_score(sub)\n"
                "\n"
                "    promoted = created = 0\n"
                "    for tid, rec in ind.items():\n"
                "        if not isinstance(rec, dict) or not rec.get(\"verified\"):\n"
                "            continue\n"
                "        for ti, o in enumerate(rec.get(\"outputs\", [])):\n"
                "            g = o.get(\"attempt\")\n"
                "            if not g:\n"
                "                continue\n"
                "            alt = o.get(\"alt\")\n"
                "            if tid not in sub:\n"
                "                sub[tid] = []\n"
                "            while len(sub[tid]) <= ti:\n"
                "                sub[tid].append({\"attempt_1\": [[0]], \"attempt_2\": [[0]]})\n"
                "                created += 1\n"
                "            e = sub[tid][ti]\n"
                "            if e[\"attempt_1\"] != g:\n"
                "                demoted = e[\"attempt_1\"]\n"
                "                e[\"attempt_2\"] = demoted if demoted != [[0]] else (alt or demoted)\n"
                "                e[\"attempt_1\"] = g\n"
                "                promoted += 1\n"
                "            elif alt and e.get(\"attempt_2\") == [[0]]:\n"
                "                e[\"attempt_2\"] = alt\n"
                "\n"
                "    with open(SUB, \"w\") as f:\n"
                "        json.dump(sub, f)\n"
                "\n"
                "    after = None if rerun_mode else local_score(sub)\n"
                "    print(f\"[LegC merge] promoted={promoted} created={created}\")\n"
                "    if before is not None and after is not None:\n"
                "        print(f\"[LegC merge] local score: base {before:.2f} -> merged {after:.2f} (unit = tasks)\")\n"
                "else:\n"
                "    print(\"[LegC] merge skipped (disabled)\")\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / "notebook.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
