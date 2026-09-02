# Nemotron Lightning probe

This directory is intentionally separate from `kaggle_notebook/`. It measures
whether NVIDIA Nemotron 3.5 Lightning can add an ARC program-generation family;
it cannot produce or submit a competition submission.

## Before any Kaggle run

1. Attach `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` as a private
   Kaggle model input and add its exact model-source slug to
   `kernel-metadata.json`.
2. The metadata attaches `vladimiryakunin/vllm-027-cuda-wheels` and pins the
   Kaggle image used by its public reference notebook. This is a CUDA-13 build
   proven on RTX Pro 6000, so L4 boot remains an explicit fail-fast gate.
3. Keep internet off and use the four-L4 competition machine.
4. Build and inspect `notebook.ipynb` with:

       python build_notebook.py

5. Run the notebook as an **unscored public-evaluation commit only**. It refuses
   to execute when `KAGGLE_IS_COMPETITION_RERUN` is set.

The first run uses 24 deterministic development outputs, eight candidates each,
and never touches fold 4. Promotion requires more than demo verification: inspect
oracle pass@8, top-2 exact accuracy, and aggregate generation throughput.
