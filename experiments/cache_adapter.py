"""Adapters and coverage inventory for the notebook's decoder cache.

The Kaggle notebook persists NVARC results as
`{task_id}_{test_index} -> {view/subkey -> sample}`.  Normalizing that shape
immediately makes selection experiments replayable and prevents a future
12-hour run from being reduced to only its final submission file.
"""

from __future__ import annotations

import bz2
from collections import Counter
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping

try:
    from experiments.candidate_records import CandidateRecord, from_nvarc_sample
except ModuleNotFoundError:  # direct ``python experiments/cache_adapter.py``
    from candidate_records import CandidateRecord, from_nvarc_sample


def _split_base_key(base_key: str) -> tuple[str, int]:
    if not isinstance(base_key, str) or "_" not in base_key:
        raise ValueError("decoded cache key must be task_id_test_index")
    task_id, index = base_key.rsplit("_", 1)
    try:
        return task_id, int(index)
    except ValueError as exc:
        raise ValueError("decoded cache key has a non-integer test index") from exc


def records_from_decoded_cache(
    decoded_results: Mapping[str, Mapping[str, Any]],
) -> list[CandidateRecord]:
    """Normalize all valid NVARC samples from an in-memory decoder cache."""

    records: list[CandidateRecord] = []
    for base_key, samples in decoded_results.items():
        task_id, test_index = _split_base_key(base_key)
        if not isinstance(samples, Mapping):
            raise ValueError("decoded cache samples must be a mapping")
        for candidate_id, sample in samples.items():
            if not isinstance(sample, Mapping) or "solution" not in sample:
                continue
            records.append(from_nvarc_sample(
                task_id=task_id,
                test_index=test_index,
                candidate_id=str(candidate_id),
                sample=sample,
            ))
    return records


def load_decoded_cache(store: str | Path) -> list[CandidateRecord]:
    """Read the notebook's one-file-per-position bz2/pickle cache."""

    decoded: dict[str, dict[str, Any]] = {}
    root = Path(store)
    if not root.is_dir():
        raise ValueError("decoder cache path must be a directory")
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        base_key = path.name.split(".", 1)[0]
        with bz2.BZ2File(path, "rb") as handle:
            samples = pickle.load(handle)
        if not isinstance(samples, (list, tuple)):
            raise ValueError("decoder cache file must contain a sample list")
        decoded[base_key] = {
            f"{path.name}:out{index}": sample
            for index, sample in enumerate(samples)
        }
    return records_from_decoded_cache(decoded)


def cache_inventory(records: Iterable[CandidateRecord]) -> dict[str, Any]:
    """Summarize cache coverage without inspecting correctness labels."""

    records = list(records)
    positions = {(record.task_id, record.test_index) for record in records}
    classes = {(record.task_id, record.test_index, record.output_hash)
               for record in records}
    families = Counter(record.family for record in records)
    return {
        "records": len(records),
        "task_positions": len(positions),
        "output_classes": len(classes),
        "families": dict(sorted(families.items())),
    }


if __name__ == "__main__":
    print("cache_adapter selftest: PASS")
