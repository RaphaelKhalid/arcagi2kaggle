import unittest
import bz2
import pickle
from tempfile import TemporaryDirectory
from pathlib import Path

from experiments.cache_adapter import (
    cache_inventory,
    load_decoded_cache,
    records_from_decoded_cache,
)


class CacheAdapterTests(unittest.TestCase):
    def test_decoder_cache_is_normalized(self) -> None:
        records = records_from_decoded_cache({
            "abc_0": {
                "view0": {"solution": [[0, 1]], "beam_score": 0.2,
                           "score_aug": [0.1, 0.2]},
                "view1": {"solution": [[0, 1]], "beam_score": 0.3,
                           "score_aug": [0.2, 0.3]},
                "bad": {"not_a_sample": True},
            },
            "abc_1": {
                "view0": {"solution": [[1]], "score_aug": []},
            },
        })
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].task_id, "abc")
        self.assertEqual(records[0].test_index, 0)
        self.assertEqual(cache_inventory(records), {
            "records": 3,
            "task_positions": 2,
            "output_classes": 2,
            "families": {"nvarc": 3},
        })

    def test_malformed_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            records_from_decoded_cache({"abc": {}})
        with self.assertRaises(ValueError):
            records_from_decoded_cache({"abc_x": {}})

    def test_compressed_notebook_directory_is_read(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "abc_0.rot90.bz2"
            with bz2.BZ2File(path, "wb") as handle:
                pickle.dump([{"solution": [[0, 1]], "score_aug": []}], handle)
            records = load_decoded_cache(directory)
            self.assertEqual(cache_inventory(records)["task_positions"], 1)
            self.assertEqual(records[0].candidate_id, "abc_0.rot90.bz2:out0")


if __name__ == "__main__":
    unittest.main()
