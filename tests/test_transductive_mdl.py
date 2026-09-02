import unittest

from experiments.candidate_records import CandidateRecord
from experiments.transductive_mdl import output_description_bits, rank_transductive


class TransductiveMdlTests(unittest.TestCase):
    def record(self, candidate_id, output, mdl_length=1, proof_status="unverified"):
        return CandidateRecord.from_output(
            task_id="t", test_index=0, family="f", candidate_id=candidate_id,
            output=output, mdl_length=mdl_length, proof_status=proof_status,
        )

    def test_sparse_delta_is_shorter_than_dense_change(self) -> None:
        reference = [[0, 0, 0], [0, 0, 0]]
        sparse = [[0, 1, 0], [0, 0, 0]]
        dense = [[1, 2, 3], [4, 5, 6]]
        self.assertLess(
            output_description_bits(reference, sparse),
            output_description_bits(reference, dense),
        )

    def test_shape_change_pays_full_grid_code(self) -> None:
        same = output_description_bits([[0, 1]], [[0, 1]])
        resized = output_description_bits([[0, 1]], [[0, 1, 0]])
        self.assertGreater(resized, same)

    def test_verified_filter_and_output_quotient_are_explicit(self) -> None:
        records = [
            self.record("short-neural", [[0, 1]], 1),
            self.record("verified", [[0, 1]], 9, "demo_verified"),
            self.record("other", [[1, 0]], 2),
        ]
        classes = rank_transductive(records, [[0, 0]])
        self.assertEqual(len(classes), 2)
        self.assertIn("verified", [item.candidate_id for item in classes])
        self.assertTrue(any(item.proof_status == "demo_verified" for item in classes))
        verified = rank_transductive(records, [[0, 0]], verified_only=True)
        self.assertEqual([item.candidate_id for item in verified], ["verified"])


if __name__ == "__main__":
    unittest.main()
