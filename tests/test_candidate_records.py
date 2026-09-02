from __future__ import annotations

import unittest
import json

from experiments.candidate_records import (
    CandidateRecord,
    grid_hash,
    normalize_grid,
    from_legc_result,
    from_nvarc_sample,
    to_selector_candidates,
    to_task_candidates,
)


class CandidateRecordTests(unittest.TestCase):
    def test_grid_normalization_and_hash_are_stable(self) -> None:
        grid = normalize_grid([[0, 1], [1, 0]])
        self.assertEqual(grid, ((0, 1), (1, 0)))
        self.assertEqual(grid_hash(grid), grid_hash([[0, 1], [1, 0]]))

    def test_correlation_group_survives_selector_adaptation(self) -> None:
        record = CandidateRecord.from_output(
            task_id="task", test_index=0, family="nvarc", candidate_id="sample",
            output=[[0]], correlation_group="checkpoint-a/prompt-1",
        )
        self.assertEqual(
            record.as_selector_candidate().correlation_group,
            "checkpoint-a/prompt-1",
        )

    def test_proof_status_is_separate_from_structural_validity(self) -> None:
        unverified = CandidateRecord.from_output(
            task_id="t", test_index=0, family="nvarc", candidate_id="n",
            output=[[0]], hard_valid=True,
        )
        verified = CandidateRecord.from_output(
            task_id="t", test_index=0, family="legc", candidate_id="p",
            output=[[0]], hard_valid=True, proof_status="demo_verified",
        )
        self.assertEqual(unverified.proof_status, "unverified")
        self.assertEqual(verified.proof_status, "demo_verified")
        with self.assertRaises(ValueError):
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="x", candidate_id="x",
                output=[[0]], proof_status="claimed",
            )

    def test_invalid_grid_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_grid([[0], [1, 2]])
        with self.assertRaises(ValueError):
            normalize_grid([[10]])

    def test_hard_validity_survives_normalization(self) -> None:
        record = CandidateRecord.from_output(
            task_id="t", test_index=0, family="dense", candidate_id="c",
            output=[[0]], hard_valid=False,
        )
        self.assertFalse(to_selector_candidates([record])[0].hard_valid)
        self.assertEqual(json.loads(json.dumps(record.as_dict()))["output"], [[0]])

    def test_only_complete_program_vectors_are_emitted(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="program", candidate_id="p0",
                program_id="p", output=[[0]],
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=1, family="program", candidate_id="p1",
                program_id="p", output=[[1]],
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="program", candidate_id="q0",
                program_id="q", output=[[2]],
            ),
        ]
        vectors = to_task_candidates(records, task_id="t", n_test=2)
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0].output_vector,
                         (grid_hash(((0,),)), grid_hash(((1,),))))

    def test_hard_invalid_position_poisoning_drops_entire_program_vector(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="program", candidate_id="p0",
                program_id="p", output=[[0]],
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=1, family="program", candidate_id="p1",
                program_id="p", output=[[1]], hard_valid=False,
            ),
        ]
        self.assertEqual(to_task_candidates(records, task_id="t", n_test=2), [])

    def test_common_correlation_group_survives_vector_adaptation(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="legc", candidate_id="p0",
                program_id="p", output=[[0]], correlation_group="legc-source",
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=1, family="legc", candidate_id="p1",
                program_id="p", output=[[1]], correlation_group="legc-source",
            ),
        ]
        vectors = to_task_candidates(records, task_id="t", n_test=2)
        self.assertEqual(vectors[0].correlation_group, "legc-source")

    def test_nvarc_adapter_preserves_scores_without_weight_invention(self) -> None:
        record = from_nvarc_sample(
            task_id="t", test_index=0, candidate_id="n",
            sample={"solution": [[0]], "beam_score": -2, "score_aug": [1, 3]},
        )
        self.assertEqual(record.family, "nvarc")
        self.assertEqual(record.weight, 1.0)
        self.assertEqual(record.augmentation_score, 2.0)
        self.assertEqual(record.augmentation_nlls, (1.0, 3.0))
        self.assertEqual(record.beam_score, -2.0)

    def test_nvarc_adapter_preserves_explicit_correlation_group(self) -> None:
        record = from_nvarc_sample(
            task_id="t", test_index=0, candidate_id="n",
            sample={
                "solution": [[0]],
                "correlation_group": "nvarc/checkpoint-a/temperature-2",
            },
        )
        self.assertEqual(record.correlation_group, "nvarc/checkpoint-a/temperature-2")

    def test_nvarc_adapter_does_not_trust_claimed_proof_metadata(self) -> None:
        record = from_nvarc_sample(
            task_id="t", test_index=0, candidate_id="n",
            sample={"solution": [[0]], "proof_status": "demo_verified"},
        )
        self.assertEqual(record.proof_status, "unverified")

    def test_legc_adapter_requires_verification_and_keeps_alternate(self) -> None:
        self.assertEqual(from_legc_result(task_id="t", result={}), [])
        records = from_legc_result(
            task_id="t",
            result={
                "verified": True,
                "outputs": [{"attempt": [[0]], "alt": [[1]], "program": "return 0"}],
            },
        )
        self.assertEqual(len(records), 2)
        self.assertEqual({record.candidate_id for record in records},
                         {"t:0:primary", "t:0:alternate"})
        self.assertTrue(all(record.proof_status == "demo_verified" for record in records))


if __name__ == "__main__":
    unittest.main()
