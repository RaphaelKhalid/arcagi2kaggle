import unittest

from experiments.candidate_cards import card_token_estimate, deduplicate_cards
from experiments.candidate_records import CandidateRecord


class CandidateCardTests(unittest.TestCase):
    def test_exact_output_classes_are_deduplicated(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="a", candidate_id="z",
                output=[[0, 1], [0, 0]], mdl_length=4,
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="b", candidate_id="a",
                output=[[0, 1], [0, 0]], mdl_length=2,
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="c", candidate_id="c",
                output=[[1, 0], [0, 0]], mdl_length=1,
            ),
        ]
        cards = deduplicate_cards(records, reference=[[0, 0], [0, 0]])
        self.assertEqual(len(cards), 2)
        first = next(card for card in cards if card.diff_cells == 1)
        self.assertEqual(first.mdl_length, 2)
        self.assertGreater(card_token_estimate(cards), 0)

    def test_card_metadata_is_color_and_shape_auditable(self) -> None:
        record = CandidateRecord.from_output(
            task_id="t", test_index=0, family="program", candidate_id="p",
            output=[[0, 2, 2], [0, 0, 0]],
        )
        card = deduplicate_cards([record])[0]
        self.assertEqual(card.shape, (2, 3))
        self.assertEqual(card.palette, (0, 2))
        self.assertEqual(card.background, 0)
        self.assertEqual(card.object_count, 1)
        self.assertIn("shape=2x3", card.prompt_line())

    def test_card_preserves_provenance_metadata(self) -> None:
        record = CandidateRecord.from_output(
            task_id="t", test_index=0, family="program", candidate_id="p",
            output=[[0, 1]], correlation_group="g1", proof_status="demo_verified",
        )
        card = deduplicate_cards([record])[0]
        self.assertEqual(card.proof_status, "demo_verified")
        self.assertEqual(card.correlation_group, "g1")
        self.assertIn("proof=demo_verified", card.prompt_line())
        self.assertIn("group=g1", card.prompt_line())

    def test_dedup_prefers_demo_verified_representative(self) -> None:
        records = [
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="neural", candidate_id="u",
                output=[[0, 1]], mdl_length=1, proof_status="unverified",
            ),
            CandidateRecord.from_output(
                task_id="t", test_index=0, family="program", candidate_id="v",
                output=[[0, 1]], mdl_length=9, proof_status="demo_verified",
            ),
        ]
        card = deduplicate_cards(records)[0]
        self.assertEqual(card.family, "program")
        self.assertEqual(card.proof_status, "demo_verified")


if __name__ == "__main__":
    unittest.main()
