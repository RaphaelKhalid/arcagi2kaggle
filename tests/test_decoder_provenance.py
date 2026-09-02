import unittest

from experiments.decoder_provenance import (
    DecodedCandidate,
    rank_decoder_classes,
    select_decoder_pass2,
)


class DecoderProvenanceTests(unittest.TestCase):
    def test_same_lineage_copies_do_not_flood_a_class(self) -> None:
        candidates = [
            *(DecodedCandidate("wrong", lineage="a", view_family="rot") for _ in range(20)),
            DecodedCandidate("right", lineage="b", view_family="rot"),
        ]
        ranked = rank_decoder_classes(candidates)
        self.assertEqual({item.output_hash for item in ranked}, {"wrong", "right"})
        self.assertAlmostEqual(sum(item.mass for item in ranked), 1.0)

    def test_missing_provenance_is_one_conservative_group(self) -> None:
        selected = select_decoder_pass2([
            DecodedCandidate("a"), DecodedCandidate("a"),
            DecodedCandidate("b"), DecodedCandidate("c"),
        ])
        self.assertEqual(selected, ("a", "b"))

    def test_duplicate_class_within_group_uses_best_weight(self) -> None:
        ranked = rank_decoder_classes([
            DecodedCandidate("a", 0.1, "x", "rot"),
            DecodedCandidate("a", 0.9, "x", "rot"),
            DecodedCandidate("b", 0.5, "x", "rot"),
        ])
        self.assertEqual(ranked[0].output_hash, "a")
        self.assertAlmostEqual(ranked[0].mass, 0.9 / 1.4)

    def test_invalid_candidate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rank_decoder_classes([DecodedCandidate("")])
        with self.assertRaises(ValueError):
            rank_decoder_classes([DecodedCandidate("a", -1.0)])


if __name__ == "__main__":
    unittest.main()
