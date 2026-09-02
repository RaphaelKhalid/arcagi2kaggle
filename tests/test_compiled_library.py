from __future__ import annotations

import unittest

from experiments.compiled_library import (
    LibraryEntry,
    retrieve_verified,
    signature_distance,
)


class CompiledLibraryTests(unittest.TestCase):
    def test_wildcards_and_length_are_deterministic(self) -> None:
        self.assertEqual(signature_distance((1, -1, 3), (1, 2, 4)), 1)
        self.assertEqual(signature_distance((1, 2), (1, 2, 3)), 1)

    def test_demo_certificate_is_a_hard_gate(self) -> None:
        result = retrieve_verified(
            [
                LibraryEntry("unsafe", (1, 2), "p", "u", 0, False),
                LibraryEntry("safe", (1, 3), "p", "s", 2, True),
            ],
            (1, 2),
        )
        self.assertEqual(tuple(entry.entry_id for entry in result), ("safe",))

    def test_exact_signature_precedes_near_match(self) -> None:
        result = retrieve_verified(
            [
                LibraryEntry("near", (1, 3), "p", "n", 1, True),
                LibraryEntry("exact", (1, 2), "p", "e", 9, True),
            ],
            (1, 2),
        )
        self.assertEqual(result[0].entry_id, "exact")


if __name__ == "__main__":
    unittest.main()
