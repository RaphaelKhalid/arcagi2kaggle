from __future__ import annotations

import unittest

from experiments.analogy_retrieval import (
    TrainingAnalogy,
    retrieve_analogy_outputs,
)


class AnalogyRetrievalTests(unittest.TestCase):
    def test_d8_and_color_role_equivariance(self) -> None:
        library = (TrainingAnalogy(
            "source", 0, ((0, 1), (0, 0)), ((0, 0), (1, 0))
        ),)
        outputs = retrieve_analogy_outputs([[0, 7], [0, 0]], library)
        self.assertTrue(any(output == ((0, 0), (7, 0)) for _, output in outputs))

    def test_non_bijective_color_match_is_rejected(self) -> None:
        library = (TrainingAnalogy(
            "source", 0, ((0, 1), (0, 0)), ((0, 0), (1, 0))
        ),)
        self.assertEqual(retrieve_analogy_outputs([[2, 2], [2, 2]], library), ())

    def test_output_cap_is_respected(self) -> None:
        library = tuple(
            TrainingAnalogy(str(index), 0, ((0, 1), (0, 0)), ((0, 0), (1, 0)))
            for index in range(4)
        )
        self.assertLessEqual(len(retrieve_analogy_outputs(
            [[0, 1], [0, 0]], library, max_outputs=2
        )), 2)


if __name__ == "__main__":
    unittest.main()
