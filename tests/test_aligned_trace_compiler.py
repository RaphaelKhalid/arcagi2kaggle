import unittest

from experiments.aligned_trace_compiler import compile_aligned_frame_programs
from experiments.frame_role_executor import execute_frame_program


class AlignedTraceCompilerTests(unittest.TestCase):
    def test_aligned_compiler_emits_exact_multi_action_program(self):
        task = {"train": [{
            "input": [[0, 1, 0, 0, 2]],
            "output": [[0, 0, 1, 0, 3]],
        }]}
        programs = compile_aligned_frame_programs(task)
        self.assertTrue(programs)
        self.assertEqual(
            execute_frame_program(programs[0], [[0, 5, 0, 0, 6]]),
            ((0, 0, 5, 0, 3),),
        )

    def test_unsupported_transform_stays_rejected(self):
        task = {"train": [{
            "input": [[0, 1, 0]], "output": [[0, 1, 1]],
        }]}
        self.assertFalse(compile_aligned_frame_programs(task))

    def test_hypothesis_cap_is_explicit(self):
        with self.assertRaises(ValueError):
            compile_aligned_frame_programs({"train": []}, max_hypotheses=0)


if __name__ == "__main__":
    unittest.main()
