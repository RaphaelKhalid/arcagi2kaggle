import unittest

from experiments.cegis_version_space import (
    Demonstration,
    Program,
    cegis_solve,
    filter_version_space,
    most_discriminating_demo,
    posterior_outputs,
    semantic_posterior_outputs,
)


class CegisVersionSpaceTests(unittest.TestCase):
  def test_exact_cegis_eliminates_first_counterexample(self):
    programs = [Program("a", "a"), Program("b", "b"), Program("c", "c")]

    def execute(program, value):
        if program.program_id == "a":
            return value
        if program.program_id == "b":
            return value + (1 if value == 0 else 2)
        return 0 if value == 0 else 2

    demos = [Demonstration(0, 0), Demonstration(1, 2)]
    self.assertEqual(
        [p.program_id for p in filter_version_space(programs, demos, execute)],
        ["c"],
    )


  def test_discriminating_demo_maximizes_partition(self):
    programs = [Program(str(i), "f") for i in range(4)]

    def execute(program, value):
        return int(program.program_id) // (2 if value == "coarse" else 1)

    self.assertEqual(most_discriminating_demo(
        programs,
        [Demonstration("coarse", 0), Demonstration("fine", 0)],
        execute,
    ), 1)


  def test_output_quotient_collapses_syntactic_duplicates(self):
    programs = [Program("p1", "f"), Program("p2", "f"), Program("p3", "f")]

    def execute(program, _):
        return 7 if program.program_id != "p3" else 8

    masses = posterior_outputs(programs, "x", execute)
    self.assertEqual(masses, ((7, 2 / 3), (8, 1 / 3)))


  def test_family_normalization_prevents_correlated_vote_flooding(self):
    programs = [
        *(Program(f"copy-{i}", "copy") for i in range(10)),
        Program("synth", "synth"),
        Program("other", "other"),
    ]

    def execute(program, _):
        return {"copy": "wrong", "synth": "right", "other": "other"}[program.family]

    masses = posterior_outputs(
        programs, "x", execute,
        family_priors={"copy": 0.5, "synth": 0.3, "other": 0.2},
    )
    self.assertEqual(masses[0][0], "wrong")
    self.assertEqual({output for output, _ in masses[:2]}, {"wrong", "right"})


  def test_semantic_quotient_collapses_same_family_output_copies(self):
    programs = [
        *(Program(f"wrong-{i}", "decoder") for i in range(10)),
        Program("right", "decoder"),
    ]

    def execute(program, _):
        return "right" if program.program_id == "right" else "wrong"

    masses = semantic_posterior_outputs(programs, "x", execute)
    self.assertEqual(masses, (("right", 0.5), ("wrong", 0.5)))


  def test_semantic_quotient_retains_best_mdl_representative(self):
    programs = [
        Program("long-right", "decoder", mdl_length=4.0),
        Program("short-right", "decoder", mdl_length=1.0),
        Program("wrong", "decoder", mdl_length=2.0),
    ]

    def execute(program, _):
        return "wrong" if program.program_id == "wrong" else "right"

    masses = semantic_posterior_outputs(programs, "x", execute)
    self.assertEqual(masses[0][0], "right")
    self.assertAlmostEqual(masses[0][1], 2.0 / 3.0)


  def test_cegis_uses_independent_top_two_per_test_input(self):
    programs = [Program("p0", "f"), Program("p1", "f"), Program("p2", "f")]

    def execute(program, value):
        return {"p0": ("a", "x"), "p1": ("a", "y"), "p2": ("b", "y")}[
            program.program_id
        ][value]

    result = cegis_solve(
        programs,
        [],
        [0, 1],
        execute,
    )
    self.assertEqual(result.selected_outputs, (("a", "b"), ("y", "x")))
