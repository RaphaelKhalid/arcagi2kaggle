import unittest

from experiments.cegis_version_space import Demonstration
from experiments.evolutionary_induction import (
    Fitness,
    Genome,
    crossover_genomes,
    bounded_frontier,
    evaluate_genome,
    evolve_generation,
    guided_mutations,
    mutate_genome,
    mutation_shell,
    pareto_frontier,
    partial_cell_match,
)


class EvolutionaryInductionTests(unittest.TestCase):
    def test_mutations_are_unique_deterministic_and_bounded(self):
        mutations = mutate_genome(Genome(("a", "b")), ["a", "b", "c"], max_steps=2)
        self.assertEqual(len(mutations), len(set(mutations)))
        self.assertTrue(all(len(item.operations) <= 2 for item in mutations))
        self.assertEqual(mutations, mutate_genome(Genome(("a", "b")), ["c", "b", "a"], max_steps=2))
        with self.assertRaises(ValueError):
            mutate_genome(Genome(("a", "b", "c")), ["a"], max_steps=2)

    def test_mutation_shell_is_bounded_and_radius_one_compatible(self):
        one = mutation_shell(Genome(("a",)), ["a", "b", "c"], radius=1)
        self.assertEqual(one, mutate_genome(Genome(("a",)), ["a", "b", "c"]))
        two = mutation_shell(
            Genome(("a",)), ["a", "b", "c"], radius=2, max_candidates=20
        )
        self.assertLessEqual(len(two), 20)
        self.assertIn(Genome(("b", "b")), two)

    def test_guided_mutation_rejects_multi_radius(self):
        with self.assertRaises(ValueError):
            evolve_generation(
                [Genome(("a",))], [Demonstration(0, 0)], lambda item, value: value,
                ["a", "b"], divergence_indices={Genome(("a",)): 1},
                mutation_radius=2,
            )

    def test_first_divergence_limits_edits_to_local_operation(self):
        guided = guided_mutations(Genome(("a", "b", "c")), 2, ["a", "b", "c", "d"])
        self.assertTrue(guided)
        self.assertTrue(all(
            item.operations[:1] == ("a",)
            and "c" in item.operations
            for item in guided
        ))

    def test_crossover_is_deterministic_and_bounded(self):
        children = crossover_genomes(Genome(("a", "b")), Genome(("c", "d")), max_steps=3)
        self.assertEqual(children, crossover_genomes(Genome(("c", "d")), Genome(("a", "b")), max_steps=3))
        self.assertTrue(all(len(child.operations) <= 3 for child in children))
        self.assertIn(Genome(("a", "d")), children)

    def test_exact_demo_fitness_is_measured_before_test_prediction(self):
        genome = Genome(("inc",))
        fitness = evaluate_genome(
            genome,
            [Demonstration(0, 1), Demonstration(1, 2)],
            lambda item, value: value + (1 if item.operations == ("inc",) else 0),
        )
        self.assertTrue(fitness.exact)
        self.assertEqual(fitness.correct_demos, 2)

    def test_pareto_frontier_keeps_accuracy_complexity_tradeoffs(self):
        exact_long = Fitness(Genome(("a", "b")), 2, 2, 2.0)
        exact_short = Fitness(Genome(("a",)), 2, 2, 1.0)
        partial_short = Fitness(Genome(("c",)), 1, 2, 0.5)
        frontier = pareto_frontier([exact_long, exact_short, partial_short])
        self.assertEqual(
            {item.genome for item in frontier},
            {exact_short.genome, partial_short.genome},
        )

    def test_partial_cell_signal_guides_same_exact_score(self):
        weaker = Fitness(Genome(("weak",)), 0, 1, 1.0, 1, 4)
        stronger = Fitness(Genome(("strong",)), 0, 1, 1.0, 3, 4)
        frontier = pareto_frontier([weaker, stronger])
        self.assertEqual([item.genome for item in frontier], [stronger.genome])

    def test_partial_cell_match_penalizes_shape_mismatch(self):
        self.assertEqual(partial_cell_match([[1, 0]], [[1], [0]]), (1, 2))

    def test_probe_diversity_preserves_distinct_unlabeled_outputs(self):
        short = Fitness(Genome(("short",)), 1, 1, 1.0, 1, 1, (("x",),))
        long = Fitness(Genome(("long",)), 1, 1, 2.0, 1, 1, (("y",),))
        frontier = pareto_frontier([short, long])
        self.assertEqual({item.genome for item in frontier}, {short.genome, long.genome})

    def test_same_probe_signature_still_allows_mdl_dominance(self):
        short = Fitness(Genome(("short",)), 1, 1, 1.0, 1, 1, (("x",),))
        long = Fitness(Genome(("long",)), 1, 1, 2.0, 1, 1, (("x",),))
        frontier = pareto_frontier([short, long])
        self.assertEqual([item.genome for item in frontier], [short.genome])

    def test_lower_demo_fitness_does_not_survive_probe_difference(self):
        strong = Fitness(Genome(("strong",)), 1, 1, 0.5, 1, 1, (("x",),))
        weak = Fitness(Genome(("weak",)), 0, 1, 1.0, 0, 1, (("y",),))
        frontier = pareto_frontier([strong, weak])
        self.assertEqual([item.genome for item in frontier], [strong.genome])

    def test_bounded_frontier_preserves_probe_coverage(self):
        candidates = [
            Fitness(Genome(("a",)), 1, 1, 1.0, 1, 1, (("x",), ("u",))),
            Fitness(Genome(("b",)), 1, 1, 1.0, 1, 1, (("y",), ("u",))),
            Fitness(Genome(("c",)), 1, 1, 1.0, 1, 1, (("x",), ("v",))),
        ]
        frontier = bounded_frontier(candidates, max_items=2)
        self.assertEqual(len(frontier), 2)
        self.assertEqual(
            {item.probe_signature for item in frontier},
            {(("y",), ("u",)), (("x",), ("v",))},
        )

    def test_probe_inputs_are_unlabeled_and_recorded(self):
        fitness = evaluate_genome(
            Genome(("copy",)), [Demonstration(0, 0)],
            lambda item, value: value,
            probe_inputs=(1, 2),
        )
        self.assertEqual(fitness.probe_signature, (1, 2))

    def test_generation_can_recover_mutated_exact_program(self):
        parent = Genome(("copy",))
        result = evolve_generation(
            [parent], [Demonstration(0, 1)],
            lambda item, value: value + (1 if item.operations == ("inc",) else 0),
            ["copy", "inc"],
        )
        self.assertTrue(any(item.exact and item.genome.operations == ("inc",) for item in result))

    def test_generation_crossover_is_opt_in_and_bounded(self):
        result = evolve_generation(
            [Genome(("a",)), Genome(("b",))],
            [Demonstration(0, 1)],
            lambda item, value: value + (1 if item.operations == ("a", "b") else 0),
            ["a", "b"],
            max_steps=2,
            include_crossover=True,
            max_parent_pairs=1,
        )
        self.assertTrue(any(item.genome.operations == ("a", "b") for item in result))
