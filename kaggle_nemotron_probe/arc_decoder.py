import os
import bz2
import pickle
from math import exp, log
import numpy as np


def _log_mean_exp(values):
    """Stable log(mean(exp(values))) for a non-empty finite iterable."""
    values = [float(value) for value in values]
    if not values:
        raise ValueError("log mean exp requires at least one value")
    pivot = max(values)
    return pivot + log(sum(exp(value - pivot) for value in values) / len(values))

def hashable(guess):
    return tuple(map(tuple, guess))

def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    # Filesystem/sample insertion order is not a semantic tie-breaker.  Use a
    # canonical grid tuple so replaying the same cache is deterministic.
    scores = sorted(scores, key=lambda x: (-x[0], hashable(x[1])))
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs

def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline-g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline-s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score

def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)

def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score

def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


def getter_log_evidence(guesses):
    """Rank one exact-output class by conservative nuisance-marginal evidence.

    View transforms and repeated samples are treated as exchangeable evidence,
    so each mean is taken in probability space before combining the beam and
    augmentation likelihood terms.  This avoids raw support-count inflation.
    It is an ablation only: the model scores are not calibrated probabilities.
    """
    if not guesses:
        return float("-inf")
    beam_log_marginal = _log_mean_exp(
        -float(guess["beam_score"]) for guess in guesses
    )
    augmentation_log_marginals = (
        _log_mean_exp(-float(score) for score in guess["score_aug"])
        for guess in guesses
    )
    return beam_log_marginal + _log_mean_exp(augmentation_log_marginals)


def score_log_evidence(guesses):
    """Return exact-output classes ranked by opt-in log evidence."""
    return score_sum(guesses, getter_log_evidence)


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
    score_log_evidence,
]


class ArcDecoder:
    
    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        for key in os.listdir(store):
            with bz2.BZ2File(os.path.join(store, key)) as f:
                outputs = pickle.load(f)
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")

        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0

        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():

            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)

            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():

                solution = sample["solution"]
                beam_score = sample["beam_score"]
                aug_mean = np.mean(sample["score_aug"])

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"

                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
        print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")

        num_puzzles = len(num_tasks_per_puzzle)

        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {k for k, v in selected.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
            print(correct_puzzles)
            score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")
