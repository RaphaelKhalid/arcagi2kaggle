"""Measurement runner for the symbolic output-property predictors.

For every test output with known ground truth in the ARC-AGI-2 training
(1000 tasks) and evaluation (120 tasks) sets, measures per rule and overall:

  * coverage  — fraction of test outputs where a prediction was made;
  * precision — fraction of those predictions that are exactly correct.

Also reports the combined "predict when any rule fires, abstain on conflict"
policy, the palette superset bound's containment rate, and a sample of
failures (task ids plus the rule that lied) for tightening iterations.

Usage (from the project root):
    python -m symbolic.measure                 # strict (default) config
    python -m symbolic.measure --preset baseline
    python -m symbolic.measure --failures 20   # show more failure samples
"""

import argparse
import json
import os
from collections import defaultdict

from . import palette_predictor as pp
from . import size_predictor as sp
from .grid_utils import palette

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "raw")

DATASETS = {
    "training": ("arc-agi_training_challenges.json",
                 "arc-agi_training_solutions.json"),
    "evaluation": ("arc-agi_evaluation_challenges.json",
                   "arc-agi_evaluation_solutions.json"),
}


def load_dataset(name):
    """Yield (task_id, train_pairs, test_input, true_output) test cases."""
    ch_file, sol_file = DATASETS[name]
    with open(os.path.join(DATA_DIR, ch_file)) as f:
        challenges = json.load(f)
    with open(os.path.join(DATA_DIR, sol_file)) as f:
        solutions = json.load(f)
    cases = []
    for task_id, task in challenges.items():
        pairs = [(p["input"], p["output"]) for p in task["train"]]
        for idx, test in enumerate(task["test"]):
            cases.append((task_id, pairs, test["input"],
                          solutions[task_id][idx]))
    return cases


class Tally:
    """Fired/correct counts for one (rule or combined) predictor."""

    def __init__(self):
        self.fired = 0
        self.correct = 0
        self.failures = []  # (task_id, predicted, truth)

    def record(self, task_id, predicted, truth):
        self.fired += 1
        if predicted == truth:
            self.correct += 1
        else:
            self.failures.append((task_id, predicted, truth))

    @property
    def precision(self):
        return self.correct / self.fired if self.fired else float("nan")


def fmt_pal(pal):
    return "{" + ",".join(str(c) for c in sorted(pal)) + "}"


def measure_dataset(name, cases, size_cfg, pal_cfg):
    """Return {(predictor, rule): Tally} plus the number of test cases."""
    tallies = defaultdict(Tally)
    for task_id, pairs, test_in, true_out in cases:
        true_size = (len(true_out), len(true_out[0]))
        true_pal = palette(true_out)

        # ---- size: per-rule ----
        for rule in sp.fit_size_rules(pairs, size_cfg):
            pred = rule.predict(test_in)
            if pred is not None:
                tallies[("size", rule.name)].record(task_id, pred, true_size)
        # ---- size: combined ----
        pred, why, _ = sp.predict_size(pairs, test_in, size_cfg)
        if pred is not None:
            tallies[("size", "COMBINED")].record(task_id, pred, true_size)

        # ---- palette: per-rule ----
        for rule in pp.fit_palette_rules(pairs, pal_cfg):
            ppred = rule._predict(test_in)
            if ppred:
                tallies[("palette", rule.name)].record(
                    task_id, frozenset(ppred), true_pal)
        # ---- palette: combined ----
        ppred, why, _ = pp.predict_palette(pairs, test_in, pal_cfg)
        if ppred is not None:
            tallies[("palette", "COMBINED")].record(task_id, ppred, true_pal)

        # ---- palette: superset bound (measured for containment) ----
        bound, bname = pp.palette_superset_bound(pairs, test_in)
        contained = true_pal <= bound
        t = tallies[("pal_bound", bname)]
        t.fired += 1
        if contained:
            t.correct += 1
        else:
            t.failures.append((task_id, frozenset(bound), true_pal))
        tallies[("pal_bound", "ANY")].fired += 1
        tallies[("pal_bound", "ANY")].correct += int(contained)

    return tallies


def print_table(name, n_cases, tallies, n_failures):
    header = f"{'dataset':<11} {'predictor':<9} {'rule':<24} " \
             f"{'n_fired':>7} {'coverage':>8} {'precision':>9}"
    print(header)
    print("-" * len(header))

    def rows(predictor, rule_order):
        for rule in rule_order:
            t = tallies.get((predictor, rule))
            if t is None or t.fired == 0:
                continue
            cov = t.fired / n_cases
            print(f"{name:<11} {predictor:<9} {rule:<24} "
                  f"{t.fired:>7} {cov:>8.3f} {t.precision:>9.4f}")

    rows("size", sp.RULE_NAMES + ["COMBINED"])
    print()
    rows("palette", pp.RULE_NAMES + ["COMBINED"])
    print()
    rows("pal_bound", ["input_palette", "input_plus_added", "ANY"])
    print()

    for predictor in ("size", "palette"):
        t = tallies.get((predictor, "COMBINED"))
        if t and t.failures:
            shown = t.failures[:n_failures]
            print(f"{name} {predictor} COMBINED failures "
                  f"({len(t.failures)} total, showing {len(shown)}):")
            for task_id, pred, truth in shown:
                if predictor == "palette":
                    pred, truth = fmt_pal(pred), fmt_pal(truth)
                print(f"  {task_id}: predicted {pred}, truth {truth}")
            print()
    t = tallies.get(("pal_bound", "ANY"))
    if t and t.fired > t.correct:
        bad = [f for key in (("pal_bound", "input_palette"),
                             ("pal_bound", "input_plus_added"))
               for f in tallies[key].failures][:n_failures]
        print(f"{name} palette superset-bound violations: {len(bad)} shown:")
        for task_id, pred, truth in bad:
            print(f"  {task_id}: bound {fmt_pal(pred)}, truth {fmt_pal(truth)}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=["baseline", "strict", "paranoid"],
                    default="strict")
    ap.add_argument("--failures", type=int, default=8,
                    help="max failure samples to print per predictor")
    args = ap.parse_args()

    size_cfg = {"baseline": sp.BASELINE_CONFIG, "strict": sp.STRICT_CONFIG,
                "paranoid": sp.PARANOID_CONFIG}[args.preset]
    pal_cfg = {"baseline": pp.BASELINE_CONFIG, "strict": pp.STRICT_CONFIG,
               "paranoid": pp.PARANOID_CONFIG}[args.preset]
    print(f"preset: {args.preset}\n")

    for name in DATASETS:
        cases = load_dataset(name)
        tallies = measure_dataset(name, cases, size_cfg, pal_cfg)
        print(f"=== {name}: {len(cases)} test outputs ===")
        print_table(name, len(cases), tallies, args.failures)


if __name__ == "__main__":
    main()
