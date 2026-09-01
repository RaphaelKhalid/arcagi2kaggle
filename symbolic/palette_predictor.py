"""Symbolic prediction of the SET of colors in an ARC test output.

Same falsification discipline as size_predictor: a rule fires only when it
explains the palette of every demonstration output exactly, and the combined
predictor abstains whenever fired rules disagree.

Two products:
  * an exact palette prediction (predict_palette) — measured for exact-set
    precision;
  * a superset bound (palette_superset_bound) — a set guaranteed (as far as
    the demos can guarantee anything) to contain the true output palette,
    measured for containment rate. Useful for pruning even when the exact
    set is unknown.

Public API:
    predict_palette(train_pairs, test_input, config)
        -> (frozenset|None, rule_name, candidates)
    palette_superset_bound(train_pairs, test_input)
        -> (frozenset|None, bound_name)
    fit_palette_rules(train_pairs, config) -> list[FittedRule]
"""

from .grid_utils import palette
from .size_predictor import FittedRule  # same lightweight wrapper


# --------------------------------------------------------------------------
# Rule fitters: pairs -> predict(test_input) -> frozenset, or None if any
# demo falsifies the rule.
# --------------------------------------------------------------------------

def _fit_same_as_input(pairs):
    """(a) Output palette equals input palette on every demo."""
    if all(palette(i) == palette(o) for i, o in pairs):
        return lambda test: palette(test)
    return None


def _fit_add_remove(pairs):
    """(b) Output palette = input palette + constant set A - constant set B.

    A (colors added) and B (colors removed) must be identical across demos,
    with A disjoint from each input and B present in each input — otherwise
    the demo could not distinguish this rule from a sloppier one. The
    trivial A = B = empty case is excluded (same_as_input covers it).
    """
    added = {frozenset(palette(o) - palette(i)) for i, o in pairs}
    removed = {frozenset(palette(i) - palette(o)) for i, o in pairs}
    if len(added) != 1 or len(removed) != 1:
        return None
    add, rem = next(iter(added)), next(iter(removed))
    if not add and not rem:
        return None
    # Exactness check: +A -B must reproduce each output exactly (it does by
    # construction of set difference, but keep the assertion cheap and safe).
    for i, o in pairs:
        if (palette(i) - rem) | add != palette(o):
            return None
    return lambda test: frozenset((palette(test) - rem) | add)


def _fit_constant(pairs):
    """(c) All demo outputs share one palette; predict that palette."""
    pals = {palette(o) for _, o in pairs}
    if len(pals) != 1:
        return None
    pal = next(iter(pals))
    return lambda test: pal


def _fit_subset_of_input(pairs):
    """(d) Output palette is a subset of the input palette on every demo.

    Weaker: as an *exact* predictor it only fires usefully when combined
    rules agree; its main value is feeding the superset bound. Tracked
    separately and disabled in the exact battery by default.
    """
    if all(palette(o) <= palette(i) for i, o in pairs):
        return lambda test: palette(test)
    return None


_ALL_RULES = [
    ("pal_same_as_input", _fit_same_as_input),
    ("pal_add_remove", _fit_add_remove),
    ("pal_constant", _fit_constant),
    ("pal_subset_of_input", _fit_subset_of_input),
]

RULE_NAMES = [name for name, _ in _ALL_RULES]

# ---------------------------------------------------------------------------
# Configuration presets (see size_predictor for the methodology). BASELINE is
# the untightened battery; STRICT reflects what the failure inspection showed:
#   * pal_subset_of_input predicts "palette(test input)" from subset evidence
#     only — measurably imprecise as an exact rule, so it is excluded from
#     the exact battery (it still powers the superset bound);
#   * pal_constant needs >= 3 demos to stand alone — two coinciding demo
#     palettes are weak evidence.
# ---------------------------------------------------------------------------

BASELINE_CONFIG = {
    "name": "baseline",
    "disabled": frozenset(),
    "min_demos": {},
    "min_demos_solo": {},
}

STRICT_CONFIG = {
    "name": "strict",
    "disabled": frozenset({"pal_subset_of_input"}),
    "min_demos": {},
    "min_demos_solo": {"pal_constant": 3},
}

# PARANOID additionally requires pal_add_remove to have >= 3 demos when it
# is the sole evidence (2-demo-solo group measured 28/29 correct pooled).
PARANOID_CONFIG = {
    "name": "paranoid",
    "disabled": STRICT_CONFIG["disabled"],
    "min_demos": {},
    "min_demos_solo": {"pal_constant": 3, "pal_add_remove": 3},
}

DEFAULT_CONFIG = STRICT_CONFIG


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def fit_palette_rules(train_pairs, config=DEFAULT_CONFIG):
    """Fit the battery; return rules consistent with every demo."""
    n = len(train_pairs)
    fitted = []
    for name, fitter in _ALL_RULES:
        if name in config["disabled"]:
            continue
        if n < config["min_demos"].get(name, 1):
            continue
        fn = fitter(train_pairs)
        if fn is not None:
            fitted.append(FittedRule(name, fn))
    return fitted


def predict_palette(train_pairs, test_input, config=DEFAULT_CONFIG):
    """Predict the exact set of colors in the test output.

    Returns (frozenset | None, rule_name, candidates); abstains with
    'conflict' when fired rules disagree, 'no_rule' when nothing fired.
    """
    fitted = fit_palette_rules(train_pairs, config)
    candidates = []
    for rule in fitted:
        pal = rule._predict(test_input)  # palettes need no size validation
        if pal:
            candidates.append((rule.name, frozenset(pal)))

    n = len(train_pairs)
    strong = [(name, p) for name, p in candidates
              if n >= config["min_demos_solo"].get(name, 1)]
    if not strong:
        return None, "no_rule", candidates

    distinct = {p for _, p in candidates}
    if len(distinct) == 1:
        return strong[0][1], strong[0][0], candidates
    return None, "conflict", candidates


def palette_superset_bound(train_pairs, test_input):
    """A set guaranteed to contain the output palette (by demo evidence).

    Construction: let A = union over demos of (output colors absent from the
    input). Every demo output satisfies out <= in | A, so the bound
    palette(test) | A always fires. When outputs were subsets of inputs on
    every demo (A empty), the bound tightens to palette(test) alone.

    Returns (bound, name): name is 'input_palette' when A is empty, else
    'input_plus_added'.
    """
    extra = frozenset().union(
        *[palette(o) - palette(i) for i, o in train_pairs])
    bound = frozenset(palette(test_input) | extra)
    name = "input_palette" if not extra else "input_plus_added"
    return bound, name
