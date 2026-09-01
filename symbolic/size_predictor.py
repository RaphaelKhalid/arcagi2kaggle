"""Symbolic prediction of ARC test-output grid dimensions.

Every rule is *falsifiable*: it is fitted against ALL demonstration pairs and
fires only when it explains every one of them exactly. At combination time,
all fired rules must agree on the test prediction; any disagreement makes the
predictor abstain (returning the conflicting candidates for inspection).

Precision is prioritised over coverage: these predictions later prune a
neural solver's decode search space, so a wrong confident answer is far more
costly than an abstention.

Public API:
    predict_size(train_pairs, test_input, config) -> (size|None, rule_name, candidates)
    fit_size_rules(train_pairs, config)           -> list[FittedRule]
"""

from fractions import Fraction

from .grid_utils import (bbox_dims, component_bbox_dims, connected_components,
                         dims, most_common_color, palette, valid_size)


class FittedRule:
    """A rule that survived falsification against all train pairs."""

    def __init__(self, name, predict_fn):
        self.name = name
        self._predict = predict_fn

    def predict(self, test_input):
        """Predicted (h, w) for the test input, or None when inapplicable."""
        size = self._predict(test_input)
        return size if valid_size(size) else None


# --------------------------------------------------------------------------
# Individual rule fitters. Each takes the list of (input, output) grid pairs
# and returns a predict function, or None when any demo falsifies the rule.
# --------------------------------------------------------------------------

def _fit_constant(pairs):
    """(a) All demo outputs share one size; predict that size."""
    sizes = {dims(o) for _, o in pairs}
    if len(sizes) != 1:
        return None
    size = next(iter(sizes))
    return lambda test: size


def _fit_same_as_input(pairs):
    """(b) Output size equals input size on every demo."""
    if all(dims(i) == dims(o) for i, o in pairs):
        return lambda test: dims(test)
    return None


def _fit_transpose(pairs):
    """(d) Output size is the transposed input size on every demo.

    Excluded when every demo input is square (indistinguishable from
    same-as-input, which already covers that case).
    """
    if all(dims(o) == (dims(i)[1], dims(i)[0]) for i, o in pairs):
        if any(dims(i)[0] != dims(i)[1] for i, _ in pairs):
            return lambda test: (dims(test)[1], dims(test)[0])
    return None


def _fit_ratio(pairs):
    """(c) Fixed multiplicative ratio per axis, exact rationals.

    Covers h_out = k*h_in and h_out = h_in/k (any exact rational). The
    trivial ratio (1, 1) is excluded — that is same_as_input's job.
    """
    rh = {Fraction(dims(o)[0], dims(i)[0]) for i, o in pairs}
    rw = {Fraction(dims(o)[1], dims(i)[1]) for i, o in pairs}
    if len(rh) != 1 or len(rw) != 1:
        return None
    fh, fw = next(iter(rh)), next(iter(rw))
    if fh == 1 and fw == 1:
        return None

    def predict(test):
        h, w = dims(test)
        nh, nw = Fraction(h) * fh, Fraction(w) * fw
        if nh.denominator != 1 or nw.denominator != 1:
            return None  # ratio does not divide the test input exactly
        return (int(nh), int(nw))

    return predict


def _fit_affine(pairs):
    """(g) h_out = h_in + c, w_out = w_in + d with constant integers c, d.

    The trivial offset (0, 0) is excluded (same_as_input covers it).
    """
    ch = {dims(o)[0] - dims(i)[0] for i, o in pairs}
    cw = {dims(o)[1] - dims(i)[1] for i, o in pairs}
    if len(ch) != 1 or len(cw) != 1:
        return None
    c, d = next(iter(ch)), next(iter(cw))
    if c == 0 and d == 0:
        return None
    return lambda test: (dims(test)[0] + c, dims(test)[1] + d)


def _make_bbox_fitter(bg_mode):
    """(e) Output size equals the bounding box of non-background cells.

    bg_mode: 'mode' (background = most common color) or 'zero' (bg = 0).
    """

    def bbox_of(grid):
        bg = most_common_color(grid) if bg_mode == "mode" else 0
        return bbox_dims(grid, bg)

    def fit(pairs):
        for i, o in pairs:
            if bbox_of(i) != dims(o):
                return None
        # Require the rule to be informative on at least one demo: if the
        # bbox always coincides with the full input, same_as_input already
        # covers it and this rule adds only coincidence risk.
        if all(bbox_of(i) == dims(i) for i, _ in pairs):
            return None
        return bbox_of

    return fit


def _make_object_fitter(connectivity, largest):
    """(f) Output size equals the bounding box of the largest/smallest
    connected non-background object (background = most common color).

    Object size is measured in cell count; ties are falsifying only if the
    tied objects have different bounding boxes (we pick deterministically by
    (count, -row, -col) but require agreement to be safe).
    """

    def target_bbox(grid):
        bg = most_common_color(grid)
        comps = connected_components(grid, connectivity, bg)
        if not comps:
            return None
        key = max if largest else min
        best = key(len(c) for c in comps)
        boxes = {component_bbox_dims(c) for c in comps if len(c) == best}
        if len(boxes) != 1:
            return None  # ambiguous tie -> refuse to guess
        return next(iter(boxes))

    def fit(pairs):
        for i, o in pairs:
            if target_bbox(i) != dims(o):
                return None
        return target_bbox

    return fit


def _make_color_count_fitter(mode):
    """(h) Output is an n x n square where n = number of distinct colors.

    mode: 'all' counts every color, 'excl_bg' excludes the most common
    color, 'excl_zero' excludes color 0. Only fires when exactly consistent
    with every demo.
    """

    def n_colors(grid):
        pal = set(palette(grid))
        if mode == "excl_bg":
            pal.discard(most_common_color(grid))
        elif mode == "excl_zero":
            pal.discard(0)
        return len(pal)

    def fit(pairs):
        for i, o in pairs:
            n = n_colors(i)
            if dims(o) != (n, n):
                return None
        return lambda test: (n_colors(test),) * 2

    return fit


# Ordered rule battery: (name, fitter). Order sets reporting priority only —
# combination requires unanimity, so order never changes a prediction.
_ALL_RULES = [
    ("same_as_input", _fit_same_as_input),
    ("constant", _fit_constant),
    ("ratio", _fit_ratio),
    ("affine_offset", _fit_affine),
    ("transpose", _fit_transpose),
    ("bbox_nonbg", _make_bbox_fitter("mode")),
    ("bbox_nonzero", _make_bbox_fitter("zero")),
    ("largest_obj_4", _make_object_fitter(4, True)),
    ("largest_obj_8", _make_object_fitter(8, True)),
    ("smallest_obj_4", _make_object_fitter(4, False)),
    ("smallest_obj_8", _make_object_fitter(8, False)),
    ("color_count_all", _make_color_count_fitter("all")),
    ("color_count_excl_bg", _make_color_count_fitter("excl_bg")),
    ("color_count_excl_zero", _make_color_count_fitter("excl_zero")),
]

RULE_NAMES = [name for name, _ in _ALL_RULES]

# ---------------------------------------------------------------------------
# Configuration presets.
#
# BASELINE is the untightened battery (measured first). STRICT is the result
# of inspecting BASELINE's failures and tightening:
#   * per-rule minimum demo counts for the coincidence-prone rules — a weak
#     rule that "explains" only two demos is often just luck;
#   * color-count rules disabled entirely (near-coin-flip precision);
#   * constant requires >= 3 demos when it is the ONLY evidence (enforced via
#     min_demos_solo: with fewer demos, constant may still fire alongside
#     other rules — unanimity checking — but cannot be the sole basis).
# ---------------------------------------------------------------------------

BASELINE_CONFIG = {
    "name": "baseline",
    "disabled": frozenset(),
    "min_demos": {},          # rule -> min number of train pairs to fire
    "min_demos_solo": {},     # rule -> min pairs to fire *alone*
}

STRICT_CONFIG = {
    "name": "strict",
    "disabled": frozenset({
        "color_count_all", "color_count_excl_bg", "color_count_excl_zero",
    }),
    "min_demos": {
        "bbox_nonbg": 3,
        "bbox_nonzero": 3,
        "largest_obj_4": 3,
        "largest_obj_8": 3,
        "smallest_obj_4": 3,
        "smallest_obj_8": 3,
    },
    "min_demos_solo": {
        "constant": 3,
    },
}

# PARANOID cuts the two evidence groups that remained under 0.98 precision
# after STRICT (constant firing solo on 3 demos: 13/14 correct pooled over
# both datasets +1 more wrong; object/bbox rules as sole evidence on 3 demos:
# 17/18 correct pooled). Zero measured size errors on both datasets, at the
# cost of ~1.5-3pp coverage.
PARANOID_CONFIG = {
    "name": "paranoid",
    "disabled": STRICT_CONFIG["disabled"],
    "min_demos": {
        "bbox_nonbg": 4,
        "bbox_nonzero": 4,
        "largest_obj_4": 4,
        "largest_obj_8": 4,
        "smallest_obj_4": 4,
        "smallest_obj_8": 4,
    },
    "min_demos_solo": {
        "constant": 4,
    },
}

DEFAULT_CONFIG = STRICT_CONFIG


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def fit_size_rules(train_pairs, config=DEFAULT_CONFIG):
    """Fit the whole battery; return the rules consistent with every demo."""
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


def predict_size(train_pairs, test_input, config=DEFAULT_CONFIG):
    """Predict the test output's (height, width).

    Returns (prediction | None, rule_name, candidates) where candidates is
    the list of (rule_name, size) for every rule that fired. rule_name is
    the first fired rule on agreement, 'conflict' on disagreement, and
    'no_rule' when nothing fired.
    """
    fitted = fit_size_rules(train_pairs, config)
    candidates = []
    for rule in fitted:
        size = rule.predict(test_input)
        if size is not None:
            candidates.append((rule.name, size))

    # Solo-firing restriction (see STRICT_CONFIG): drop predictions whose
    # only support is a rule that needs more demos to stand alone.
    n = len(train_pairs)
    strong = [(name, s) for name, s in candidates
              if n >= config["min_demos_solo"].get(name, 1)]
    if not strong:
        return None, "no_rule", candidates

    distinct = {s for _, s in candidates}
    if len(distinct) == 1:
        return strong[0][1], strong[0][0], candidates
    return None, "conflict", candidates
