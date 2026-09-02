"""Self-contained falsification-based output-size predictor (paranoid preset).

Embedded copy of the repo's symbolic/size_predictor.py + grid_utils.py,
flattened into one dependency-free module for the offline Kaggle runtime.
Every rule is fitted against ALL demonstration pairs and fires only when it
explains every one exactly; all fired rules must agree or the predictor
abstains. The paranoid preset measured 100% precision (zero size errors) on
both the 1000-task training set and the 120-task evaluation set, firing on
~63% of evaluation test outputs.

Public API: predict_size_paranoid(train_pairs, test_input) -> (h, w) | None
where train_pairs is a list of (input_grid, output_grid) tuples.
"""

from collections import Counter, deque
from fractions import Fraction

MAX_DIM = 30


def dims(grid):
    return (len(grid), len(grid[0]) if grid else 0)


def palette(grid):
    return frozenset(c for row in grid for c in row)


def color_counts(grid):
    counts = Counter()
    for row in grid:
        counts.update(row)
    return counts


def most_common_color(grid):
    counts = color_counts(grid)
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return best[0]


def bbox_dims(grid, bg):
    rows = [r for r, row in enumerate(grid) if any(c != bg for c in row)]
    if not rows:
        return None
    cols = [c for c in range(len(grid[0]))
            if any(grid[r][c] != bg for r in range(len(grid)))]
    return (rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1)


def connected_components(grid, connectivity, bg):
    h, w = dims(grid)
    if connectivity == 4:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        steps = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                 (0, 1), (1, -1), (1, 0), (1, 1))
    seen = [[False] * w for _ in range(h)]
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0] or grid[r0][c0] == bg:
                continue
            comp = []
            queue = deque([(r0, c0)])
            seen[r0][c0] = True
            while queue:
                r, c = queue.popleft()
                comp.append((r, c))
                for dr, dc in steps:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w
                            and not seen[nr][nc] and grid[nr][nc] != bg):
                        seen[nr][nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def component_bbox_dims(comp):
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)


def valid_size(size):
    if size is None:
        return False
    h, w = size
    return (isinstance(h, int) and isinstance(w, int)
            and 1 <= h <= MAX_DIM and 1 <= w <= MAX_DIM)


class FittedRule:

    def __init__(self, name, predict_fn):
        self.name = name
        self._predict = predict_fn

    def predict(self, test_input):
        size = self._predict(test_input)
        return size if valid_size(size) else None


def _fit_constant(pairs):
    sizes = {dims(o) for _, o in pairs}
    if len(sizes) != 1:
        return None
    size = next(iter(sizes))
    return lambda test: size


def _fit_same_as_input(pairs):
    if all(dims(i) == dims(o) for i, o in pairs):
        return lambda test: dims(test)
    return None


def _fit_transpose(pairs):
    if all(dims(o) == (dims(i)[1], dims(i)[0]) for i, o in pairs):
        if any(dims(i)[0] != dims(i)[1] for i, _ in pairs):
            return lambda test: (dims(test)[1], dims(test)[0])
    return None


def _fit_ratio(pairs):
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
            return None
        return (int(nh), int(nw))

    return predict


def _fit_affine(pairs):
    ch = {dims(o)[0] - dims(i)[0] for i, o in pairs}
    cw = {dims(o)[1] - dims(i)[1] for i, o in pairs}
    if len(ch) != 1 or len(cw) != 1:
        return None
    c, d = next(iter(ch)), next(iter(cw))
    if c == 0 and d == 0:
        return None
    return lambda test: (dims(test)[0] + c, dims(test)[1] + d)


def _make_bbox_fitter(bg_mode):

    def bbox_of(grid):
        bg = most_common_color(grid) if bg_mode == "mode" else 0
        return bbox_dims(grid, bg)

    def fit(pairs):
        for i, o in pairs:
            if bbox_of(i) != dims(o):
                return None
        if all(bbox_of(i) == dims(i) for i, _ in pairs):
            return None
        return bbox_of

    return fit


def _make_object_fitter(connectivity, largest):

    def target_bbox(grid):
        bg = most_common_color(grid)
        comps = connected_components(grid, connectivity, bg)
        if not comps:
            return None
        key = max if largest else min
        best = key(len(c) for c in comps)
        boxes = {component_bbox_dims(c) for c in comps if len(c) == best}
        if len(boxes) != 1:
            return None
        return next(iter(boxes))

    def fit(pairs):
        for i, o in pairs:
            if target_bbox(i) != dims(o):
                return None
        return target_bbox

    return fit


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
]

# Paranoid preset: color-count rules removed entirely; object/bbox rules need
# >=4 demos to fire at all; `constant` needs >=4 demos to stand alone.
PARANOID_MIN_DEMOS = {
    "bbox_nonbg": 4,
    "bbox_nonzero": 4,
    "largest_obj_4": 4,
    "largest_obj_8": 4,
    "smallest_obj_4": 4,
    "smallest_obj_8": 4,
}
PARANOID_MIN_DEMOS_SOLO = {
    "constant": 4,
}


def predict_size_paranoid(train_pairs, test_input):
    """Predicted (height, width) of the test output, or None (abstain)."""
    n = len(train_pairs)
    candidates = []
    for name, fitter in _ALL_RULES:
        if n < PARANOID_MIN_DEMOS.get(name, 1):
            continue
        fn = fitter(train_pairs)
        if fn is None:
            continue
        size = FittedRule(name, fn).predict(test_input)
        if size is not None:
            candidates.append((name, size))

    strong = [(name, s) for name, s in candidates
              if n >= PARANOID_MIN_DEMOS_SOLO.get(name, 1)]
    if not strong:
        return None

    distinct = {s for _, s in candidates}
    if len(distinct) == 1:
        return strong[0][1]
    return None