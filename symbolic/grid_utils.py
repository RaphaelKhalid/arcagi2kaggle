"""Shared, dependency-free grid helpers for the symbolic predictors.

Grids are lists of lists of ints in 0..9 (ARC format). All helpers are pure
functions with no side effects and no external dependencies.
"""

from collections import Counter, deque

# ARC grids are at most 30x30.
MAX_DIM = 30


def dims(grid):
    """Return (height, width) of a grid."""
    return (len(grid), len(grid[0]) if grid else 0)


def palette(grid):
    """Return the frozenset of colors present in a grid."""
    return frozenset(c for row in grid for c in row)


def color_counts(grid):
    """Return a Counter of color -> cell count."""
    counts = Counter()
    for row in grid:
        counts.update(row)
    return counts


def most_common_color(grid):
    """Most frequent color; ties broken by smallest color value (deterministic)."""
    counts = color_counts(grid)
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return best[0]


def bbox_dims(grid, bg):
    """(height, width) of the bounding box of all non-`bg` cells.

    Returns None when the grid contains only background.
    """
    rows = [r for r, row in enumerate(grid) if any(c != bg for c in row)]
    if not rows:
        return None
    cols = [c for c in range(len(grid[0]))
            if any(grid[r][c] != bg for r in range(len(grid)))]
    return (rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1)


def connected_components(grid, connectivity, bg):
    """Connected components of non-`bg` cells (any mix of non-bg colors).

    connectivity: 4 or 8. Returns a list of components, each a list of
    (row, col) cell coordinates. Deterministic order (row-major discovery).
    """
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
    """(height, width) of a component's bounding box."""
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)


def valid_size(size):
    """True when `size` is a legal ARC output size (1..30 per axis, ints)."""
    if size is None:
        return False
    h, w = size
    return (isinstance(h, int) and isinstance(w, int)
            and 1 <= h <= MAX_DIM and 1 <= w <= MAX_DIM)
