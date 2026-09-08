#!/usr/bin/env python3
# =============================================================================
#  ARC-AGI-2  Leg C : Verified-Induction Pre-Pass  (v2, base-notebook-aligned)
#  -------------------------------------------------------------
#  Role      : runs BEFORE the base 4B TTT pipeline (failed-in-aimo line).
#              Samples DSL programs with a 7B coder (one per L4, 4 workers),
#              verifies them against ALL train pairs in an isolated,
#              torch-free sandbox, and emits train-verified test outputs.
#              A separate merge cell then lets verified outputs pre-empt
#              attempt_1 while attempt_2 keeps the base (transduction)
#              candidate -> monotonic: final score can only add coverage.
#  Env facts : Kaggle AGI-2 = Python 3.11, L4 22GB x4, offline.
#              -> NO vLLM / cp312 wheelhouse dependency (27B v4 incident).
#              -> 7B bf16 fits a single L4; 4 independent workers.
#              -> heavy imports live INSIDE worker fns (spawn re-import cheap).
#  Fixes vs the "hacker plan":
#     1. exec single-namespace (two-dict exec made the DSL unreachable)
#     2. `import re` present; no undefined names in the submit loop
#     3. DSL is ENFORCED via AST whitelist, not politely requested
#     4. 35-primitive library instead of 3; per-program SIGALRM timeout
#     5. batched sampling (K per call) + 1 repair round, not 3 blind shots
#     6. official 2026 submission format {tid: [{attempt_1, attempt_2}, ..]}
#  Usage (Kaggle):
#     !python arc_induction_v1.py --budget-h 1.5
#     !python arc_induction_v1.py --selftest        # sandbox unit tests, no GPU
#     !python arc_induction_v1.py --smoke 4         # 4 easiest tasks, verbose
# =============================================================================

import os
import sys
import re
import json
import time
import math
import random
import hashlib
import argparse
import subprocess
import multiprocessing as mp

# ----------------------------------------------------------------------------
# Paths / defaults
# ----------------------------------------------------------------------------
DATA_DIR_CANDIDATES = [
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2",  # path used by the LB-proven base notebook
    "/kaggle/input/arc-prize-2026-arc-agi-2",               # standard mount fallback
]
WORK_DIR      = os.environ.get("ARC_WORK_DIR", "/kaggle/working")
RUNTIME_PATH  = os.path.join(WORK_DIR, "_dsl_runtime.py")
RESULTS_PATH  = os.path.join(WORK_DIR, "induction_results.json")
AUDIT_PATH    = os.path.join(WORK_DIR, "induction_audit.json")
SUB_PATH      = os.path.join(WORK_DIR, "submission_induction_only.json")
PARTIAL_TPL   = os.path.join(WORK_DIR, "_legc_partial_{rank}.jsonl")

MODEL_PATH_CANDIDATES = [
    "/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/7b-instruct/1",  # user's attach path
    "/kaggle/input/qwen-lm/qwen2.5-coder/transformers/7b-instruct/1",
    "/kaggle/input/qwen2.5-coder-7b-instruct/transformers/default/1",
    "/kaggle/input/qwen2.5-coder/transformers/7b-instruct/1",
    "/kaggle/input/qwen2.5-coder-7b-instruct",
]

DEFAULTS = dict(
    budget_h        = 1.5,    # wall budget of the whole induction phase
    per_task_cap    = 200.0,  # seconds per task (both rounds included)
    k1              = 8,      # round-1 samples per task
    k2              = 6,      # repair-round samples
    temp1           = 0.8,
    temp2           = 0.7,
    max_new_tokens  = 640,
    max_prompt_chars= 26000,
    prog_timeout    = 4.0,    # per-candidate CPU seconds in the sandbox
    seed            = 1234,
    print_grids     = 8,      # dump grids for first N verified tasks
)

# ----------------------------------------------------------------------------
# The sandbox runtime (torch-free).  Written to _dsl_runtime.py and executed
# as a subprocess:  python _dsl_runtime.py in.json out.json
# Candidate code is exec'd into ONE namespace that already CONTAINS the DSL
# functions as values -> transform.__globals__ sees them.  (This is the
# structural fix for the two-dict exec bug.)
# ----------------------------------------------------------------------------
RUNTIME_CODE = r'''
import sys, json, ast, signal, builtins
import numpy as np
from scipy.ndimage import label as _cc_label

MAXD = 30
_ST4 = np.array([[0,1,0],[1,1,1],[0,1,0]])
_ST8 = np.ones((3,3), dtype=int)

def _np(g): return np.asarray(g, dtype=int)
def _ls(a): return [[int(x) for x in row] for row in np.asarray(a)]

# ---------------- grid basics ----------------
def grid_shape(g):
    a = _np(g); return (int(a.shape[0]), int(a.shape[1]))
def new_grid(h, w, fill=0):
    return [[int(fill)] * int(w) for _ in range(int(h))]
def rot90(g):  return _ls(np.rot90(_np(g), 3))     # clockwise
def rot180(g): return _ls(np.rot90(_np(g), 2))
def rot270(g): return _ls(np.rot90(_np(g), 1))     # counter-clockwise
def flip_h(g): return _ls(np.fliplr(_np(g)))       # mirror left-right
def flip_v(g): return _ls(np.flipud(_np(g)))       # mirror up-down
def transpose(g): return _ls(_np(g).T)
def crop(g, r, c, h, w):
    return _ls(_np(g)[int(r):int(r)+int(h), int(c):int(c)+int(w)])

# ---------------- colors ----------------
def bg_color(g):
    v, n = np.unique(_np(g), return_counts=True); return int(v[np.argmax(n)])
def palette(g): return [int(x) for x in np.unique(_np(g))]
def color_counts(g):
    v, n = np.unique(_np(g), return_counts=True)
    return {int(a): int(b) for a, b in zip(v, n)}
def replace_color(g, a, b):
    arr = _np(g).copy(); arr[arr == int(a)] = int(b); return _ls(arr)
def swap_colors(g, a, b):
    arr = _np(g).copy(); ma = arr == int(a); mb = arr == int(b)
    arr[ma] = int(b); arr[mb] = int(a); return _ls(arr)
def map_colors(g, mapping):
    arr = _np(g); out = arr.copy()
    for k, v in mapping.items(): out[arr == int(k)] = int(v)
    return _ls(out)
def find_color_cells(g, color):
    ys, xs = np.where(_np(g) == int(color))
    return [(int(r), int(c)) for r, c in zip(ys, xs)]
def crop_to_content(g, bg=None):
    a = _np(g); b = bg_color(g) if bg is None else int(bg)
    ys, xs = np.where(a != b)
    if len(ys) == 0: return _ls(a)
    return _ls(a[ys.min():ys.max()+1, xs.min():xs.max()+1])

# ---------------- assembly ----------------
def paste(dst, src, r, c, transparent=None):
    D = _np(dst).copy(); S = _np(src)
    r0, c0 = int(r), int(c); h, w = S.shape
    r1, c1 = max(r0, 0), max(c0, 0)
    r2, c2 = min(r0 + h, D.shape[0]), min(c0 + w, D.shape[1])
    if r2 <= r1 or c2 <= c1: return _ls(D)
    sub = S[r1-r0:r2-r0, c1-c0:c2-c0]
    if transparent is None:
        D[r1:r2, c1:c2] = sub
    else:
        m = sub != int(transparent)
        D[r1:r2, c1:c2][m] = sub[m]
    return _ls(D)
def concat_h(a, b): return _ls(np.concatenate([_np(a), _np(b)], axis=1))
def concat_v(a, b): return _ls(np.concatenate([_np(a), _np(b)], axis=0))
def tile(g, ny, nx): return _ls(np.tile(_np(g), (int(ny), int(nx))))
def upscale(g, fy, fx):
    return _ls(np.kron(_np(g), np.ones((int(fy), int(fx)), dtype=int)))
def downscale(g, fy, fx):
    a = _np(g); fy, fx = int(fy), int(fx)
    h, w = a.shape[0] // fy, a.shape[1] // fx
    out = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            blk = a[i*fy:(i+1)*fy, j*fx:(j+1)*fx].ravel()
            v, n = np.unique(blk, return_counts=True)
            out[i, j] = v[np.argmax(n)]
    return _ls(out)
def cellwise(a, b, f):
    A, B = _np(a), _np(b)
    if A.shape != B.shape: raise ValueError("cellwise: shape mismatch")
    return [[int(f(int(A[i, j]), int(B[i, j]))) for j in range(A.shape[1])]
            for i in range(A.shape[0])]
def grid_eq(a, b):
    A, B = _np(a), _np(b)
    return A.shape == B.shape and bool((A == B).all())
def split_rows(g, n):
    a = _np(g); h = a.shape[0] // int(n)
    return [_ls(a[i*h:(i+1)*h, :]) for i in range(int(n))]
def split_cols(g, n):
    a = _np(g); w = a.shape[1] // int(n)
    return [_ls(a[:, i*w:(i+1)*w]) for i in range(int(n))]
def split_by_separator(g):
    # splits on full uniform rows/cols of the dominant separator color;
    # returns row-major list of sub-grids (or [g] when no separator found)
    a = _np(g); H, W = a.shape
    frows = [i for i in range(H) if len(set(a[i, :].tolist())) == 1]
    fcols = [j for j in range(W) if len(set(a[:, j].tolist())) == 1]
    colors = [int(a[i, 0]) for i in frows] + [int(a[0, j]) for j in fcols]
    if not colors: return [_ls(a)]
    v, n = np.unique(np.array(colors), return_counts=True)
    sep = int(v[np.argmax(n)])
    rs = set(i for i in frows if a[i, 0] == sep)
    cs = set(j for j in fcols if a[0, j] == sep)
    def runs(banned, N):
        out, s = [], None
        for i in range(N):
            if i in banned:
                if s is not None: out.append((s, i)); s = None
            else:
                if s is None: s = i
        if s is not None: out.append((s, N))
        return out
    parts = []
    for (r0, r1) in runs(rs, H):
        for (c0, c1) in runs(cs, W):
            parts.append(_ls(a[r0:r1, c0:c1]))
    return parts if len(parts) > 1 else [_ls(a)]

# ---------------- objects ----------------
def _mk_obj(cells):
    rs  = [r for r, _, _ in cells]; csx = [c for _, c, _ in cells]
    col = [v for _, _, v in cells]
    v, n = np.unique(np.array(col), return_counts=True)
    return {'cells': cells,
            'r': min(rs), 'c': min(csx),
            'h': max(rs) - min(rs) + 1, 'w': max(csx) - min(csx) + 1,
            'size': len(cells),
            'color': int(v[np.argmax(n)]),
            'colors': [int(x) for x in v]}
def get_objects(g, bg=None, diagonal=False, multicolor=True):
    a = _np(g); b = bg_color(g) if bg is None else int(bg)
    st = _ST8 if diagonal else _ST4
    objs = []
    if multicolor:
        lab, n = _cc_label(a != b, structure=st)
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            objs.append(_mk_obj([(int(r), int(c), int(a[r, c]))
                                 for r, c in zip(ys, xs)]))
    else:
        for col in palette(g):
            if col == b: continue
            lab, n = _cc_label(a == col, structure=st)
            for i in range(1, n + 1):
                ys, xs = np.where(lab == i)
                objs.append(_mk_obj([(int(r), int(c), col)
                                     for r, c in zip(ys, xs)]))
    return objs
def obj_size(o):  return o['size']
def obj_color(o): return o['color']
def obj_grid(o, bg=0):
    g = np.full((o['h'], o['w']), int(bg), dtype=int)
    for r, c, v in o['cells']: g[r - o['r'], c - o['c']] = v
    return _ls(g)
def subgrid_at(g, o):
    return crop(g, o['r'], o['c'], o['h'], o['w'])
def largest_object(objs):
    return max(objs, key=lambda o: (o['size'], -o['r'], -o['c']))
def smallest_object(objs):
    return min(objs, key=lambda o: (o['size'], o['r'], o['c']))
def paint_objects(g, objs, color):
    a = _np(g).copy()
    for o in objs:
        for r, c, _ in o['cells']: a[r, c] = int(color)
    return _ls(a)
def erase_objects(g, objs, bg=0):
    a = _np(g).copy()
    for o in objs:
        for r, c, _ in o['cells']: a[r, c] = int(bg)
    return _ls(a)
def move_object(g, o, dr, dc, bg=0):
    a = _np(erase_objects(g, [o], bg)); H, W = a.shape
    for r, c, v in o['cells']:
        nr, nc = r + int(dr), c + int(dc)
        if 0 <= nr < H and 0 <= nc < W: a[nr, nc] = v
    return _ls(a)
def fold_repaint(g, objs, color_fn):
    a = _np(g).copy()
    for o in objs:
        col = color_fn(o)
        if col is None: continue
        for r, c, _ in o['cells']: a[r, c] = int(col)
    return _ls(a)
def count_holes(o):
    m = np.zeros((o['h'], o['w']), dtype=int)
    for r, c, _ in o['cells']: m[r - o['r'], c - o['c']] = 1
    lab, n = _cc_label(m == 0, structure=_ST4)
    holes = 0
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if ys.min() > 0 and xs.min() > 0 and \
           ys.max() < o['h'] - 1 and xs.max() < o['w'] - 1:
            holes += 1
    return holes

# ---------------- physics-ish ----------------
def flood_fill(g, r, c, color):
    a = _np(g).copy(); H, W = a.shape
    r0, c0 = int(r), int(c); tgt = int(a[r0, c0]); col = int(color)
    if tgt == col: return _ls(a)
    stack = [(r0, c0)]
    while stack:
        y, x = stack.pop()
        if 0 <= y < H and 0 <= x < W and a[y, x] == tgt:
            a[y, x] = col
            stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
    return _ls(a)
def gravity(g, direction, bg=None):
    a = _np(g); b = bg_color(g) if bg is None else int(bg)
    H, W = a.shape
    out = np.full_like(a, b)
    if direction in ('down', 'up'):
        for j in range(W):
            col = [v for v in a[:, j].tolist() if v != b]
            if direction == 'down': out[H - len(col):, j] = col
            else:                   out[:len(col), j] = col
    elif direction in ('left', 'right'):
        for i in range(H):
            row = [v for v in a[i, :].tolist() if v != b]
            if direction == 'left': out[i, :len(row)] = row
            else:                   out[i, W - len(row):] = row
    else:
        raise ValueError("gravity: direction must be up/down/left/right")
    return _ls(out)
def is_symmetric_h(g): return grid_eq(g, flip_h(g))
def is_symmetric_v(g): return grid_eq(g, flip_v(g))

EXPORTS = {k: v for k, v in list(globals().items())
           if not k.startswith('_') and callable(v)
           and k not in ('main',)}

# ---------------- enforcement ----------------
_ALLOWED_BUILTINS = ['abs','all','any','bool','dict','divmod','enumerate',
                     'filter','float','frozenset','int','isinstance','len',
                     'list','map','max','min','pow','print','range','repr',
                     'reversed','round','set','sorted','str','sum','tuple','zip']
SAFE_BUILTINS = {k: getattr(builtins, k) for k in _ALLOWED_BUILTINS}

_BANNED_NODES = (ast.While, ast.Import, ast.ImportFrom, ast.Global,
                 ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef,
                 ast.AsyncFor, ast.AsyncWith, ast.Await)
_BANNED_NAMES = {'exec','eval','open','__import__','compile','globals',
                 'locals','vars','getattr','setattr','delattr','input',
                 'breakpoint','exit','quit','help','super','type','object',
                 'memoryview','bytearray','bytes'}

def validate_source(src):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return 'syntax error: %s' % e
    has_transform = False
    for node in ast.walk(tree):
        if isinstance(node, _BANNED_NODES):
            return 'banned construct: %s' % type(node).__name__
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            return 'banned name: %s' % node.id
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            return 'banned dunder attribute: %s' % node.attr
        if isinstance(node, ast.FunctionDef) and node.name == 'transform':
            has_transform = True
    if not has_transform:
        return 'missing def transform(grid)'
    return None

class _Timeout(Exception): pass
def _on_alarm(signum, frame): raise _Timeout()
_HAS_ALARM = hasattr(signal, 'SIGALRM')
if _HAS_ALARM:
    signal.signal(signal.SIGALRM, _on_alarm)

def _norm_out(g):
    a = np.asarray(g)
    if a.dtype == object or a.ndim != 2 or a.size == 0:
        raise ValueError('output must be a non-empty rectangular 2D grid')
    if a.shape[0] > MAXD or a.shape[1] > MAXD:
        raise ValueError('output larger than %dx%d' % (MAXD, MAXD))
    a = a.astype(int)
    if a.min() < 0 or a.max() > 9:
        raise ValueError('colors must be integers 0..9')
    return a.tolist()

def _copy_grid(g): return [row[:] for row in g]

def run_one(src, train, test, tmo):
    err = validate_source(src)
    if err:
        return {'status': 'rejected', 'error': err, 'pairs_passed': 0}
    env = dict(EXPORTS)
    env['np'] = np                      # numpy allowed read-only by convention
    env['__builtins__'] = SAFE_BUILTINS
    try:
        if _HAS_ALARM:
            signal.setitimer(signal.ITIMER_REAL, float(tmo))
        exec(compile(src, '<candidate>', 'exec'), env)   # SINGLE namespace
        tf = env.get('transform')
        if not callable(tf):
            return {'status': 'rejected', 'error': 'transform not callable',
                    'pairs_passed': 0}
        pairs_passed, fail = 0, None
        for i, p in enumerate(train):
            try:
                got = _norm_out(tf(_copy_grid(p['input'])))
            except _Timeout:
                raise
            except Exception as e:
                if fail is None:
                    fail = {'idx': i, 'err': repr(e)[:200]}
                continue
            if got == p['output']:
                pairs_passed += 1
            elif fail is None:
                fail = {'idx': i, 'got': got}
        if pairs_passed == len(train):
            touts = [_norm_out(tf(_copy_grid(t))) for t in test]
            return {'status': 'verified', 'pairs_passed': pairs_passed,
                    'n_pairs': len(train), 'test_outputs': touts}
        out = {'status': 'partial', 'pairs_passed': pairs_passed,
               'n_pairs': len(train)}
        if fail: out.update(fail)
        return out
    except _Timeout:
        return {'status': 'timeout', 'pairs_passed': 0,
                'error': 'exceeded %.1fs' % tmo}
    except Exception as e:
        return {'status': 'error', 'pairs_passed': 0,
                'error': repr(e)[:300]}
    finally:
        if _HAS_ALARM:
            signal.setitimer(signal.ITIMER_REAL, 0)

def main():
    sys.setrecursionlimit(300)
    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp) as f:
        req = json.load(f)
    results = [run_one(src, req['train'], req['test'], req.get('timeout', 4.0))
               for src in req['programs']]
    with open(outp, 'w') as f:
        json.dump({'results': results}, f)

if __name__ == '__main__':
    main()
'''

# ----------------------------------------------------------------------------
# Prompting
# ----------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are an expert ARC puzzle solver. You write short Python solutions "
    "using ONLY the provided DSL helpers plus basic pure Python "
    "(for-loops, comprehensions, if/else, lambda). Follow the rules exactly."
)

CHEAT_SHEET = """AVAILABLE DSL (grids are list[list[int]], colors 0-9):
grid_shape(g)->(h,w) | new_grid(h,w,fill=0) | crop(g,r,c,h,w)
rot90(g) rot180(g) rot270(g) | flip_h(g) flip_v(g) | transpose(g)
bg_color(g) | palette(g) | color_counts(g)->dict | find_color_cells(g,color)
replace_color(g,a,b) | swap_colors(g,a,b) | map_colors(g,{a:b,...})
crop_to_content(g,bg=None)
paste(dst,src,r,c,transparent=None) | concat_h(a,b) concat_v(a,b)
tile(g,ny,nx) | upscale(g,fy,fx) | downscale(g,fy,fx)
cellwise(a,b,f) | grid_eq(a,b) | split_rows(g,n) split_cols(g,n)
split_by_separator(g)->list of sub-grids (row-major)
get_objects(g,bg=None,diagonal=False,multicolor=True)->list of obj
  obj = {'cells':[(r,c,color)..],'r','c','h','w','size','color','colors'}
obj_size(o) obj_color(o) | obj_grid(o,bg=0) | subgrid_at(g,o)
largest_object(objs) smallest_object(objs)
paint_objects(g,objs,color) | erase_objects(g,objs,bg=0)
move_object(g,o,dr,dc,bg=0) | fold_repaint(g,objs,color_fn)
count_holes(o) | flood_fill(g,r,c,color) | gravity(g,'down'|'up'|'left'|'right')
is_symmetric_h(g) is_symmetric_v(g)

RULES:
- Define exactly one function: def transform(grid) -> grid.
- NO imports. NO while loops. NO classes. Use only helpers above + pure python.
- Return a new grid (list of lists of ints 0-9), never mutate weird state.
"""

FEWSHOT = """Two tiny examples of the expected style:

Example task: recolor every object to the color of the largest object.
```python
def transform(grid):
    objs = get_objects(grid)
    big = largest_object(objs)
    return fold_repaint(grid, objs, lambda o: obj_color(big))
```

Example task: output = input rotated 90 degrees clockwise, colors 1<->3 swapped.
```python
def transform(grid):
    return swap_colors(rot90(grid), 1, 3)
```
"""

def render_grid(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in g)

def shape_hint(task):
    ins  = [(len(p['input']),  len(p['input'][0]))  for p in task['train']]
    outs = [(len(p['output']), len(p['output'][0])) for p in task['train']]
    if all(a == b for a, b in zip(ins, outs)):
        return "Observed shape rule: every output has the SAME shape as its input."
    if len(set(outs)) == 1:
        h, w = outs[0]
        return f"Observed shape rule: every output is exactly {h}x{w}."
    ratios = set()
    ok = True
    for (ih, iw), (oh, ow) in zip(ins, outs):
        if oh % ih == 0 and ow % iw == 0:
            ratios.add((oh // ih, ow // iw))
        else:
            ok = False
    if ok and len(ratios) == 1:
        fy, fx = next(iter(ratios))
        return f"Observed shape rule: output = input scaled by ({fy}x, {fx}x)."
    return "Observed shape rule: varies -- infer it from the pairs. " + \
           f"in->out shapes: {list(zip(ins, outs))}"

def build_prompt(task, repair=None):
    lines = [CHEAT_SHEET, FEWSHOT, "--- TASK ---"]
    for i, p in enumerate(task['train'], 1):
        ih, iw = len(p['input']), len(p['input'][0])
        oh, ow = len(p['output']), len(p['output'][0])
        lines.append(f"Train {i} Input ({ih}x{iw}):\n{render_grid(p['input'])}")
        lines.append(f"Train {i} Output ({oh}x{ow}):\n{render_grid(p['output'])}")
    for j, t in enumerate(task['test'], 1):
        th, tw = len(t['input']), len(t['input'][0])
        lines.append(f"Test {j} Input ({th}x{tw}):\n{render_grid(t['input'])}")
    lines.append(shape_hint(task))
    if repair:
        lines.append(
            "### PREVIOUS ATTEMPT (best so far) ###\n"
            f"```python\n{repair['code']}\n```\n"
            f"It passed {repair['pairs_passed']}/{repair['n_pairs']} train pairs "
            f"but FAILED on Train {repair['fail_idx'] + 1}.\n"
            f"{repair['diff']}\n"
            "Fix the LOGIC. Keep the DSL rules. Output the corrected full solution."
        )
    lines.append("Think briefly, then answer with ONE ```python code block "
                 "containing def transform(grid).")
    return "\n\n".join(lines)

def repair_diff(task, part):
    i = part.get('idx', part.get('fail_idx', 0))
    exp = task['train'][i]['output']
    if 'got' in part and part['got'] is not None:
        got = part['got']
        gh, gw = len(got), len(got[0])
        eh, ew = len(exp), len(exp[0])
        if (gh, gw) != (eh, ew):
            return f"Your output shape was {gh}x{gw}, expected {eh}x{ew}."
        if gh <= 15 and gw <= 15:
            return ("Expected:\n" + render_grid(exp) +
                    "\nYour output:\n" + render_grid(got))
        wrong = sum(1 for r in range(eh) for c in range(ew)
                    if got[r][c] != exp[r][c])
        return f"Shape correct ({eh}x{ew}) but {wrong} cells are wrong."
    if 'err' in part:
        return f"It raised: {part['err']}"
    return "Output did not match."

def extract_code(text):
    m = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if m: return m[-1].strip()
    m = re.findall(r"```(.*?)```", text, re.DOTALL)
    if m: return m[-1].strip()
    idx = text.find("def transform")
    return text[idx:].strip() if idx >= 0 else text.strip()

def code_key(src):
    return hashlib.md5(re.sub(r"\s+", "", src).encode()).hexdigest()

# ----------------------------------------------------------------------------
# Sandbox client (used by workers AND selftest; torch-free)
# ----------------------------------------------------------------------------
def run_verifier(programs, train, test, prog_timeout, tag):
    inp  = os.path.join(WORK_DIR, f"_vin_{tag}.json")
    outp = os.path.join(WORK_DIR, f"_vout_{tag}.json")
    with open(inp, 'w') as f:
        json.dump({'programs': programs, 'train': train,
                   'test': test, 'timeout': prog_timeout}, f)
    total = 10 + prog_timeout * len(programs) + 5
    try:
        subprocess.run([sys.executable, RUNTIME_PATH, inp, outp],
                       timeout=total,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return [{'status': 'timeout', 'pairs_passed': 0,
                 'error': 'verifier batch timeout'}] * len(programs)
    finally:
        try: os.remove(inp)
        except OSError: pass
    if not os.path.exists(outp):
        return [{'status': 'error', 'pairs_passed': 0,
                 'error': 'verifier crashed'}] * len(programs)
    with open(outp) as f:
        res = json.load(f)['results']
    try: os.remove(outp)
    except OSError: pass
    return res

# ----------------------------------------------------------------------------
# Per-task solve (runs inside a GPU worker)
# ----------------------------------------------------------------------------
def gen_batch(model, tok, prompt, k, temp, max_new):
    import torch
    msgs = [{'role': 'system', 'content': SYSTEM_MSG},
            {'role': 'user',   'content': prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors='pt').to(model.device)
    try:
        with torch.inference_mode():
            out = model.generate(**enc,
                                 do_sample=True,
                                 temperature=max(temp, 1e-3),
                                 top_p=0.95,
                                 num_return_sequences=k,
                                 max_new_tokens=max_new,
                                 pad_token_id=tok.eos_token_id)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if k > 2:
            return gen_batch(model, tok, prompt, k // 2, temp, max_new)
        raise
    L = enc['input_ids'].shape[1]
    return [tok.decode(o[L:], skip_special_tokens=True) for o in out]

def vote_outputs(verified, n_test):
    # verified: list of (src, result) ; result['test_outputs'] aligned by index
    outs = []
    for t in range(n_test):
        buckets = {}
        for src, res in verified:
            key = json.dumps(res['test_outputs'][t])
            buckets.setdefault(key, {'votes': 0, 'srcs': []})
            buckets[key]['votes'] += 1
            buckets[key]['srcs'].append(src)
        ranked = sorted(buckets.items(),
                        key=lambda kv: (-kv[1]['votes'],
                                        min(len(s) for s in kv[1]['srcs'])))
        top_key, top = ranked[0]
        alt = json.loads(ranked[1][0]) if len(ranked) > 1 else None
        outs.append({'attempt': json.loads(top_key),
                     'votes': top['votes'],
                     'alt': alt,
                     'program': min(top['srcs'], key=len)})
    return outs

def solve_task(tid, task, cfg, deadline_ts, rank, log):
    t0 = time.time()
    task_end = min(t0 + cfg['per_task_cap'], deadline_ts)
    rec = {'verified': False, 'n_verified': 0, 'outputs': [],
           'best_partial': 0, 'n_pairs': len(task['train']),
           'n_cand': 0, 'elapsed': 0.0}
    prompt = build_prompt(task)
    if len(prompt) > cfg['max_prompt_chars']:
        rec['skip'] = 'prompt_too_long(%d)' % len(prompt)
        log(f"[W{rank}] {tid}  SKIP prompt too long ({len(prompt)} chars)")
        return rec
    model, tok = cfg['_model'], cfg['_tok']
    test_inputs = [t['input'] for t in task['test']]

    seen, all_codes, all_res = set(), [], []
    def run_round(k, temp, repair=None):
        texts = gen_batch(model, tok, build_prompt(task, repair),
                          k, temp, cfg['max_new_tokens'])
        codes = []
        for txt in texts:
            c = extract_code(txt)
            h = code_key(c)
            if c and h not in seen:
                seen.add(h); codes.append(c)
        if not codes: return
        res = run_verifier(codes, task['train'], test_inputs,
                           cfg['prog_timeout'], f"{rank}_{tid}")
        all_codes.extend(codes); all_res.extend(res)

    run_round(cfg['k1'], cfg['temp1'])
    verified = [(c, r) for c, r in zip(all_codes, all_res)
                if r['status'] == 'verified']
    if not verified and time.time() < task_end - 30:
        partials = [(c, r) for c, r in zip(all_codes, all_res)
                    if r['status'] == 'partial' and r['pairs_passed'] >= 1]
        if partials:
            c, r = max(partials, key=lambda cr: (cr[1]['pairs_passed'],
                                                 -len(cr[0])))
            rep = {'code': c[:4000],
                   'pairs_passed': r['pairs_passed'],
                   'n_pairs': r['n_pairs'],
                   'fail_idx': r.get('idx', 0),
                   'diff': repair_diff(task, r)}
            run_round(cfg['k2'], cfg['temp2'], repair=rep)
            verified = [(c2, r2) for c2, r2 in zip(all_codes, all_res)
                        if r2['status'] == 'verified']

    rec['n_cand'] = len(all_codes)
    rec['best_partial'] = max([r.get('pairs_passed', 0) for r in all_res],
                              default=0)
    if verified:
        rec['verified']   = True
        rec['n_verified'] = len(verified)
        rec['outputs']    = vote_outputs(verified, len(test_inputs))
        rec['program']    = rec['outputs'][0]['program']
    rec['elapsed'] = round(time.time() - t0, 1)
    tag = (f"VERIFIED {rec['n_verified']} prog(s)" if rec['verified']
           else f"no verified (best {rec['best_partial']}/{rec['n_pairs']})")
    log(f"[W{rank}] {tid}  {tag}  cand {rec['n_cand']}  {rec['elapsed']}s")
    return rec

# ----------------------------------------------------------------------------
# GPU worker process (heavy imports live here -> spawn stays cheap)
# ----------------------------------------------------------------------------
def worker_main(rank, gpu_id, items, cfg, deadline_ts, partial_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    def log(msg): print(msg, flush=True)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(cfg['seed'] + rank)
        random.seed(cfg['seed'] + rank)
        log(f"[W{rank}] loading {cfg['model_path']} on gpu {gpu_id} ...")
        tok = AutoTokenizer.from_pretrained(cfg['model_path'])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_path'], torch_dtype=torch.bfloat16
        ).to('cuda').eval()
        cfg = dict(cfg); cfg['_model'] = model; cfg['_tok'] = tok
        log(f"[W{rank}] ready, {len(items)} tasks")
    except Exception as e:
        with open(partial_path, 'a') as f:
            f.write(json.dumps({'tid': '__worker_error__',
                                'error': repr(e)[:500]}) + "\n")
        print(f"[W{rank}] FATAL during model load: {e!r}", flush=True)
        return
    with open(partial_path, 'a') as f:
        for tid, task in items:
            if time.time() > deadline_ts - 25:
                log(f"[W{rank}] deadline reached, stopping")
                break
            try:
                rec = solve_task(tid, task, cfg, deadline_ts, rank, log)
            except Exception as e:
                rec = {'verified': False, 'error': repr(e)[:300]}
                log(f"[W{rank}] {tid}  ERROR {e!r}")
            f.write(json.dumps({'tid': tid, **{k: v for k, v in rec.items()
                                               if not k.startswith('_')}}) + "\n")
            f.flush()

# ----------------------------------------------------------------------------
# Data / model resolution
# ----------------------------------------------------------------------------
def resolve_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(d, "arc-agi_evaluation_challenges.json")) \
           or os.path.exists(os.path.join(d, "arc-agi_test_challenges.json")):
            return d
    return DATA_DIR_CANDIDATES[0]

def load_challenges():
    data_dir = resolve_data_dir()
    rerun  = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    test_p = os.path.join(data_dir, "arc-agi_test_challenges.json")
    eval_p = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")
    sol_p  = os.path.join(data_dir, "arc-agi_evaluation_solutions.json")
    if rerun or (os.path.exists(test_p) and not os.path.exists(eval_p)):
        with open(test_p) as f:
            return json.load(f), None, 'SUBMIT', data_dir
    with open(eval_p) as f:
        ch = json.load(f)
    sols = None
    if os.path.exists(sol_p):
        with open(sol_p) as f:
            sols = json.load(f)
    return ch, sols, 'LOCAL_EVAL', data_dir

def resolve_model_path(override=None):
    cands = ([override] if override else []) + MODEL_PATH_CANDIDATES
    for p in cands:
        if p and os.path.exists(os.path.join(p, "config.json")):
            return p
    root = "/kaggle/input"
    if os.path.isdir(root):
        hits = []
        for dirpath, _, files in os.walk(root):
            if dirpath.count(os.sep) - root.count(os.sep) > 7:
                continue
            if "config.json" in files and (
                "tokenizer.json" in files or "tokenizer_config.json" in files):
                low = dirpath.lower()
                if "coder" in low or "qwen" in low:
                    hits.append((0 if "7b" in low else 1, dirpath))
        if hits:
            return sorted(hits)[0][1]
    return None

def est_cost(task):
    s = 0
    for p in task['train']:
        s += len(p['input']) * len(p['input'][0])
        s += len(p['output']) * len(p['output'][0])
    for t in task['test']:
        s += len(t['input']) * len(t['input'][0])
    return s

# ----------------------------------------------------------------------------
# Selftest: proves the sandbox (and the exec fix) with zero GPU/model
# ----------------------------------------------------------------------------
def selftest():
    print("== Leg C sandbox selftest (no GPU, no model) ==")
    train = [
        {'input':  [[0,0,0],[0,3,0],[0,0,0]],
         'output': [[0,0,0],[0,2,0],[0,0,0]]},
        {'input':  [[5,0],[0,0]],
         'output': [[2,0],[0,0]]},
    ]
    test = [[[0,7,0],[0,0,0],[0,0,0]]]
    cases = [
        ("dsl_reachable (old two-dict exec bug regression)",
         "def transform(grid):\n"
         "    objs = get_objects(grid)\n"
         "    return paint_objects(grid, objs, 2)\n", 'verified'),
        ("while loop must be rejected",
         "def transform(grid):\n"
         "    while True:\n        pass\n    return grid\n", 'rejected'),
        ("import must be rejected",
         "import os\ndef transform(grid):\n    return grid\n", 'rejected'),
        ("missing transform must be rejected",
         "def solve(grid):\n    return grid\n", 'rejected'),
        ("runaway loop must time out",
         "def transform(grid):\n"
         "    x = 0\n"
         "    for i in range(10**9):\n        x += i\n"
         "    return grid\n", 'timeout'),
        ("wrong logic must be partial",
         "def transform(grid):\n    return replace_color(grid, 3, 4)\n",
         'partial'),
        ("dunder escape must be rejected",
         "def transform(grid):\n"
         "    f = (1).__class__\n    return grid\n", 'rejected'),
    ]
    programs = [src for _, src, _ in cases]
    res = run_verifier(programs, train, test, 3.0, "selftest")
    ok = True
    for (name, _, want), r in zip(cases, res):
        got = r['status']
        mark = "PASS" if got == want else "FAIL"
        if got != want: ok = False
        print(f"  [{mark}] {name:48s} -> {got} "
              f"({r.get('error','')[:60]})")
    v = res[0]
    if v['status'] == 'verified':
        pred = v['test_outputs'][0]
        want = [[0,2,0],[0,0,0],[0,0,0]]
        mark = "PASS" if pred == want else "FAIL"
        if pred != want: ok = False
        print(f"  [{mark}] verified program predicts test correctly -> {pred}")
    names = re.findall(r"([a-z_][a-z0-9_]*)\(", CHEAT_SHEET)
    import ast as _ast
    tree = _ast.parse(RUNTIME_CODE)
    defined = {n.name for n in _ast.walk(tree)
               if isinstance(n, _ast.FunctionDef)}
    missing = [n for n in set(names) - {'f', 'color_fn', 'transform'}
               if n not in defined]
    mark = "PASS" if not missing else "FAIL"
    if missing: ok = False
    print(f"  [{mark}] cheat sheet <-> runtime consistency "
          f"(missing: {missing})")
    print("== selftest", "OK ==" if ok else "FAILED ==")
    return 0 if ok else 1

# ----------------------------------------------------------------------------
# Reporting: input + guessed result + mark
# ----------------------------------------------------------------------------
def mark_and_report(results, challenges, sols, cfg):
    audit, n_ver, n_out, n_correct = {}, 0, 0, 0
    printed = 0
    for tid, rec in results.items():
        if tid == '__worker_error__':
            continue
        task = challenges[tid]
        entry = {'verified': rec.get('verified', False),
                 'n_verified': rec.get('n_verified', 0),
                 'best_partial': rec.get('best_partial', 0),
                 'n_pairs': rec.get('n_pairs', 0),
                 'elapsed': rec.get('elapsed', 0),
                 'tests': []}
        for t_idx, t in enumerate(task['test']):
            out = (rec['outputs'][t_idx]['attempt']
                   if rec.get('verified') and t_idx < len(rec.get('outputs', []))
                   else None)
            exp = sols[tid][t_idx] if sols and tid in sols else None
            mark = None
            if out is not None:
                n_out += 1
                if exp is not None:
                    mark = (out == exp)
                    n_correct += int(mark)
            entry['tests'].append({'input': t['input'],
                                   'guess': out,
                                   'expected': exp,
                                   'mark': mark})
        if rec.get('verified'):
            n_ver += 1
            if printed < cfg['print_grids']:
                printed += 1
                for t_idx, tr in enumerate(entry['tests']):
                    sym = ('?' if tr['mark'] is None
                           else ('O' if tr['mark'] else 'X'))
                    print(f"\n--- {tid} test#{t_idx}  mark={sym} "
                          f"(votes {rec['outputs'][t_idx]['votes']}"
                          f"/{rec['n_verified']}) ---")
                    print("input:");  print(render_grid(tr['input']))
                    print("guess:");  print(render_grid(tr['guess']))
                    if tr['expected'] is not None:
                        print("expected:"); print(render_grid(tr['expected']))
        audit[tid] = entry
    with open(AUDIT_PATH, 'w') as f:
        json.dump(audit, f)
    print("\n================ Leg C summary ================")
    print(f" tasks attempted      : {len(audit)}")
    print(f" tasks verified       : {n_ver}")
    print(f" test outputs guessed : {n_out}")
    if sols is not None:
        pct = 100.0 * n_correct / max(n_out, 1)
        print(f" verified & correct   : {n_correct}/{n_out}  "
              f"(precision {pct:.1f}%)")
        solved = [t for t, e in audit.items()
                  if e['verified'] and all(x['mark'] for x in e['tests'])]
        print(f" fully-correct tasks  : {len(solved)} -> {solved}")
    print(f" results -> {RESULTS_PATH}")
    print(f" audit   -> {AUDIT_PATH}")
    print("===============================================")

def write_outputs(results, challenges):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f)
    # standalone submission (smoke/debug only; merge path is the real one)
    sub = {}
    for tid, task in challenges.items():
        rec = results.get(tid, {})
        entries = []
        for t_idx, t in enumerate(task['test']):
            if rec.get('verified') and t_idx < len(rec.get('outputs', [])):
                o = rec['outputs'][t_idx]
                a1 = o['attempt']
                a2 = o['alt'] if o.get('alt') else a1
            else:
                a1 = a2 = t['input']
            entries.append({'attempt_1': a1, 'attempt_2': a2})
        sub[tid] = entries
    with open(SUB_PATH, 'w') as f:
        json.dump(sub, f)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-h',     type=float, default=DEFAULTS['budget_h'])
    ap.add_argument('--per-task-cap', type=float, default=DEFAULTS['per_task_cap'])
    ap.add_argument('--k1',   type=int, default=DEFAULTS['k1'])
    ap.add_argument('--k2',   type=int, default=DEFAULTS['k2'])
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--model-path', type=str, default=None)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--max-tasks', type=int, default=0)
    ap.add_argument('--print-grids', type=int, default=DEFAULTS['print_grids'])
    ap.add_argument('--end-ts', type=float, default=0.0,
                    help='absolute unix ts; deadline = min(now+budget, end-ts)')
    ap.add_argument('--tasks', type=str, default='',
                    help='comma-separated task ids: restrict the run to these')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    with open(RUNTIME_PATH, 'w') as f:
        f.write(RUNTIME_CODE)

    if args.selftest:
        sys.exit(selftest())

    challenges, sols, mode, data_dir = load_challenges()
    print(f"[LegC] mode={mode}  data={data_dir}  tasks={len(challenges)}")

    model_path = resolve_model_path(args.model_path)
    if model_path is None:
        print("[LegC] WARNING: no coder model found under /kaggle/input.")
        print("       Attach Qwen2.5-Coder-7B-Instruct, or pass --model-path.")
        print("       Writing empty results so the merge step is a no-op.")
        write_outputs({}, challenges)
        mark_and_report({}, challenges, sols,
                        {'print_grids': 0})
        return
    print(f"[LegC] model: {model_path}")

    cfg = dict(DEFAULTS)
    cfg.update(per_task_cap=args.per_task_cap, k1=args.k1, k2=args.k2,
               print_grids=args.print_grids, model_path=model_path)

    items = sorted(challenges.items(), key=lambda kv: est_cost(kv[1]))
    if args.tasks:
        only = {t.strip() for t in args.tasks.split(',') if t.strip()}
        items = [kv for kv in items if kv[0] in only]
        print(f"[LegC] restricted to {len(items)} task(s) via --tasks")
    if args.smoke:
        items = items[:args.smoke]
        cfg['print_grids'] = max(cfg['print_grids'], args.smoke)
    elif args.max_tasks:
        items = items[:args.max_tasks]

    try:
        n_gpu = int(subprocess.run(
            ['nvidia-smi', '-L'], capture_output=True, text=True
        ).stdout.strip().count('GPU '))
    except Exception:
        n_gpu = 1
    n_workers = args.workers or max(1, min(4, n_gpu))
    shards = [items[i::n_workers] for i in range(n_workers)]

    deadline_ts = time.time() + args.budget_h * 3600
    if args.end_ts > 0:
        deadline_ts = min(deadline_ts, args.end_ts)
    remain = deadline_ts - time.time()
    if remain < 300:
        print(f"[LegC] budget too small ({remain:.0f}s) -- writing empty "
              "results and exiting so the base pipeline keeps all its time.")
        write_outputs({}, challenges)
        mark_and_report({}, challenges, sols, {'print_grids': 0})
        return
    print(f"[LegC] {len(items)} tasks, {n_workers} worker(s), "
          f"budget {remain/3600:.2f} h")

    for r in range(n_workers):
        p = PARTIAL_TPL.format(rank=r)
        if os.path.exists(p): os.remove(p)

    ctx = mp.get_context('spawn')
    procs = []
    for r in range(n_workers):
        p = ctx.Process(target=worker_main,
                        args=(r, r % max(n_gpu, 1), shards[r], cfg,
                              deadline_ts, PARTIAL_TPL.format(rank=r)))
        p.start(); procs.append(p)
    hard_stop = deadline_ts + 600
    for p in procs:
        p.join(max(1.0, hard_stop - time.time()))
    for p in procs:
        if p.is_alive():
            p.terminate(); p.join()

    results = {}
    for r in range(n_workers):
        path = PARTIAL_TPL.format(rank=r)
        if not os.path.exists(path): continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    results[rec.pop('tid')] = rec
                except Exception:
                    continue

    write_outputs(results, challenges)
    mark_and_report(results, challenges, sols, cfg)

if __name__ == '__main__':
    main()
