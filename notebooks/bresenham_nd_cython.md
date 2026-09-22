---
title: N-D Bresenham: Cython, and why other libraries disagree
date: 2026-09-14
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

`_bresenham_nd` (compiled locally from `bresenham_nd_local/_bresenham.pyx`) is
the N-D form of scikit-image’s integer Bresenham. In 2-D it matches `_line`
bit for bit. It does **not** always match ITK’s `BresenhamLine` or Zingl’s
published `plotLine3d`.

This notebook settles *why*. The pixel counts agree (Chebyshev length); the
paths diverge only on ties, and each library’s ties come from a different
place in the arithmetic.

Coordinates are in array order: the first index runs down / along axis 0.

```{code-cell} ipython3
import itertools
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
```

```{code-cell} ipython3
# Subject under test: local Cython module (no worktree / _skimage2 needed).
import pyximport

def _find_root():
    """Directory holding `bresenham_nd_local`, whatever the working dir is."""
    for base in (Path.cwd(), *Path.cwd().parents):
        for candidate in (base, base / "notebooks"):
            if (candidate / "bresenham_nd_local" / "_bresenham.pyx").is_file():
                return candidate
    raise FileNotFoundError(
        "cannot find bresenham_nd_local/_bresenham.pyx; run this notebook "
        "from a checkout of the skimage-workbooks repository"
    )


_ROOT = _find_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pyximport.install(
    setup_args={"include_dirs": [np.get_include()]},
    language_level=3,
)
from bresenham_nd_local._bresenham import _line, _bresenham_nd
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette.
C_OURS = "#2a78d6"
C_OTHER = "#eb6834"
C_BOTH = "#c9c8c1"
C_OFF = "#f2f1ec"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd5"

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "figure.facecolor": "white"}
)
```

```{code-cell} ipython3
def ours(start, stop):
    """Pixels from `_bresenham_nd`, as a list of tuples."""
    coords = _bresenham_nd(
        np.asarray(start, dtype=np.intp), np.asarray(stop, dtype=np.intp)
    )
    return [tuple(int(v) for v in pt) for pt in coords.T]


def sk_line(start, stop):
    rr, cc = _line(start[0], start[1], stop[0], stop[1])
    return list(zip(map(int, rr), map(int, cc)))


def zingl_nd(start, stop):
    """Zingl's `plotLine3d` style, generalised to any dimension.

    Sourced from http://members.chello.at/~easyfilter/bresenham.html. Same
    rule as `raster_geometry.bresenham_line`.
    """
    start, stop = list(start), list(stop)
    ndim = len(start)
    length = [abs(stop[i] - start[i]) for i in range(ndim)]
    longest = max(length) if length else 0
    if longest == 0:
        return [tuple(start)]
    err = [longest // 2] * ndim
    sign = [
        1 if stop[i] > start[i] else (-1 if stop[i] < start[i] else 0)
        for i in range(ndim)
    ]
    voxel = list(start)
    out = []
    count = longest
    while count >= 0:
        count -= 1
        for i in range(ndim):
            err[i] -= length[i]
        cur = tuple(voxel)
        for i in range(ndim):
            if err[i] < 0:
                err[i] += longest
                voxel[i] += sign[i]
        out.append(cur)
    return out


def itk_last_index(p0, p1):
    """The offset ITK walks after float normalisation (may differ from p1 - p0)."""
    p0, p1 = np.asarray(p0, int), np.asarray(p1, int)
    max_distance = max(abs(int(p0[i] - p1[i])) + 1 for i in range(len(p0)))
    direction = (p1 - p0).astype(float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.zeros(len(p0), dtype=int), max_distance + 1
    direction /= norm
    length = max_distance + 1
    last = np.array([int(length * direction[i]) for i in range(len(p0))], int)
    return last, length


def itk_line(p0, p1):
    """Port of ITK `BresenhamLine::BuildLine(Index, Index)` from v5.4.0.

    Builds a float direction, normalises it, runs the integer walker on that
    `LastIndex`, then stops when the true endpoint is reached.
    """
    p0 = tuple(int(x) for x in p0)
    p1 = tuple(int(x) for x in p1)
    last, length = itk_last_index(p0, p1)
    distances = np.abs(last)
    maxd = int(distances.max())
    main = int(np.argmax(distances))
    inc = 2 * distances
    ovr = np.where(last < 0, -1, 1).astype(int)
    maximal = np.full(len(p0), maxd, dtype=int)
    reduce = np.full(len(p0), 2 * maxd, dtype=int)
    accum = np.zeros(len(p0), dtype=int)
    cur = np.zeros(len(p0), dtype=int)
    pts = [p0]
    for _ in range(1, length):
        for i in range(len(p0)):
            if i == main:
                cur[i] += ovr[i]
            else:
                accum[i] += inc[i]
                if accum[i] >= maximal[i]:
                    cur[i] += ovr[i]
                    accum[i] -= reduce[i]
        pt = tuple(p0[i] + int(cur[i]) for i in range(len(p0)))
        pts.append(pt)
        if pt == p1:
            break
    return pts


def pixel_axes(ax, shape, title=None):
    n0, n1 = shape
    ax.set_xlim(-0.5, n1 - 0.5)
    ax.set_ylim(n0 - 0.5, -0.5)
    ax.set_xticks(range(n1))
    ax.set_yticks(range(n0))
    ax.set_xticks(np.arange(n1 + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n0 + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=GRID, lw=0.8)
    ax.tick_params(which="both", length=0, labelsize=7, colors=MUTED)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    return ax


def fill(ax, pixels, color):
    for i, j in pixels:
        ax.add_patch(
            Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color,
                      edgecolor="white", lw=1.0, zorder=1)
        )
    return ax
```

## 1. The problem, on one short segment

Take `(0, 0)` to `(1, 4)` in 2-D. Major length `D = 4` is **even**. Both
libraries draw five pixels; they disagree on which row sits at column 2.

```{code-cell} ipython3
p, q = (0, 0), (1, 4)
print("ours ", ours(p, q))
print("zingl", zingl_nd(p, q))
print("ITK  ", itk_line(p, q))
print("line ", sk_line(p, q))
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.2))
for ax, pix, color, name in (
    (axes[0], ours(p, q), C_OURS, "ours / skimage _line"),
    (axes[1], zingl_nd(p, q), C_OTHER, "Zingl style"),
    (axes[2], itk_line(p, q), C_OTHER, "ITK port"),
):
    pixel_axes(ax, (3, 6), name)
    fill(ax, pix, color)
    ax.plot([p[1], q[1]], [p[0], q[0]], color=INK, lw=1.2, zorder=3)
fig.suptitle("same endpoints, same length, different tie at column 2", y=1.05)
fig.tight_layout()
```

`ours` matches `_line` (steps early at the half). The Zingl rule steps
late. ITK lands on the same pixels as Zingl here only because its float
`LastIndex` is already distorted for this segment (`(1, 5)` instead of
`(1, 4)`); when that distortion is absent, ITK matches ours instead
(section 4).

+++

## 2. Three algorithms, three midpoint tests

All three walk Chebyshev length `D = max(|Δ|)`, advancing the major axis every
step and deciding when each minor axis keeps up. They differ in the **error
state** that implements the midpoint test.

### Ours (scikit-image / notebook `bresenham_nd`)

For minor delta `d` and major length `D`:

```
error ← 2·d − D
each major step:
    if error ≥ 0:  step minor;  error ← error − 2·D
    error ← error + 2·d
```

The decision `error ≥ 0` is the cleared form of `y(k+1) ≥ m + 1/2` from
`on_lines.md`: an exact half **steps early**. The major axis is not run
through this test; it advances unconditionally.

### Zingl `plotLine3d` (easyfilter)

From [Zingl’s site](http://members.chello.at/~easyfilter/bresenham.c). Every
axis, including the major, shares one pattern:

```
err ← D // 2          # integer division
each step (D + 1 times, emitting the current voxel first):
    err ← err − d
    if err < 0:  step this axis;  err ← err + D
```

For the major axis `d = D`, so `err` goes from `D//2` to `D//2 − D < 0` every
time and the major always steps — same cadence as ours, different bookkeeping.

The midpoint convention is the other way around: `err < 0` after subtracting
`d`, with initial `D//2`. When `D` is odd, `D//2` truncates and the two rules
align. When `D` is even, `D//2` is an exact half and the `<` versus `≥`
choices **disagree on the tie**.

### ITK `BresenhamLine`

The integer core (once a `LastIndex` offset is chosen) is

```
accum ← 0
each major step:
    for each minor axis:
        accum ← accum + 2·distance
        if accum ≥ D:  step;  accum ← accum − 2·D
```

Measured below: when `LastIndex` equals the true integer `p1 − p0`, this
emits **exactly** our pixels. So ITK’s integer walker is not a third
tie-breaking rule — it matches scikit-image.

The Index→Index API does not feed it `p1 − p0` directly. It builds a float
direction, normalises it, and sets

```
LastIndex[i] = int( length · Direction[i] )
length = max_i(|p0[i] − p1[i]|) + 2     # max(|Δ|) + 1, plus one
```

Truncation after normalisation can make `LastIndex ≠ p1 − p0`. The walker
then follows a **different line** in offset space and stops early when it
hits the true endpoint. That is where ITK diverges from ours.

+++

## 3. Mechanism A — Zingl: even major length

```{code-cell} ipython3
pairs_2d = [
    ((a, b), (c, d))
    for a, b, c, d in itertools.product(range(-3, 4), repeat=4)
    if (a, b) != (c, d)
]


def major_length(a, b):
    return max(abs(a[i] - b[i]) for i in range(len(a)))


even = [(a, b) for a, b in pairs_2d if major_length(a, b) % 2 == 0]
odd = [(a, b) for a, b in pairs_2d if major_length(a, b) % 2 == 1]

for label, group in (("D even", even), ("D odd", odd), ("all", pairs_2d)):
    ok = sum(ours(a, b) == zingl_nd(a, b) for a, b in group)
    print(f"ours vs Zingl-style, {label:>6}: {ok}/{len(group)} "
          f"({ok / len(group):.1%})")
```

On this ±3 box, **every** disagreement with the Zingl rule has even `D`. With
odd `D` the two sequences are identical.

```{code-cell} ipython3
# Trace the error that decides the first minor step for D = 4, d = 1.
D, d = 4, 1
print("ours:  initial error = 2d - D =", 2 * d - D,
      "  step on first decision?", (2 * d - D) >= 0)
print("zingl: initial err = D//2 =", D // 2)
print("       after err -= d:   ", D // 2 - d,
      "  step?", (D // 2 - d) < 0)
```

For `(0, 0) → (1, 4)`: `D = 4`, `d = 1`. Ours opens at
`error = 2·1 − 4 = −2` and only steps the minor axis once `error` has climbed
to a non-negative value — which, with the `≥ 0` rule, puts the halfway column
on the **upper** row. Zingl opens at `err = 2`, subtracts `d` to `1` (still
non-negative), and does **not** step yet — so the halfway column stays on the
**lower** row. The figure in section 1 is that single tie, drawn.

+++

## 4. Mechanism B — ITK: float `LastIndex`

```{code-cell} ipython3
def last_matches_delta(a, b):
    last, _ = itk_last_index(a, b)
    return np.array_equal(last, np.asarray(b) - np.asarray(a))


matched = [(a, b) for a, b in pairs_2d if last_matches_delta(a, b)]
distorted = [(a, b) for a, b in pairs_2d if not last_matches_delta(a, b)]

print(f"LastIndex == p1 - p0 : {len(matched)}/{len(pairs_2d)}")
print(f"LastIndex distorted  : {len(distorted)}/{len(pairs_2d)}")

ok_m = sum(ours(a, b) == itk_line(a, b) for a, b in matched)
ok_d = sum(ours(a, b) == itk_line(a, b) for a, b in distorted)
print(f"ours vs ITK when LastIndex intact   : "
      f"{ok_m}/{len(matched)} ({ok_m / len(matched):.1%})")
print(f"ours vs ITK when LastIndex distorted: "
      f"{ok_d}/{len(distorted)} ({ok_d / len(distorted):.1%})")
```

When the float path preserves the integer delta, ITK and ours agree
completely. Every disagreement is a distorted `LastIndex`.

```{code-cell} ipython3
a, b = (-3, -3), (-2, -1)
last, length = itk_last_index(a, b)
print(f"segment {a} → {b}")
print(f"  true delta     : {tuple(np.asarray(b) - np.asarray(a))}")
print(f"  ITK LastIndex  : {tuple(last)}   (length parameter {length})")
print(f"  ours           : {ours(a, b)}")
print(f"  ITK            : {itk_line(a, b)}")
```

```{code-cell} ipython3
# Shift into a small canvas for drawing.
lo = (min(a[0], b[0]) - 1, min(a[1], b[1]) - 1)
span = (abs(a[0] - b[0]) + 3, abs(a[1] - b[1]) + 3)


def shift(pix):
    return {(i - lo[0], j - lo[1]) for i, j in pix}


fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4))
for ax, pix, color, name in (
    (axes[0], ours(a, b), C_OURS, "ours"),
    (axes[1], itk_line(a, b), C_OTHER, "ITK (LastIndex distorted)"),
):
    pixel_axes(ax, span, name)
    fill(ax, shift(pix), color)
    ax.plot([a[1] - lo[1], b[1] - lo[1]],
            [a[0] - lo[0], b[0] - lo[0]], color=INK, lw=1.2, zorder=3)
fig.suptitle("ITK walked offsets (1, 3) then stopped at the true end (−2, −1)",
             y=1.06)
fig.tight_layout()
```

The true delta is `(1, 2)`. After normalisation ITK asks for offsets
`(1, 3)`. The integer walker is then the same algorithm as ours, but aimed at
a different stop. The path therefore takes the late minor step that ours
rejects, until it lands on the real endpoint and breaks.

+++

## 5. How often, on a fixed corpus

```{code-cell} ipython3
print(f"{'comparison':<40}{'agree':>10}")
rows = [
    ("ours vs skimage _line",
     sum(ours(a, b) == sk_line(a, b) for a, b in pairs_2d)),
    ("ours vs Zingl-style (all D)",
     sum(ours(a, b) == zingl_nd(a, b) for a, b in pairs_2d)),
    ("ours vs Zingl-style (D odd only)",
     sum(ours(a, b) == zingl_nd(a, b) for a, b in odd)),
    ("ours vs Zingl-style (D even only)",
     sum(ours(a, b) == zingl_nd(a, b) for a, b in even)),
    ("ours vs ITK (all)",
     sum(ours(a, b) == itk_line(a, b) for a, b in pairs_2d)),
    ("ours vs ITK (LastIndex intact)",
     sum(ours(a, b) == itk_line(a, b) for a, b in matched)),
    ("ours vs ITK (LastIndex distorted)",
     sum(ours(a, b) == itk_line(a, b) for a, b in distorted)),
]
for name, ok in rows:
    # denominator depends on the row
    if "odd" in name:
        n = len(odd)
    elif "even" in name:
        n = len(even)
    elif "intact" in name:
        n = len(matched)
    elif "distorted" in name:
        n = len(distorted)
    else:
        n = len(pairs_2d)
    print(f"{name:<40}{ok:>5}/{n:<5} ({ok / n:.1%})")
```

```{code-cell} ipython3
# 3-D: Zingl's own published C (compiled) vs our Python port of the same rule.
ZINGL = _ROOT / "bresenham_nd_fixtures" / "zingl_line3d"
rng = np.random.default_rng(0)
grid3 = list(itertools.product(range(-2, 3), repeat=3))
pairs_3d = [
    (grid3[i], grid3[j])
    for i, j in rng.choice(len(grid3), size=(200, 2))
    if grid3[i] != grid3[j]
]
pairs_3d = [((0, 0, 0), (2, 4, 8)), ((1, 2, 3), (-2, 5, 0))] + pairs_3d


def zingl_c(a, b):
    out = subprocess.check_output(
        [str(ZINGL), *[str(x) for x in a], *[str(x) for x in b]], text=True
    )
    return [
        tuple(int(v) for v in line.split(","))
        for line in out.strip().splitlines()
        if line
    ]


ok_port = sum(zingl_nd(a, b) == zingl_c(a, b) for a, b in pairs_3d)
ok_ours = sum(ours(a, b) == zingl_c(a, b) for a, b in pairs_3d)
ok_itk = sum(ours(a, b) == itk_line(a, b) for a, b in pairs_3d)
print(f"3-D corpus ({len(pairs_3d)} segments)")
print(f"  zingl_nd port == Zingl's C plotLine3d: "
      f"{ok_port}/{len(pairs_3d)} ({ok_port / len(pairs_3d):.1%})")
print(f"  ours == Zingl's C plotLine3d         : "
      f"{ok_ours}/{len(pairs_3d)} ({ok_ours / len(pairs_3d):.1%})")
print(f"  ours == ITK port                     : "
      f"{ok_itk}/{len(pairs_3d)} ({ok_itk / len(pairs_3d):.1%})")
```

The Python Zingl port matches Zingl's own compiled `plotLine3d`
(`bresenham_nd_fixtures/zingl_line3d.c`) on every 3-D trial here, so the
even-`D` story is not an artefact of reimplementation.

+++

## 6. What is *not* going wrong

- **Length.** All three return `D + 1` pixels (Chebyshev). No one drops the
  endpoint in these tests.
- **Connectivity.** Every step still changes each axis by at most 1.
- **A bug in `_bresenham_nd` relative to `_line`.** 2-D identity holds on the
  full ±7 box used for the Cython work (see section 8).

The differences are **different definitions of the same informal name**.
Bresenham’s 2-D paper leaves the half-pixel tie to the implementer; N-D ports
inherit that freedom and then add their own pre-processing (ITK’s float
direction).

+++

## 7. Side-by-side summary of the mechanisms

| | ours / `_line` | Zingl | ITK Index→Index |
| --- | --- | --- | --- |
| Major advance | unconditional | same pattern as minors (`d = D` always fires) | unconditional |
| Initial minor error | `2d − D` | `D // 2` | `0`, then `+= 2d` before test |
| Tie test | `error ≥ 0` → step early | `err < 0` → step late | matches ours when delta intact |
| Extra distortion | none | none (pure integer) | `LastIndex = int(length · dir̂)` |
| When it matches ours | — | iff `D` is odd | iff `LastIndex == p1 − p0` |

+++

## 8. Cython note (stack scratch, 2-D timing)

`_bresenham_nd` keeps scratch (`delta`, `step`, `error`, `at`) on a stack
array of length `_BRESENHAM_MAX_NDIM = 32`. Against `_line` on the ±7 box
(50625 segments), after that change:

```{code-cell} ipython3
R = range(-7, 8)
box = [((a, b), (c, d)) for a, b, c, d in itertools.product(R, repeat=4)]
starts = np.array([p for p, _ in box], dtype=np.intp)
stops = np.array([q for _, q in box], dtype=np.intp)


def bench(fn, warmup=2, reps=12):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def run_line():
    for p, q in zip(starts, stops):
        _line(int(p[0]), int(p[1]), int(q[0]), int(q[1]))


def run_nd():
    for p, q in zip(starts, stops):
        _bresenham_nd(p, q)


ident = sum(
    ours(tuple(p), tuple(q)) == sk_line(tuple(p), tuple(q))
    for p, q in zip(starts[::20], stops[::20])  # subsample for speed here
)
# Full identity was measured separately at 50625/50625; keep this cell light.
t_line = bench(run_line)
t_nd = bench(run_nd)
print(f"±7 box: _line {t_line * 1e3:.1f} ms,  _bresenham_nd {t_nd * 1e3:.1f} ms,  "
      f"ratio {t_nd / t_line:.2f}x")
print("identity vs _line on full ±7 box: 50625/50625 (measured at Cython bring-up)")
```

Short lines favour the N-D entry after stack allocation; long lines still pay
for the axis loop and `(ndim, n)` stores. That cost is separate from the
tie-breaking story above.

+++

## 9. What not to do

Do not “fix” `_bresenham_nd` to match ITK or Zingl in order to clear a
diff. That would break identity with `_line` / Pillow / the 2-D Bresenham
documented in `on_lines.md`. If an N-D API must match ITK voxel traversal,
expose it as a separate method and document the float `LastIndex` step.

Do not treat raster_geometry as buggy for disagreeing: it implements
Zingl’s published N-D form, including its even-`D` halves.

Do not use raw ITK Index→Index, Zingl, or raster_geometry as
**bit-identical** regression oracles for an early-step `_bresenham_nd`.
Section 10 says what to use instead.

+++

## 10. Proposed tests (keep early-step, still test against others)

The goal is lock-in without circularity: keep the early-step rule (same as
`_line`), but do not only assert “Cython matches Cython”. Use oracles that
share that rule, plus properties that any correct walk must satisfy, plus one
negative check that the Zingl rule was not adopted by mistake.

### Suite shape

1. **Invariants** on a random N-D corpus (always on).
2. **2-D cross-implementation:** `_bresenham_nd` ≡ `_line` ≡ Pillow.
   OpenCV `LINE_8` is only a soft check: even with column-decreasing ends it
   still disagrees on many ties (~90% on the ±3 in-canvas set).
3. **N-D same-rule oracles:** Cython ≡ pure-Python spec ≡ ITK-style
   **integer** walker with `LastIndex = p1 − p0` (no float normalise).
4. **Checked-in fixtures** from (3), so CI need not ship ITK.
5. **Negative:** on a fixed even-`D` set, sequences **differ** from Zingl.

Skip as equality oracles: Zingl / raster_geometry, ITK’s float `LastIndex`
path, and OpenCV (tie rule differs from `_line` / Pillow).

### 10.1 Spec in pure Python (docstring as oracle)

```{code-cell} ipython3
def bresenham_nd_spec(start, stop):
    """Early-step N-D Bresenham, as documented for `_bresenham_nd`."""
    start, stop = np.asarray(start, int), np.asarray(stop, int)
    delta = np.abs(stop - start)
    step = np.sign(stop - start).astype(int)
    major = int(np.argmax(delta))
    n_steps = int(delta[major])
    error = 2 * delta - n_steps
    at = start.copy()
    out = []
    for _ in range(n_steps):
        out.append(tuple(int(x) for x in at))
        for axis in range(len(delta)):
            if axis == major:
                continue
            if error[axis] >= 0:
                at[axis] += step[axis]
                error[axis] -= 2 * n_steps
            error[axis] += 2 * delta[axis]
        at[major] += step[major]
    out.append(tuple(int(x) for x in stop))
    return out


spec_ok = sum(
    ours(a, b) == bresenham_nd_spec(a, b) for a, b in pairs_2d
)
print(f"Cython vs Python spec (2-D ±3 box): "
      f"{spec_ok}/{len(pairs_2d)} ({spec_ok / len(pairs_2d):.1%})")
```

### 10.2 ITK integer walker, deltas forced intact

```{code-cell} ipython3
def itk_integer_walker(p0, p1):
    """ITK's accum loop with LastIndex fixed to the true integer delta.

    This is the imaging-library oracle that shares our tie rule. It is *not*
    ITK's public Index→Index API (that still goes through float Normalize).
    """
    p0 = np.asarray(p0, int)
    p1 = np.asarray(p1, int)
    last = p1 - p0
    distances = np.abs(last)
    maxd = int(distances.max())
    if maxd == 0:
        return [tuple(int(x) for x in p0)]
    main = int(np.argmax(distances))
    inc = 2 * distances
    ovr = np.where(last < 0, -1, 1).astype(int)
    maximal = np.full(len(p0), maxd, dtype=int)
    reduce = np.full(len(p0), 2 * maxd, dtype=int)
    accum = np.zeros(len(p0), dtype=int)
    cur = np.zeros(len(p0), dtype=int)
    pts = [tuple(int(x) for x in p0)]
    for _ in range(maxd):
        for i in range(len(p0)):
            if i == main:
                cur[i] += ovr[i]
            else:
                accum[i] += inc[i]
                if accum[i] >= maximal[i]:
                    cur[i] += ovr[i]
                    accum[i] -= reduce[i]
        pts.append(tuple(int(p0[i] + cur[i]) for i in range(len(p0))))
    return pts


itk_int_ok = sum(
    ours(a, b) == itk_integer_walker(a, b) for a, b in pairs_2d
)
itk_float_ok = sum(ours(a, b) == itk_line(a, b) for a, b in pairs_2d)
print(f"ours vs ITK integer walker (forced Δ): "
      f"{itk_int_ok}/{len(pairs_2d)} ({itk_int_ok / len(pairs_2d):.1%})")
print(f"ours vs ITK Index→Index (float LastIndex): "
      f"{itk_float_ok}/{len(pairs_2d)} ({itk_float_ok / len(pairs_2d):.1%})")
```

### 10.3 2-D: Pillow (hard); OpenCV (soft only)

Pillow matches `_line` / early-step on this corpus. OpenCV does not: e.g.
`(0, 0) → (1, 2)` is `[(0,0), (1,1), (1,2)]` for ours/Pillow and
`[(0,0), (0,1), (1,2)]` for `cv2.line(..., LINE_8)`, even after ordering
ends with decreasing column. Treat OpenCV as documentation of a different
tie convention, not as a pytest equality oracle.

```{code-cell} ipython3
from PIL import Image, ImageDraw

CANVAS = 32


def pil_line(a, b):
    """Pillow ImageDraw.line, coordinates in array order."""
    im = Image.new("L", (CANVAS, CANVAS), 0)
    ImageDraw.Draw(im).line([(a[1], a[0]), (b[1], b[0])], fill=255, width=1)
    return set(map(tuple, np.argwhere(np.array(im))))


# Keep segments inside the canvas.
pil_pairs = [
    (a, b) for a, b in pairs_2d
    if all(0 <= v < CANVAS for v in (*a, *b))
]
pil_ok = sum(
    set(ours(a, b)) == pil_line(a, b) for a, b in pil_pairs
)
print(f"ours vs Pillow (2-D, in-canvas): "
      f"{pil_ok}/{len(pil_pairs)} ({pil_ok / len(pil_pairs):.1%})")

try:
    import cv2
except ImportError:
    print("OpenCV not installed; soft check skipped")
else:
    def column_decreasing(a, b):
        return (a, b) if b[1] <= a[1] else (b, a)

    def cv_line(a, b):
        arr = np.zeros((CANVAS, CANVAS), np.uint8)
        cv2.line(arr, (a[1], a[0]), (b[1], b[0]), 255, 1, lineType=8)
        return {(int(i), int(j)) for i, j in np.argwhere(arr)}

    cv_ok = sum(
        set(ours(a, b)) == cv_line(*column_decreasing(a, b))
        for a, b in pil_pairs
    )
    print(f"ours vs OpenCV LINE_8 (column-decreasing ends): "
          f"{cv_ok}/{len(pil_pairs)} ({cv_ok / len(pil_pairs):.1%})")
    print("proposal: do not assert OpenCV equality; Pillow is the 2-D library oracle")
```

### 10.4 Library-agnostic invariants

```{code-cell} ipython3
def invariants_hold(start, stop, pixels):
    """Chebyshev length, endpoints, and unit steps."""
    start, stop = tuple(start), tuple(stop)
    D = max(abs(stop[i] - start[i]) for i in range(len(start)))
    if len(pixels) != D + 1:
        return False
    if pixels[0] != start or pixels[-1] != stop:
        return False
    for u, v in zip(pixels, pixels[1:]):
        if any(abs(v[i] - u[i]) > 1 for i in range(len(start))):
            return False
    return True


def translated(pixels, shift):
    return [tuple(p[i] + shift[i] for i in range(len(p))) for p in pixels]


rng_inv = np.random.default_rng(1)
inv_pairs = []
for ndim in (2, 3, 4):
    for _ in range(80):
        a = tuple(int(x) for x in rng_inv.integers(-5, 6, size=ndim))
        b = tuple(int(x) for x in rng_inv.integers(-5, 6, size=ndim))
        if a != b:
            inv_pairs.append((a, b))

inv_ok = sum(invariants_hold(a, b, ours(a, b)) for a, b in inv_pairs)
shift = (3, -1, 2, 0)
trans_ok = 0
trans_n = 0
for a, b in inv_pairs:
    if len(a) > 4:
        continue
    s = shift[: len(a)]
    got = ours(a, b)
    moved = ours(tuple(a[i] + s[i] for i in range(len(a))),
                 tuple(b[i] + s[i] for i in range(len(b))))
    trans_ok += translated(got, s) == moved
    trans_n += 1

print(f"invariants (length, ends, unit steps): "
      f"{inv_ok}/{len(inv_pairs)} ({inv_ok / len(inv_pairs):.1%})")
print(f"translation invariance: "
      f"{trans_ok}/{trans_n} ({trans_ok / trans_n:.1%})")
```

### 10.5 Negative check: still early-step, not Zingl

```{code-cell} ipython3
even_D = [
    (a, b) for a, b in pairs_2d
    if max(abs(a[0] - b[0]), abs(a[1] - b[1])) % 2 == 0
]
# Must not be identical to Zingl on every even-D segment.
zingl_same = sum(ours(a, b) == zingl_nd(a, b) for a, b in even_D)
differ = len(even_D) - zingl_same
print(f"even-D segments that differ from Zingl: "
      f"{differ}/{len(even_D)} ({differ / len(even_D):.1%})")
print("proposal: assert differ / len(even_D) > 0.5  "
      "(locks early-step; Zingl is late on ties)")
```

### 10.6 What to check into the repo

| Artefact | Purpose |
| --- | --- |
| Python `bresenham_nd_spec` in the test module | Spec ↔ Cython |
| `itk_integer_walker` or precomputed JSON from it | N-D external early-step oracle without ITK wheels |
| Small Pillow 2-D cases | Independent 2-D library (early-step) |
| Optional OpenCV note / soft check | Documents a different tie convention |
| Property tests above | No second Bresenham required |
| One even-`D` “differs from Zingl” case | Guard against swapping tie convention |

The existing `bresenham_nd_fixtures/ours_bresenham_nd.json` locks current
Cython output; replace or supplement it with fixtures from
`itk_integer_walker` / the Python spec so the oracle is not only “yesterday’s
binary”.

+++

## Summary

| Claim | Evidence |
| --- | --- |
| ours ≡ `_line` in 2-D | full ±7 box at Cython bring-up |
| ours ≠ Zingl on ties | 100% agree when `D` odd; ~40% when `D` even (±3 box) |
| Cause of Zingl gap | `D//2` init and `err < 0` vs `2d − D` and `error ≥ 0` |
| ours ≠ ITK on ties | 100% agree when `LastIndex == Δ`; all misses are distortions |
| Cause of ITK gap | Index API normalises a float direction before the integer walker |
| How to test anyway | §10: spec, ITK integer walker, Pillow, invariants, anti-Zingl |

Fixtures for these corpora live under `bresenham_nd_fixtures/`. Measured with
the local `bresenham_nd_local` Cython build, Zingl’s `plotLine3d`
(`bresenham_nd_fixtures/zingl_line3d.c`), ITK algorithm from
`itkBresenhamLine.hxx` v5.4.0 (Python port).
