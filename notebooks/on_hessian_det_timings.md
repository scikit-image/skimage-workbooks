---
title: 'On the Hessian determinant, timed'
date: 2026-11-28
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

:::{changelog}
2026-11-28 | opencode:grok-4.7 | Create workbook: the formula, its accuracy, and the timings
2026-11-28 | opencode:deepseek-v4.1-flash-free | Add the range of formula differences on three photographs
:::

`hessian_matrix_det` builds a full matrix at every pixel and calls
`np.linalg.det`, except on a 2D image when `approximate=True`. The Hessian is
symmetric. In 2D and 3D the determinant is a short polynomial in the unique
elements. This notebook measures that polynomial against `np.linalg.det`: the
error, and the time.

Coordinates are in array order. `hessian_matrix` returns the upper triangle in
the order of `combinations_with_replacement`: `[Hrr, Hrc, Hcc]` in 2D, and
`[H00, H01, H02, H11, H12, H22]` in 3D. `on_hessian_matrix_det.md` compares
routes that produce a determinant image. It does not time a 3D formula. This
notebook starts from the Hessian elements.

The source read for the formula is
`hessian-refactor/src/_skimage2/feature/corner.py` at `060a313d1`. The cells
call the installed copy, which is the unmodified function.

## 1. Setup

```{code-cell} ipython3
import sys
import time
import platform
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
```

```{code-cell} ipython3
for base in (Path.cwd(), *Path.cwd().parents):
    helper = base / "notebooks" / "nbhelper.py"
    if helper.is_file():
        sys.path.insert(0, str(helper.parent))
        break
else:
    raise FileNotFoundError("nbhelper.py")

from nbhelper import show_table
```

```{code-cell} ipython3
import skimage as ski
from skimage.feature import hessian_matrix, hessian_matrix_det
from _skimage2.feature.corner import _symmetric_image

print(platform.platform())
print("numpy", np.__version__)
print("scikit-image", ski.__version__)
```

```{code-cell} ipython3
# Categorical slots 1-3 of the default palette. validate_palette.js, light
# mode, as recorded for these slots in on_hessian_matrix_det.md: all checks
# pass; worst adjacent CVD ΔE 11.8; contrast WARN, so the chart has a legend
# and direct labels.
COLOURS = {
    "current": "#257de0",
    "closed form": "#f47230",
    "assemble": "#00b894",
}
INK, MUTED, GRID = "#191919", "#666666", "#dedcd5"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_orange", ["#1b64be", "#efefef", "#e76624"])

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "axes.edgecolor": GRID,
     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
     "figure.facecolor": "white"}
)
```

## 2. Helpers

The [determinant](https://en.wikipedia.org/wiki/Determinant) of the symmetric
2 × 2 matrix with diagonal entries $a, b$ and off-diagonal entry $d$ is
$ab - d^{2}$.
NumPy's documentation of
[`numpy.linalg.det`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html)
uses that formula for a general 2 × 2 matrix, and says the implementation is
the LAPACK routine `dgetrf`.

For the symmetric 3 × 3 matrix

$$
H = \begin{bmatrix}
a & d & e \\
d & b & f \\
e & f & c
\end{bmatrix}
$$

the [Laplace expansion](https://en.wikipedia.org/wiki/Laplace_expansion) along
the first row is

$$
\det H = a(bc - f^{2}) - d(dc - ef) + e(df - be).
$$

The six letters are the upper triangle in list order: $a, d, e, b, f, c$ are
$H_{00}, H_{01}, H_{02}, H_{11}, H_{12}, H_{22}$.

```{code-cell} ipython3
def closed_det(elems):
    """Determinant of a symmetric 2×2 or 3×3 from its upper triangle.

    The products are computed in float64, then cast back to the element dtype.
    `np.linalg.det` does the same upcast: its LAPACK signature is ``d->d``.
    """
    dtype = np.asarray(elems[0]).dtype
    work = [np.asarray(e, dtype=np.float64) for e in elems]
    if len(work) == 3:
        a, d, b = work
        det = a * b - d * d
    elif len(work) == 6:
        a, d, e, b, f, c = work
        det = (a * (b * c - f * f)
               - d * (d * c - e * f)
               + e * (d * f - b * e))
    else:
        raise ValueError(f"{len(work)} unique elements is not 2D or 3D")
    return det.astype(dtype, copy=False)


def best_time(func, repeats=3):
    """Shortest of `repeats` wall-clock times of `func()`, in milliseconds.

    One untimed call warms the caches.
    """
    func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return 1e3 * min(times)
```

```{code-cell} ipython3
import inspect

print("dgetrf" in np.linalg.det.__doc__)
text = inspect.getsource(hessian_matrix_det)
start = text.index("else:")
print(text[start:].rstrip())
```

`_symmetric_image` writes each off-diagonal element into both triangles, then
`np.linalg.det` factorizes every matrix. The list from `hessian_matrix` already
holds each element once. The next cell checks that list order against the
assembled matrix.

```{code-cell} ipython3
rng = np.random.default_rng(0)


def order_matches(image):
    """Largest difference between the upper triangle and `hessian_matrix`."""
    elems = hessian_matrix(image, sigma=1, use_gaussian_derivatives=False)
    full = _symmetric_image(elems)
    gaps = []
    for idx, (i, j) in enumerate(combinations_with_replacement(range(image.ndim), 2)):
        gaps.append(np.max(np.abs(full[..., i, j] - elems[idx])))
        gaps.append(np.max(np.abs(full[..., j, i] - elems[idx])))
    return len(elems), max(gaps)


print("2D", order_matches(rng.random((24, 24))))
print("3D", order_matches(rng.random((12, 12, 12))))
```

## 3. One matrix

Two symmetric integer matrices. The exact determinant is an integer, and it
fits in the 53-bit mantissa, so a float64 result can equal it.

```{code-cell} ipython3
def exact2(a, d, b):
    """Exact 2×2 determinant, in Python integers."""
    a, d, b = int(a), int(d), int(b)
    return a * b - d * d


def exact3(a, d, e, b, f, c):
    """Exact 3×3 determinant, in Python integers."""
    a, d, e, b, f, c = (int(x) for x in (a, d, e, b, f, c))
    return a * (b * c - f * f) - d * (d * c - e * f) + e * (d * f - b * e)


M2 = np.array([[34, 40], [40, -35]], dtype=np.float64)
M3 = np.array([[40, -38, -32],
               [-38, 22, -40],
               [-32, -40, -7]], dtype=np.float64)
rows = []
for name, M, exact in (
        ("2×2", M2, exact2(34, 40, -35)),
        ("3×3", M3, exact3(40, -38, -32, 22, -40, -7))):
    closed = closed_det([M[i, j] for i, j in
                          combinations_with_replacement(range(M.shape[0]), 2)])
    rows.append({"matrix": name, "method": "exact",
                 "determinant": exact, "absolute error": 0.0})
    rows.append({"matrix": name, "method": "closed form",
                 "determinant": float(closed),
                 "absolute error": abs(float(closed) - exact)})
    rows.append({"matrix": name, "method": "np.linalg.det",
                 "determinant": float(np.linalg.det(M)),
                 "absolute error": abs(float(np.linalg.det(M)) - exact)})
show_table(pd.DataFrame(rows), index=["matrix", "method"], floatfmt=".16g")
```

The closed form returns the integer. `np.linalg.det` does not. Both absolute
errors are far below 1, so both round to the same integer. §4 shows that the
polynomials are the same. The residual is rounding.

## 4. The identity

```{code-cell} ipython3
a, d, e, b, f, c = sp.symbols("a d e b f c")
M2 = sp.Matrix([[a, d], [d, b]])
M3 = sp.Matrix([[a, d, e], [d, b, f], [e, f, c]])
formula2 = a * b - d**2
formula3 = a * (b * c - f**2) - d * (d * c - e * f) + e * (d * f - b * e)
sym_gap2 = sp.simplify(M2.det() - formula2)
sym_gap3 = sp.simplify(M3.det() - formula3)
print(sym_gap2)
print(sym_gap3)
```

Both differences are 0. The polynomials are the determinant, not an
approximation to it. The only gap between the closed form and `np.linalg.det`
is the floating-point evaluation.

## 5. Side by side on one blob

A Gaussian blob has a positive determinant at the centre, where both
curvatures have the same sign, and a negative ring where the surface is a
saddle. The three panels share one color scale. The difference panel is the
closed form minus `np.linalg.det`.

```{code-cell} ipython3
def blob(n, width):
    """`n` × `n` Gaussian of `width`, centred on a pixel, peak 1."""
    r = np.arange(n)[:, None] - n // 2
    c = np.arange(n)[None, :] - n // 2
    return np.exp(-(r**2 + c**2) / (2 * width**2))


blob_image = blob(81, 8)
blob_elems = hessian_matrix(blob_image, sigma=2, use_gaussian_derivatives=False)
blob_closed = closed_det(blob_elems)
blob_linalg = np.linalg.det(_symmetric_image(blob_elems))
blob_diff = blob_closed - blob_linalg
blob_peak = np.max(np.abs(blob_linalg))
print("max |difference| / max |det|", np.max(np.abs(blob_diff)) / blob_peak)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), constrained_layout=True)
limit = np.max(np.abs(blob_closed))
norm = TwoSlopeNorm(0, -limit, limit)
panels = (("closed form", blob_closed),
          ("np.linalg.det", blob_linalg),
          ("difference", blob_diff))
for ax, (name, field) in zip(axes, panels):
    im = ax.imshow(field, cmap=DIVERGING, norm=norm)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.046, pad=0.02)
plt.show()
```

On this color scale the difference panel is blank. The printed ratio is the
largest absolute difference divided by the largest absolute determinant.

The same comparison on `camera`, at `sigma` 1. The entry scale $s^{n}$ uses
$s$, the largest absolute entry of the matrix at that pixel, and $n$, the
matrix order. The condition proxy is $s^{n} / |{\det}|$.

```{code-cell} ipython3
camera = ski.util.img_as_float(ski.data.camera())
cam_elems = hessian_matrix(camera, sigma=1, use_gaussian_derivatives=False)
cam_closed = closed_det(cam_elems)
cam_linalg = np.linalg.det(_symmetric_image(cam_elems))
peak = np.max(np.abs(cam_linalg))
diff = np.abs(cam_closed - cam_linalg)
sign_cut = 1e-6 * peak
sign_bad = np.sum((np.sign(cam_closed) != np.sign(cam_linalg))
                  & (np.abs(cam_linalg) > sign_cut))
cam_scale = np.maximum.reduce([np.abs(e) for e in cam_elems])**2
cam_cond = cam_scale / np.maximum(np.abs(cam_linalg), np.finfo(float).tiny)
print("shape", camera.shape, "dtype", cam_closed.dtype)
print("max |difference| / max |det|", diff.max() / peak)
print("pixels with a sign disagreement above 1e-6 of the peak",
      int(sign_bad), "of", cam_closed.size)
print("max s^n / |det|", cam_cond.max())
```

The two arrays differ in the last bits. No pixel with a determinant above
$10^{-6}$ of the peak changes sign. The largest $s^{n} / |{\det}|$ is the
number printed above. A target eigenvalue ratio of $10^{-16}$ in §7 is a
condition near $10^{16}$. The printed proxy is smaller than that.

## 6. A corpus of matrices

Four thousand symmetric integer matrices, entries drawn uniformly from $-40$
to $40$. The exact determinant is the integer polynomial. Every value in this
corpus fits in the 53-bit mantissa: the cell prints the largest absolute
determinant next to $2^{53}$.

```{code-cell} ipython3
N_INT = 4000
entries = rng.integers(-40, 41, size=(N_INT, 6))
ai, di, ei, bi, fi, ci = entries.T
truth = (ai * (bi * ci - fi * fi)
         - di * (di * ci - ei * fi)
         + ei * (di * fi - bi * ei))
print("max |entry|", int(np.max(np.abs(entries))))
print("max |exact det|", int(np.max(np.abs(truth))), "  2**53", 2**53)
af, df, ef, bf, ff, cf = (entries.astype(np.float64)).T
int_closed = af * (bf * cf - ff * ff) - df * (df * cf - ef * ff) + ef * (df * ff - bf * ef)
int_mats = np.empty((N_INT, 3, 3))
int_mats[:, 0, 0] = af
int_mats[:, 0, 1] = df
int_mats[:, 0, 2] = ef
int_mats[:, 1, 0] = df
int_mats[:, 1, 1] = bf
int_mats[:, 1, 2] = ff
int_mats[:, 2, 0] = ef
int_mats[:, 2, 1] = ff
int_mats[:, 2, 2] = cf
int_linear = np.linalg.det(int_mats)
truth_f = truth.astype(np.float64)


def error_row(method, got):
    err = np.abs(got - truth_f)
    sign_bad = np.sum((np.sign(got) != np.sign(truth_f)) & (truth != 0))
    return {"method": method,
            "max absolute error": err.max(),
            "matrices with error > 0.5": int(np.sum(err > 0.5)),
            "sign disagreements, det ≠ 0": int(sign_bad)}


show_table(pd.DataFrame([
    error_row("closed form", int_closed),
    error_row("np.linalg.det", int_linear),
]), floatfmt=".3g")
```

The closed form matches the integer on every matrix in this corpus.
`np.linalg.det` misses by the max absolute error in the table. That error is
below 0.5, so the result still rounds to the integer. Neither method changes
a nonzero sign.

A symmetric outer product $vv^{\mathsf T}$ has determinant 0. The closed form
returns 0 on every matrix in the next cell. `np.linalg.det` does not.

```{code-cell} ipython3
N_SING = 2000
vecs = rng.integers(-20, 21, size=(N_SING, 3))
a = vecs[:, 0] * vecs[:, 0]
d = vecs[:, 0] * vecs[:, 1]
e = vecs[:, 0] * vecs[:, 2]
b = vecs[:, 1] * vecs[:, 1]
f = vecs[:, 1] * vecs[:, 2]
c = vecs[:, 2] * vecs[:, 2]
closed0 = closed_det([x.astype(np.float64) for x in (a, d, e, b, f, c)])
mats = np.empty((N_SING, 3, 3))
mats[:, 0, 0] = a
mats[:, 0, 1] = d
mats[:, 0, 2] = e
mats[:, 1, 0] = d
mats[:, 1, 1] = b
mats[:, 1, 2] = f
mats[:, 2, 0] = e
mats[:, 2, 1] = f
mats[:, 2, 2] = c
linear0 = np.linalg.det(mats.astype(np.float64))
show_table(pd.DataFrame([
    {"method": "closed form",
     "nonzero results": int(np.sum(closed0 != 0)),
     "max absolute result": float(np.max(np.abs(closed0)))},
    {"method": "np.linalg.det",
     "nonzero results": int(np.sum(linear0 != 0)),
     "max absolute result": float(np.max(np.abs(linear0)))},
]), floatfmt=".3g")
```

The nonzero `np.linalg.det` results are at most the max absolute result in
the table.

`np.linalg.det` widens a float32 matrix to float64, factorizes, and casts the
result back. A formula evaluated in float32 does not take that path. The
proposed function does: it computes in float64 and casts back.

```{code-cell} ipython3
N_F32 = 2000
raw = rng.normal(size=(N_F32, 6)).astype(np.float32)
elems32 = list(raw.T)
f32_formula = (elems32[0] * (elems32[3] * elems32[5] - elems32[4] ** 2)
               - elems32[1] * (elems32[1] * elems32[5] - elems32[2] * elems32[4])
               + elems32[2] * (elems32[1] * elems32[4] - elems32[3] * elems32[2]))
cast_back = closed_det(elems32)
mats = np.empty((N_F32, 3, 3), np.float32)
mats[:, 0, 0] = elems32[0]
mats[:, 0, 1] = elems32[1]
mats[:, 0, 2] = elems32[2]
mats[:, 1, 0] = elems32[1]
mats[:, 1, 1] = elems32[3]
mats[:, 1, 2] = elems32[4]
mats[:, 2, 0] = elems32[2]
mats[:, 2, 1] = elems32[4]
mats[:, 2, 2] = elems32[5]
linear32 = np.linalg.det(mats)
show_table(pd.DataFrame([
    {"method": "float32 formula",
     "result dtype": str(f32_formula.dtype),
     "values differing from np.linalg.det": int(np.sum(f32_formula != linear32))},
    {"method": "float64, then cast",
     "result dtype": str(cast_back.dtype),
     "values differing from np.linalg.det": int(np.sum(cast_back != linear32))},
    {"method": "np.linalg.det",
     "result dtype": str(linear32.dtype),
     "values differing from np.linalg.det": 0},
]))
```

On this sample the float64 formula, cast back to float32, matches
`np.linalg.det` at every matrix. The float32 formula does not. Both results
have dtype float32. §9 checks that `hessian_matrix_det` returns that dtype
for a float32 volume.

Image Hessians are a different distribution from these random matrices. The
reference for a sample of pixels is the same polynomial evaluated at 40 decimal
digits from the exact float64 entries. The entry scale is $s^{n}$, with $s$
the largest absolute entry of that matrix and $n$ the order. The scaled error
is the absolute error divided by the entry scale.

```{code-cell} ipython3
from decimal import Decimal, localcontext


def decimal_det(elems):
    """Determinant of one symmetric matrix, at 40 decimal places."""
    vals = [Decimal(float(x)) for x in elems]
    with localcontext() as ctx:
        ctx.prec = 40
        if len(vals) == 3:
            a, d, b = vals
            return a * b - d * d
        a, d, e, b, f, c = vals
        return a * (b * c - f * f) - d * (d * c - e * f) + e * (d * f - b * e)


def matrix_order(elems):
    """2 if `elems` is a 2×2 upper triangle, 3 if it is a 3×3 upper triangle."""
    if len(elems) == 3:
        return 2
    if len(elems) == 6:
        return 3
    raise ValueError(len(elems))


def sample_errors(elems, closed, linear, indices):
    """Scaled error of both methods against the decimal determinant."""
    order = matrix_order(elems)
    rows_c, rows_l = [], []
    for idx in indices:
        entry = [float(e.ravel()[idx]) for e in elems]
        ref = decimal_det(entry)
        scale = max(abs(x) for x in entry) ** order
        scale = max(scale, 1e-300)
        rows_c.append(abs(float(closed.ravel()[idx]) - float(ref)) / scale)
        rows_l.append(abs(float(linear.ravel()[idx]) - float(ref)) / scale)
    return np.array(rows_c), np.array(rows_l)


def image_sample(image, sigma):
    elems = hessian_matrix(image, sigma=sigma, use_gaussian_derivatives=False)
    closed = closed_det(elems)
    linear = np.linalg.det(_symmetric_image(elems))
    scale = np.maximum.reduce([np.abs(e) for e in elems]) ** matrix_order(elems)
    ratio = np.abs(linear) / np.maximum(scale, 1e-300)
    worst = np.argpartition(ratio.ravel(), 20)[:20]
    pick = rng.choice(ratio.size, 20, replace=False)
    return elems, closed, linear, np.concatenate([worst, pick])


sphere = np.zeros((40, 40, 40))
zz, yy, xx = np.ogrid[-20:20, -20:20, -20:20]
sphere[(zz**2 + yy**2 + xx**2) < 10**2] = 1.0
sphere = sphere + 0.05 * rng.random(sphere.shape)

rows = []
for name, image in (("camera", camera), ("sphere", sphere)):
    elems, closed, linear, indices = image_sample(image, sigma=1)
    err_c, err_l = sample_errors(elems, closed, linear, indices)
    rows.append({"image": name, "method": "closed form",
                 "pixels": len(indices),
                 "max scaled error": err_c.max(),
                 "median scaled error": np.median(err_c)})
    rows.append({"image": name, "method": "np.linalg.det",
                 "pixels": len(indices),
                 "max scaled error": err_l.max(),
                 "median scaled error": np.median(err_l)})
show_table(pd.DataFrame(rows), index=["image", "method"], floatfmt=".3g")
```

The sample is 20 pixels with the smallest $|{\det}| / s^{n}$, and 20 pixels
drawn at random, at `sigma` 1. $s^{n}$ uses the matrix order, 2 or 3, not the
length of the upper-triangle list. On this sample the scaled error of both
methods is at most a few times $10^{-15}$.

## 7. Ill-conditioned matrices

Cancellation is a different question. A matrix with one tiny eigenvalue has a
determinant far below its entry scale. Both evaluations then lose relative
accuracy. The matrices below are random rotations of a diagonal matrix, then
symmetrized. The target ratio is the small diagonal entry. The reference is
the decimal determinant of the float64 matrix that results, not the product of
the requested eigenvalues.

```{code-cell} ipython3
def conditioned(ndim, log_ratio, n, rng):
    """`n` symmetric matrices with a target eigenvalue ratio of 10**log_ratio."""
    mats = np.empty((n, ndim, ndim))
    actual = np.empty(n)
    for i in range(n):
        Q, _ = np.linalg.qr(rng.normal(size=(ndim, ndim)))
        ev = np.ones(ndim)
        ev[-1] = 10.0**log_ratio
        M = Q @ np.diag(ev) @ Q.T
        M = (M + M.T) / 2
        mats[i] = M
        evs = np.linalg.eigvalsh(M)
        actual[i] = np.min(np.abs(evs)) / np.max(np.abs(evs))
    return mats, actual


rows = []
for ndim in (2, 3):
    for log_ratio in (0, -4, -8, -12, -16):
        mats, actual = conditioned(ndim, log_ratio, n=24, rng=rng)
        if ndim == 2:
            elems = [mats[:, 0, 0], mats[:, 0, 1], mats[:, 1, 1]]
        else:
            elems = [mats[:, 0, 0], mats[:, 0, 1], mats[:, 0, 2],
                     mats[:, 1, 1], mats[:, 1, 2], mats[:, 2, 2]]
        closed = closed_det(elems)
        linear = np.linalg.det(mats)
        scale = np.max(np.abs(mats), axis=(1, 2))**ndim
        refs = np.empty(len(mats))
        for i in range(len(mats)):
            entry = [float(e[i]) for e in elems]
            refs[i] = float(decimal_det(entry))
        for method, got in (("closed form", closed), ("np.linalg.det", linear)):
            abs_err = np.abs(got - refs)
            rows.append({
                "order": ndim,
                "target ratio": 10.0**log_ratio,
                "median actual ratio": float(np.median(actual)),
                "method": method,
                "max scaled error": float(np.max(abs_err / scale)),
                "max relative error": float(np.max(
                    abs_err / np.maximum(np.abs(refs), 1e-300))),
            })
ill = pd.DataFrame(rows)
show_table(ill, index=["order", "target ratio", "method"], floatfmt=".3g")
```

The scaled error stays near $10^{-16}$ as the target ratio falls. The relative
error grows in both methods. At a target ratio of $10^{-16}$, the 3 × 3
closed form has a larger relative error than `np.linalg.det` in this sample;
the 2 × 2 pair does not separate in the same way. The camera condition proxy
in §5 is the largest $s^{n} / |{\det}|$ on that image. It is not $10^{16}$.

## 8. Timings

The clock starts with the upper triangle already computed, and stops when the
determinant array is returned. "Current" is `_symmetric_image` followed by
`np.linalg.det`, which is the tail of `hessian_matrix_det`. "Assemble" is
`_symmetric_image` alone. "`np.linalg.det`" is the factorization of a stack
that is already assembled. Each time is the shortest of three calls after one
warmup. The elements come from a random image at `sigma` 1; the arithmetic
does not branch on the values.

```{code-cell} ipython3
def time_det(image):
    elems = hessian_matrix(image, sigma=1, use_gaussian_derivatives=False)
    full = _symmetric_image(elems)
    return {
        "assemble": best_time(lambda: _symmetric_image(elems)),
        "np.linalg.det": best_time(lambda: np.linalg.det(full)),
        "current": best_time(lambda: np.linalg.det(_symmetric_image(elems))),
        "closed form": best_time(lambda: closed_det(elems)),
    }


timing_rows = []
for side in (128, 256, 512, 1024):
    row = {"order": 2, "side": side, "pixels": side**2}
    row.update(time_det(rng.random((side, side))))
    timing_rows.append(row)
for side in (32, 48, 64):
    row = {"order": 3, "side": side, "pixels": side**3}
    row.update(time_det(rng.random((side, side, side))))
    timing_rows.append(row)
timings = pd.DataFrame(timing_rows)
timings["current / closed form"] = timings["current"] / timings["closed form"]
show_table(timings, floatfmt=".3g")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
for ax, order, title in zip(axes, (2, 3), ("2D", "3D")):
    part = timings[timings.order == order]
    for name in ("current", "closed form"):
        ax.plot(part["side"], part[name], "o-", lw=2, ms=5,
                color=COLOURS[name], label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(part["side"], [str(s) for s in part["side"]])
    ax.set_xlabel("side length, pixels")
    ax.set_title(title)
    ax.grid(color=GRID, lw=0.6)
    last = part.iloc[-1]
    speedup = last["current"] / last["closed form"]
    ax.annotate(f"{speedup:.0f}×",
                (last["side"], last["closed form"]),
                textcoords="offset points", xytext=(-8, 6),
                color=INK, ha="right", va="bottom")
axes[0].set_ylabel("time, ms")
axes[0].legend(frameon=False)
plt.show()
print(timings.loc[timings.groupby("order")["side"].idxmax(),
                  ["order", "side", "current / closed form"]]
      .to_string(index=False))
```

Assembly is a small part of the current tail. The time is in
`np.linalg.det`. The closed form is the same polynomial, evaluated with array
multiplies, so its time follows the number of pixels. The label on each panel
is the speed-up at the largest side length in that panel.

## 9. The call in `hessian_matrix_det`

The determinant is not the whole function. `hessian_matrix` still smooths the
image and takes the second differences. The replacement is that Hessian plus
`closed_det`. For a 2D image the comparison passes `approximate=False`, because
the default 2D path is the box filters, which are a different quantity. The
times below are at `sigma` 1.

```{code-cell} ipython3
def full_call(image):
    kwargs = {"approximate": False} if image.ndim == 2 else {}
    return {
        "hessian_matrix_det": best_time(
            lambda: hessian_matrix_det(image, sigma=1, **kwargs)),
        "Hessian + closed form": best_time(
            lambda: closed_det(hessian_matrix(
                image, sigma=1, use_gaussian_derivatives=False))),
    }


rows = []
for image, order, side in (
        (rng.random((512, 512)), 2, 512),
        (rng.random((64, 64, 64)), 3, 64)):
    row = {"order": order, "side": side, "sigma": 1}
    row.update(full_call(image))
    rows.append(row)
full_times = pd.DataFrame(rows)
full_times["hessian_matrix_det / replacement"] = (
    full_times["hessian_matrix_det"] / full_times["Hessian + closed form"])
show_table(full_times, floatfmt=".3g")
```

```{code-cell} ipython3
tiny = np.zeros((8, 8, 8), np.float32)
tiny[3, 3, 3] = 1
out = hessian_matrix_det(tiny, sigma=1)
print("3D float32 output dtype", out.dtype)
print("closed_det dtype", closed_det(
    hessian_matrix(tiny, sigma=1, use_gaussian_derivatives=False)).dtype)
print("max |det(approximate=True) - det(approximate=False)| on 3D",
      np.max(np.abs(hessian_matrix_det(tiny, sigma=1, approximate=True)
                    - hessian_matrix_det(tiny, sigma=1, approximate=False))))
```

The printed difference is 0, so on this volume `approximate` does not change
the result. The printed dtypes are float32. The ratio in the table above is
smaller than the tail speed-up in §8, because both calls compute the Hessian.

## 10. What not to do

The product of the eigenvalues is also the determinant. `eigvalsh` is the
routine `_symmetric_compute_eigenvalues` uses for a matrix larger than 2 × 2.

```{code-cell} ipython3
int_eigs = np.prod(np.linalg.eigvalsh(int_mats), axis=-1)
err = np.abs(int_eigs - truth_f)
print("integer corpus, product of eigenvalues")
print("max absolute error", err.max())
print("matrices with error > 0.5", int(np.sum(err > 0.5)))

eig_rows = []
for side, image in ((128, rng.random((128, 128))),
                    (32, rng.random((32, 32, 32)))):
    elems = hessian_matrix(image, sigma=1, use_gaussian_derivatives=False)
    full = _symmetric_image(elems)
    eig_rows.append({
        "order": image.ndim, "side": side,
        "eigvalsh product": best_time(
            lambda: np.prod(np.linalg.eigvalsh(full), axis=-1)),
        "closed form": best_time(lambda: closed_det(elems)),
    })
show_table(pd.DataFrame(eig_rows), floatfmt=".3g")
```

The printed count of matrices with absolute error above 0.5 is 0, so the
eigenvalue product rounds to the same integer as the exact determinant on
this corpus. The table is the time. `eigvalsh` is slower than the closed form
at both sizes.

Do not evaluate the 3 × 3 polynomial in float32. §6 counts the matrices where
that result differs from `np.linalg.det`. The upcast is the accurate match to
the current function, and it is still an array multiply.

Do not read this polynomial as the box-filter determinant. That quantity is
$D_{xx} D_{yy} - 0.81 D_{xy}^{2}$ on an integral image.
`on_hessian_matrix_det.md` and `on_hessian_det_approx.md` measure it. It is not
a shortcut for `np.linalg.det`.

## 11. The range on three photographs

The closed form and `np.linalg.det` run on the same finite-difference Hessian.
The difference is the closed form minus `np.linalg.det`. The peak is the
largest absolute value of the closed form for that image and `sigma`. The
interior leaves out a border of width $\lfloor 8\sigma + 0.5 \rfloor + 1$, the
reach of an 8-sigma kernel. The photographs are `camera`, `coins`, and
greyscale `astronaut`. `sigma` is 1, 2, 4 and 8.

```{code-cell} ipython3
def formula_gap(image, sigma):
    """Closed form, its difference from `np.linalg.det`, and the interior mask."""
    elems = hessian_matrix(image, sigma=sigma, use_gaussian_derivatives=False)
    closed = np.asarray(closed_det(elems), dtype=np.float64)
    linear = np.linalg.det(_symmetric_image(elems))
    diff = closed - linear
    reach = int(8 * sigma + 0.5) + 1
    inner = np.zeros(image.shape, dtype=bool)
    inner[reach:-reach, reach:-reach] = True
    return closed, diff, inner


PHOTO_SIGMAS = (1, 2, 4, 8)
photographs = {
    "camera": ski.util.img_as_float(ski.data.camera()),
    "coins": ski.util.img_as_float(ski.data.coins()),
    "astronaut": ski.util.img_as_float(ski.color.rgb2gray(ski.data.astronaut())),
}
print({name: image.shape for name, image in photographs.items()})
```

```{code-cell} ipython3
photo_rows = []
for name, image in photographs.items():
    for sigma in PHOTO_SIGMAS:
        closed, diff, inner = formula_gap(image, sigma)
        peak = np.max(np.abs(closed))
        space = np.spacing(np.where(closed == 0, np.finfo(float).tiny, closed))
        ulps = np.abs(diff) / space
        for region, sel in (("interior", inner), ("border", ~inner)):
            dd, uu, cc = diff[sel], ulps[sel], closed[sel]
            above_1e6 = np.abs(cc) > 1e-6 * peak
            above_1e3 = np.abs(cc) > 1e-3 * peak
            photo_rows.append({
                "image": name,
                "sigma": sigma,
                "region": region,
                "min diff": dd.min(),
                "max diff": dd.max(),
                "max abs diff": np.max(np.abs(dd)),
                "max abs diff / peak": np.max(np.abs(dd)) / peak,
                "fraction differing": np.mean(dd != 0),
                "p99 ulps": np.percentile(uu, 99),
                "max ulps": uu.max(),
                "max ulps, abs det > 1e-6 peak": uu[above_1e6].max(),
                "max ulps, abs det > 1e-3 peak": uu[above_1e3].max(),
            })
photo_gaps = pd.DataFrame(photo_rows)
interior_gaps = photo_gaps[photo_gaps.region == "interior"]
show_table(
    interior_gaps[["image", "sigma", "min diff", "max diff",
                   "max abs diff", "max abs diff / peak", "fraction differing"]],
    floatfmt=".3e",
)
```

```{code-cell} ipython3
span = interior_gaps.assign(abs_span=interior_gaps["max diff"] - interior_gaps["min diff"])
widest = span.loc[span.abs_span.idxmax()]
narrowest = span.loc[span.abs_span.idxmin()]
rel = interior_gaps["max abs diff / peak"]
print("widest signed span:", widest["image"], "sigma", int(widest["sigma"]),
      f"{widest['min diff']:.3e}", "to", f"{widest['max diff']:.3e}")
print("narrowest signed span:", narrowest["image"], "sigma", int(narrowest["sigma"]),
      f"{narrowest['min diff']:.3e}", "to", f"{narrowest['max diff']:.3e}")
print("interior max abs diff / peak, over 3 images and 4 sigmas:",
      f"{rel.min():.3e}", "to", f"{rel.max():.3e}")
print("fraction differing, interior:",
      f"{interior_gaps['fraction differing'].min():.3f}",
      "to", f"{interior_gaps['fraction differing'].max():.3f}")
print("every interior row has a negative minimum and a positive maximum:",
      bool(np.all(interior_gaps["min diff"] < 0)
           and np.all(interior_gaps["max diff"] > 0)))
growth = interior_gaps.groupby("image")["max abs diff / peak"].agg(
    lambda s: s.iloc[-1] / s.iloc[0])
spread = interior_gaps.groupby("sigma")["max abs diff / peak"].agg(
    lambda s: s.max() / s.min())
print("growth, sigma 1 to 8, per image:", f"{growth.min():.2f}", "to",
      f"{growth.max():.2f}")
print("largest / smallest image at fixed sigma:", f"{spread.min():.2f}", "to",
      f"{spread.max():.2f}")
```

Every interior row has a negative minimum and a positive maximum, so the
signed difference changes sign on each image and each `sigma`. The widest
and narrowest spans are the first two lines. The third line is the range of
the largest gap relative to the peak. The fourth is the rise from `sigma` 1
to `sigma` 8 on each image, and the fifth is the ratio between the largest
and smallest image at a fixed `sigma`.

```{code-cell} ipython3
show_table(
    interior_gaps[["image", "sigma", "p99 ulps", "max ulps",
                   "max ulps, abs det > 1e-6 peak",
                   "max ulps, abs det > 1e-3 peak"]],
    floatfmt=".3g",
)
```

An ulp here is $|D_{\mathrm{closed}} - D_{\mathrm{linalg}}|$ divided by the
spacing of the closed-form value. The 99th percentile is the `p99 ulps`
column: 13 to 22 ulps. The unrestricted maximum reaches $1.3 \times 10^5$
ulps, at pixels where the determinant is close to zero. The last two columns
keep only pixels whose absolute determinant exceeds $10^{-6}$ and $10^{-3}$ of
the peak. At $10^{-6}$ of the peak the maximum is still as large as
$2.3 \times 10^4$ ulps; at $10^{-3}$ of the peak it falls to 58 to 319 ulps.

```{code-cell} ipython3
border_rel = photo_gaps[photo_gaps.region == "border"]["max abs diff / peak"]
print("border max abs diff / peak:", f"{border_rel.min():.3e}", "to",
      f"{border_rel.max():.3e}")
print("border maximum / interior maximum:", f"{border_rel.max() / rel.max():.3f}")
```

The border does not widen the range much. Its largest gap relative to the peak
is $1.7 \times 10^{-16}$ to $1.5 \times 10^{-15}$, and its overall maximum is
within a few percent of the interior maximum.

```{code-cell} ipython3
# Same three colours as COLOURS, one image each. validate_palette.js, light
# mode, as recorded in §1: all checks pass; worst adjacent CVD ΔE 11.8;
# contrast WARN, so the lines are labelled.
IMAGE_COLOURS = {
    "camera": COLOURS["current"],
    "coins": COLOURS["closed form"],
    "astronaut": COLOURS["assemble"],
}
fig, ax = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
for name, colour in IMAGE_COLOURS.items():
    part = interior_gaps[interior_gaps.image == name]
    ax.plot(part["sigma"], part["max abs diff / peak"], "o-", color=colour,
            lw=2, ms=5, label=name)
ax.legend(frameon=False, loc="upper left")
ax.set_yscale("log")
ax.set_xticks(PHOTO_SIGMAS, [str(s) for s in PHOTO_SIGMAS])
ax.set_xlim(0.7, 11)
ax.set_xlabel("sigma")
ax.set_ylabel("max |difference| / peak")
ax.set_title("interior")
ax.grid(color=GRID, lw=0.6)
plt.show()
```

The relative gap lies between $3.7 \times 10^{-16}$ and $1.5 \times 10^{-15}$.
It rises by a factor of 2.5 to 3.5 from `sigma` 1 to `sigma` 8, and at a fixed
`sigma` the three images are within a factor of two of each other.

## 12. Summary

```{code-cell} ipython3
summary = pd.DataFrame([
    {"question": "identity, 2×2 and 3×3, sympy gaps",
     "result": f"{sym_gap2}, {sym_gap3}"},
    {"question": "integer corpus, closed form, max abs error",
     "result": f"{np.max(np.abs(int_closed - truth_f)):.3g}"},
    {"question": "integer corpus, np.linalg.det, max abs error",
     "result": f"{np.max(np.abs(int_linear - truth_f)):.3g}"},
    {"question": "outer products, closed form, nonzero results",
     "result": f"{int(np.sum(closed0 != 0))} of {N_SING}"},
    {"question": "float32 sample, cast-back values differing",
     "result": f"{int(np.sum(cast_back != linear32))} of {N_F32}"},
    {"question": "2D, largest side, current / closed form",
     "result": f"{timings[(timings.order == 2)].iloc[-1]['current / closed form']:.3g}"},
    {"question": "3D, largest side, current / closed form",
     "result": f"{timings[(timings.order == 3)].iloc[-1]['current / closed form']:.3g}"},
    {"question": "full call, sigma 1, 512 squared, ratio",
     "result": f"{full_times.loc[full_times.order == 2, 'hessian_matrix_det / replacement'].iloc[0]:.3g}"},
    {"question": "full call, sigma 1, 64 cubed, ratio",
     "result": f"{full_times.loc[full_times.order == 3, 'hessian_matrix_det / replacement'].iloc[0]:.3g}"},
    {"question": "three photographs, max abs diff / peak",
     "result": f"{rel.min():.3g} to {rel.max():.3g}"},
])
show_table(summary)
```

The replacement for the `np.linalg.det` tail, in 2D and 3D, is `closed_det`
on the list `hessian_matrix` already returns. It is the determinant. On the
integer corpus it is exact, where `np.linalg.det` is not. On the float32
sample it matches `np.linalg.det` after the same upcast. The summary table
has the speed-up of the determinant tail, and the speed-up of the full call
at `sigma` 1. The full call keeps the cost of `hessian_matrix`.

### Limits

* The integer corpus is 4000 matrices with entries in $-40, \ldots, 40$, one
  seed. The singular test is 2000 outer products with entries in
  $-20, \ldots, 20$. The float32 sample is 2000 normal matrices. None of these
  is a sample of image Hessians.
* The image sample is 40 pixels of `camera` and 40 pixels of one 40³ sphere,
  both at `sigma` 1. Twenty of each 40 are the pixels with the smallest
  $|{\det}| / s^{n}$.
* The ill-conditioned sweep is 24 matrices at each target ratio. The actual
  ratio after symmetrization is the median column of that table, not the
  target.
* Timings are one process, the shortest of three calls after one warmup, on
  the machine printed in §1. Sides are 128, 256, 512 and 1024 in 2D, and 32,
  48 and 64 in 3D. The full call is timed only at 512² and 64³, and only at
  `sigma` 1.
* The photograph comparison is `camera`, `coins`, and greyscale `astronaut`,
  at `sigma` 1, 2, 4 and 8. The Hessian is the finite-difference Hessian.
  The two formulas are not compared on a 3D image.
* There is no 4D formula here. Above 3D, `np.linalg.det` remains the path.
* The installed scikit-image is the one printed in §1. The source that was
  read is `hessian-refactor` at `060a313d1`.
