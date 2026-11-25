---
title: 'Sum-of-Gaussians Test Images'
date: 2026-11-17
options:
  updated: 2026-11-18
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

:::{changelog}
2026-11-18 | claude-code:claude-opus-5-5 | Fit the sums at run time, in place of stored files
2026-11-17 | claude-code:claude-opus-5-5 | Add §7, the sums as a gold standard for the Hessian and the shape index
2026-11-17 | claude-code:claude-opus-5-5 | Create workbook
:::

Three skimage images, `camera`, `coins` and `astronaut` in greyscale, each
approximated by a sum of isotropic Gaussians at five widths. Smoothing a
Gaussian gives a Gaussian, so each sum has an exact smoothed image and an exact
Hessian at every `sigma`. That makes the sums usable as gold-standard test
images. This notebook shows the sums beside the originals, and what each width
contributes.

Coordinates are in skimage `rc` order: `row` down the image, `col` across it,
with pixel centres at integers.

## Setup

```{code-cell} ipython3
from itertools import combinations_with_replacement

import numpy as np
import scipy.ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

from nbhelper import show_table
```

```{code-cell} ipython3
# The subject: the fitting module.
import gaussian_sum as gs
```

```{code-cell} ipython3
# The Hessian routes the sums score.
import skimage as ski
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
```

```{code-cell} ipython3
# Slots 1 and 2 of the reference categorical palette, as in `on_shape_index.md`.
# Validated all-pairs: CVD dE 24.7 (protan), 32.7 (tritan), normal-vision 33.6.
C_ONE, C_TWO = "#2a78d6", "#eb6834"
MUTED, RULE = "#52514e", "#dedcd5"

# Diverging, for signed quantities: two hues with a neutral midpoint.
DIV = LinearSegmentedColormap.from_list("div", [C_ONE, "#f2f1ec", C_TWO])

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "figure.facecolor": "white"}
)
```

## Helpers

```{code-cell} ipython3
def bare(ax, title=None):
    """Strip an image axes down to the pixels."""
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def signed(ax, data, limit, title):
    """Show signed data on the diverging map, symmetric about zero."""
    return bare(ax, title).imshow(data, cmap=DIV, norm=TwoSlopeNorm(0, -limit, limit))


def grey(ax, data, title):
    """Show an image on the 0 to 1 grey scale."""
    return bare(ax, title).imshow(data, cmap="gray", vmin=0, vmax=1)


IMAGES = {
    "camera": lambda: ski.data.camera(),
    "coins": lambda: ski.data.coins(),
    "astronaut": lambda: ski.color.rgb2gray(ski.data.astronaut()),
}


def original(name):
    """The source image, as float in 0 to 1."""
    return ski.util.img_as_float(IMAGES[name]())


def fit_sum(name):
    """Fit the sum of Gaussians to one image; return its levels and shape."""
    image = original(name)
    return gs.fit(image)[0], image.shape


SUMS = {name: fit_sum(name) for name in IMAGES}
```

## 1. The model

Each sum is

$$
f(r, c) = \sum_k h_k \exp\left(-\frac{(r - r_k)^2 + (c - c_k)^2}{2 w_k^2}\right),
$$

with $h_k$ the peak height and $w_k$ the width, the standard deviation, of
Gaussian $k$. The Gaussians of one width form a level. Their centres lie on a
grid whose spacing equals the width, and the grid reaches one spacing past each
image edge. The levels are fitted coarse to fine, each to the residual that the
coarser levels leave, by least squares with a small ridge.

Smoothing by a Gaussian of scale $\sigma$ gives the same sum, with width
$\sqrt{w_k^2 + \sigma^2}$ and height $h_k w_k^2 / (w_k^2 + \sigma^2)$.

As a table, with one Gaussian per row, the first rows for `camera` are:

```{code-cell} ipython3
level = SUMS["camera"][0][0]
rows, cols = np.meshgrid(level.row_centres, level.col_centres, indexing="ij")
show_table(pd.DataFrame({"row": rows.ravel(), "col": cols.ravel(),
                         "sigma": level.width, "height": level.heights.ravel()}).head())
```

The levels of each sum:

```{code-cell} ipython3
def level_rows(name):
    """Count and height range of each level of one sum."""
    levels, shape = SUMS[name]
    return [{"image": name, "width": lv.width,
             "grid": f"{lv.heights.shape[0]} x {lv.heights.shape[1]}",
             "Gaussians": lv.heights.size,
             "min height": f"{lv.heights.min():+.2f}",
             "max height": f"{lv.heights.max():+.2f}"} for lv in levels]


show_table(pd.DataFrame([r for name in IMAGES for r in level_rows(name)]),
           index="image")
```

+++

## 2. Originals and sums

```{code-cell} ipython3
def fit_quality(name):
    """How close the sum is to the original."""
    levels, shape = SUMS[name]
    error = gs.render(levels, shape) - original(name)
    rms = np.sqrt(np.mean(error**2))
    return {"image": name, "shape": f"{shape[0]} x {shape[1]}",
            "Gaussians": sum(lv.heights.size for lv in levels),
            "rms error": f"{rms:.4f}", "PSNR": f"{20 * np.log10(1 / rms):.1f} dB",
            "max |error|": f"{np.abs(error).max():.3f}"}


show_table(pd.DataFrame([fit_quality(n) for n in IMAGES]), index="image")
```

Each row shows the original, the sum, and their difference. The difference
uses one colour scale for all three images.

```{code-cell} ipython3
LIMIT = 0.2
fig, axes = plt.subplots(3, 3, figsize=(9, 8.6), layout="constrained")
for row, name in zip(axes, IMAGES):
    levels, shape = SUMS[name]
    model = gs.render(levels, shape)
    grey(row[0], original(name), f"{name}, original")
    grey(row[1], model, "sum of Gaussians")
    shown = signed(row[2], model - original(name), LIMIT, "sum - original")
fig.colorbar(shown, ax=axes[:, 2], fraction=0.05, shrink=0.6).outline.set_visible(False)
```

The difference sits on edges and fine texture: the coin rims and faces, the
grass and the tripod legs in `camera`, the flag and the suit in `astronaut`.
Smooth regions, such as the sky in `camera`, match closely.

+++

## 3. A close-up

The face and the camera in `camera`, pixel by pixel:

```{code-cell} ipython3
CROP = (slice(120, 216), slice(216, 312))

levels, shape = SUMS["camera"]
model = gs.render(levels, shape)
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
grey(axes[0], original("camera")[CROP], "original")
grey(axes[1], model[CROP], "sum of Gaussians")
shown = signed(axes[2], (model - original("camera"))[CROP], LIMIT, "sum - original")
fig.colorbar(shown, ax=axes[2], fraction=0.046).outline.set_visible(False)
fig.tight_layout()
```

The sum is slightly softer than the original. Its finest Gaussians have a width
of 1.5 pixels, so it cannot hold a step that is sharper than that. Along each
step the difference alternates in sign, parallel to the edge: the least-squares
fit overshoots on both sides, as a truncated Fourier series does at a jump.

+++

## 4. What each width contributes

Render one level at a time. The top row adds the levels from coarse to fine.
The bottom row shows the contribution of each level alone, on a signed scale
set by that level's largest values.

```{code-cell} ipython3
def show_levels(name, crop=(slice(None), slice(None))):
    """Cumulative sums, and single-level contributions, coarse to fine."""
    levels, shape = SUMS[name]
    fig, axes = plt.subplots(2, len(levels), figsize=(2.0 * len(levels), 4.8),
                             layout="constrained")
    total = np.zeros(shape)
    for k, level in enumerate(levels):
        part = gs.render([level], shape)
        total += part
        grey(axes[0, k], total[crop], f"widths >= {level.width:g}")
        limit = np.percentile(np.abs(part[crop]), 99.5)
        signed(axes[1, k], part[crop], limit, f"width {level.width:g} alone")
    fig.suptitle(name, color=MUTED, fontsize=9)


show_levels("camera")
```

```{code-cell} ipython3
show_levels("coins")
```

```{code-cell} ipython3
show_levels("astronaut")
```

The width-16 level carries the overall brightness. Each finer level adds
detail at its own scale, and is signed: it adds light on one side of an edge
and removes it on the other.

The contribution of each level, as a fraction of the image:

```{code-cell} ipython3
def contribution_rows(name):
    """RMS of each level's rendered contribution, over the RMS of the image."""
    levels, shape = SUMS[name]
    scale = np.sqrt(np.mean(original(name) ** 2))
    return {"image": name,
            **{f"width {lv.width:g}":
               f"{np.sqrt(np.mean(gs.render([lv], shape) ** 2)) / scale:.3f}"
               for lv in levels}}


show_table(pd.DataFrame([contribution_rows(n) for n in IMAGES]), index="image")
```

+++

## 5. The Gaussians themselves

The centres of three levels over the close-up of `camera`. Each dot is one
Gaussian. Its area follows the absolute height. Orange is a positive height,
blue a negative one.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), layout="constrained")
for ax, level in zip(axes, [lv for lv in levels if lv.width in (8.0, 4.0, 2.0)]):
    ax.imshow(original("camera")[CROP], cmap="gray", vmin=0, vmax=1, alpha=0.35,
              extent=(CROP[1].start - 0.5, CROP[1].stop - 0.5,
                      CROP[0].stop - 0.5, CROP[0].start - 0.5))
    r, c = np.meshgrid(level.row_centres, level.col_centres, indexing="ij")
    inside = ((r >= CROP[0].start) & (r < CROP[0].stop)
              & (c >= CROP[1].start) & (c < CROP[1].stop))
    h = level.heights[inside]
    ax.scatter(c[inside], r[inside], s=60 * np.abs(h) / np.abs(h).max(),
               c=np.where(h > 0, C_TWO, C_ONE), linewidths=0)
    ax.set_xlim(CROP[1].start - 0.5, CROP[1].stop - 0.5)
    ax.set_ylim(CROP[0].stop - 0.5, CROP[0].start - 0.5)
    bare(ax, f"width {level.width:g}: {inside.sum()} Gaussians")
```

The large heights at the finer levels follow the edges: the outline of the
face, the camera body, and the hand.

+++

## 6. Exact smoothing

The sum smoothed by `sigma` is exact, and so is its Hessian. The smoothed
`coins` sum, and its exact $H_{rr}$, at three scales:

```{code-cell} ipython3
levels, shape = SUMS["coins"]
fig, axes = plt.subplots(2, 3, figsize=(9, 5.4))
for k, sigma in enumerate((0.5, 2.0, 8.0)):
    grey(axes[0, k], gs.render(levels, shape, sigma), f"smoothed, sigma = {sigma:g}")
    h_rr = gs.hessian(levels, shape, sigma)[0]
    signed(axes[1, k], h_rr, np.percentile(np.abs(h_rr), 99.5), f"exact H_rr, sigma = {sigma:g}")
fig.tight_layout()
```

+++

## 7. As a gold standard

### 7.1 Four Hessians against the truth

The sums score four Hessian routes:

- the shipped finite differences, `hessian_matrix(use_gaussian_derivatives=False)`,
  which `shape_index` uses;
- the installed Gaussian derivatives, `hessian_matrix(use_gaussian_derivatives=True)`;
- the moment-corrected one-pass kernels of the `change-hessian-default` branch,
  transcribed as in `hessian_testing.md`;
- a spectral reference, defined below.

```{code-cell} ipython3
def taps(sigma, order, truncate=8):
    """1-D Gaussian derivative kernel, with its discrete moments restored."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    if order == 0:
        return g
    if order == 1:
        k = g * (x / sigma**2)
        return k / (k * x).sum()
    k = g * ((x**2 - sigma**2) / sigma**4)
    k = k - k.sum() * g
    k[len(k) // 2] -= k.sum()
    return k / ((k * x**2).sum() / 2)


def one_pass(image, sigma, mode="nearest"):
    """One separable pass per Hessian element, with the corrected kernels."""
    kernels = {o: taps(sigma, o) for o in (0, 1, 2)}
    elements = []
    for ax0, ax1 in combinations_with_replacement(range(2), 2):
        orders = [0, 0]
        orders[ax0] += 1
        orders[ax1] += 1
        result = np.asarray(image, dtype=float)
        for axis, o in enumerate(orders):
            result = ndi.correlate1d(result, kernels[o], axis=axis, mode=mode)
        elements.append(result)
    return elements
```

The spectral reference treats the mirror-extended image as band-limited, as
Rey-Otero and Delbracio do for Gaussian smoothing
[@rey_otero2016computing_exact], and differentiates it exactly in the Fourier
domain. It is close to the best that any method working from the samples can
do.

```{code-cell} ipython3
def spectral_hessian(image, sigma):
    """Exact Hessian of the mirror-extended, band-limited interpolant of `image`."""
    M, N = image.shape
    extended = np.block([[image, image[:, ::-1]], [image[::-1, :], image[::-1, ::-1]]])
    spectrum = np.fft.fft2(extended)
    fr, fc = np.fft.fftfreq(2 * M)[:, None], np.fft.fftfreq(2 * N)[None, :]
    wr, wc = 2 * np.pi * fr, 2 * np.pi * fc
    smooth = np.exp(-sigma**2 * (wr**2 + wc**2) / 2)
    # An odd derivative has no real value at the Nyquist frequency.
    dr = np.where(np.abs(fr) == 0.5, 0, 1j * wr)
    dc = np.where(np.abs(fc) == 0.5, 0, 1j * wc)
    return [np.real(np.fft.ifft2(spectrum * smooth * d))[:M, :N]
            for d in ((1j * wr) ** 2, dr * dc, (1j * wc) ** 2)]


ROUTES = {
    "shipped FD": lambda im, s: hessian_matrix(im, s, order="rc",
                                               use_gaussian_derivatives=False),
    "installed GD": lambda im, s: hessian_matrix(im, s, order="rc",
                                                 use_gaussian_derivatives=True),
    "branch GD": one_pass,
    "spectral": spectral_hessian,
}
MARGIN = 60   # past every kernel's reach at sigma <= 4, and the spectral border
```

The score is the worst element error over the interior, relative to the
largest value of the true Hessian there:

```{code-cell} ipython3
TRUTH, COMPUTED = {}, {}
for name in IMAGES:
    levels, shape = SUMS[name]
    image = gs.render(levels, shape)
    for sigma in (0.5, 1.0, 2.0, 4.0):
        TRUTH[name, sigma] = gs.hessian(levels, shape, sigma)
        for route, func in ROUTES.items():
            COMPUTED[name, sigma, route] = func(image, sigma)


def hessian_error(H, truth, margin=MARGIN):
    """Worst element error over the interior, relative to the true peak there."""
    inner = (slice(margin, -margin),) * 2
    peak = max(np.abs(t[inner]).max() for t in truth)
    return max(np.abs(h[inner] - t[inner]).max() for h, t in zip(H, truth)) / peak


show_table(pd.DataFrame(
    [{"image": name, "sigma": sigma,
      **{route: f"{hessian_error(COMPUTED[name, sigma, route], TRUTH[name, sigma]):.1e}"
         for route in ROUTES}}
     for name in IMAGES for sigma in (0.5, 1.0, 2.0, 4.0)]), index=["image", "sigma"])
```

The sums separate the routes over twelve orders of magnitude, and the
three images agree:

- The shipped finite differences are 30 to 40% wrong at `sigma` 0.5 and 1. The
  error falls slowly, to about 5% at `sigma` 4.
- The installed Gaussian derivatives are 94% wrong at `sigma` 0.5 and about 2%
  at `sigma` 1, then reach $10^{-9}$ at `sigma` 2 and the rounding floor at 4.
- The branch kernels are 6 to 9% wrong at `sigma` 0.5 and about
  $3 \times 10^{-4}$ at `sigma` 1, then sit at the rounding floor, near
  $10^{-12}$, from `sigma` 2.
- The spectral reference is within about $10^{-5}$ at `sigma` 0.5 and
  $4 \times 10^{-7}$ at `sigma` 1.

### 7.2 The truth is reachable from the samples

The spectral reference does not continue a sum past the image edge, so its
error depends on the distance from the border:

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"image": name, "sigma": sigma,
      **{f"margin {m}": f"{hessian_error(COMPUTED[name, sigma, 'spectral'], TRUTH[name, sigma], m):.1e}"
         for m in (10, 20, 40, 60)}}
     for name in IMAGES for sigma in (0.5, 1.0)]), index=["image", "sigma"])
```

At `sigma` 0.5 the spectral error falls from about $4 \times 10^{-4}$ at 10
pixels from the border to about $10^{-5}$ at 60 pixels. So, away from the
border, the samples fix the truth to about $10^{-5}$ of the peak at `sigma`
0.5 and to below $10^{-6}$ at `sigma` 1. The finite kernels of §7.1 miss by
far more than that, so their errors are their own and not the sums'.

### 7.3 A score for the shape index

The shape index means little where the curvedness is small. The score below
weights each pixel's error by the true curvedness,
$c = \sqrt{(\lambda_1^2 + \lambda_2^2)/2}$, and also reports the median and
the 99th percentile over pixels whose curvedness is at least 1% of the maximum.
Both the truth and the routes use the shape index formula with the umbilic
resolved, as in `on_shape_index.md` §3, so the score measures the Hessian and
not that defect.

```{code-cell} ipython3
CURVEDNESS_FLOOR = 0.01


def s_from(H):
    """The shape index, with the umbilic at -1 or +1 and the planar point NaN."""
    l1, l2 = hessian_matrix_eigvals(H)
    s = -(2.0 / np.pi) * np.arctan2(l1 + l2, l1 - l2)
    return np.where((l1 == 0) & (l2 == 0), np.nan, s)


def curvedness(H):
    """Koenderink and van Doorn's curvedness of the Hessian eigenvalue pair."""
    l1, l2 = hessian_matrix_eigvals(H)
    return np.sqrt((l1**2 + l2**2) / 2)


def s_score(H, truth, margin=MARGIN):
    """Curvedness-weighted mean, median and 99th percentile of |s - s_true|."""
    inner = (slice(margin, -margin),) * 2
    c = curvedness(truth)[inner]
    ds = np.abs(s_from(H) - s_from(truth))[inner]
    ok = np.isfinite(ds)
    keep = ok & (c >= CURVEDNESS_FLOOR * c.max())
    return {"weighted": np.sum(c[ok] * ds[ok]) / np.sum(c[ok]),
            "median": np.median(ds[keep]), "p99": np.percentile(ds[keep], 99),
            "below floor": np.mean(c < CURVEDNESS_FLOOR * c.max())}


def score_row(name, sigma):
    """Shape index scores of the shipped and branch routes on one sum."""
    truth = TRUTH[name, sigma]
    inner = (slice(MARGIN, -MARGIN),) * 2
    l1, l2 = (e[inner] for e in hessian_matrix_eigvals(truth))
    row = {"image": name, "sigma": sigma,
           "below floor": f"{100 * s_score(truth, truth)['below floor']:.0f}%",
           "exact ties": int(np.sum((l1 == l2) & (l1 != 0)))}
    for route in ("shipped FD", "branch GD"):
        score = s_score(COMPUTED[name, sigma, route], truth)
        row[f"{route}, weighted"] = f"{score['weighted']:.1e}"
        row[f"{route}, p99"] = f"{score['p99']:.1e}"
    return row


show_table(pd.DataFrame([score_row(n, s) for n in IMAGES for s in (0.5, 1.0, 2.0)]),
           index=["image", "sigma"])
```

No pixel of the truth has an exact tie, so the sums do not test the
umbilic defect; that needs a symmetric case of its own. Between 4% and 32% of
the interior falls below the curvedness floor, most in `camera` and
`astronaut`, which have sky and smooth background.

The shipped route has a curvedness-weighted error of 0.02 to 0.07. At `sigma`
1 and below, its 99th percentile above the floor is 0.7 to 0.9, which is about
three of the docstring's class widths of 0.25. The branch route is about
$5 \times 10^{-5}$ at `sigma` 1 and at the rounding floor at `sigma` 2.

Where the errors lie on `camera`:

```{code-cell} ipython3
name, sigma = "camera", 1.0
truth = TRUTH[name, sigma]
inner = (slice(MARGIN, -MARGIN),) * 2
fig, axes = plt.subplots(1, 3, figsize=(9, 3.3), layout="constrained")
c = curvedness(truth)[inner]
bare(axes[0], "true curvedness").imshow(c, cmap="Greys", vmin=0,
                                        vmax=np.percentile(c, 99))
for ax, route in zip(axes[1:], ("shipped FD", "branch GD")):
    ds = np.abs(s_from(COMPUTED[name, sigma, route]) - s_from(truth))[inner]
    shown = bare(ax, f"|s - s_true|, {route}").imshow(
        np.maximum(ds, 1e-6), cmap="Blues", norm=LogNorm(1e-6, 1))
fig.colorbar(shown, ax=axes[1:], fraction=0.03).outline.set_visible(False)
fig.suptitle(f"{name} sum, sigma = {sigma:g}", color=MUTED, fontsize=9)
```

The shipped error is largest in flat regions, and smaller along strong edges,
where the curvedness is high. The branch error is about four orders of
magnitude smaller over most of the frame, and rises a little along edges.

+++

## 8. Summary

| question | answer |
| --- | --- |
| what the sums are | sums of isotropic Gaussians at widths 16, 8, 4, 2 and 1.5 pixels (§1) |
| how close to the originals | 30 to 34 dB PSNR; the difference sits on edges and fine texture (§2) |
| what they cannot hold | steps sharper than the finest width of 1.5 pixels (§3) |
| how the widths divide the image | the coarsest carries the brightness; finer levels add signed detail at their own scale (§4) |
| what is exact | the smoothed image and its Hessian, at any `sigma` (§6) |
| how the Hessian routes score | shipped finite differences 30 to 40% wrong at `sigma` up to 1, about 5% at 4; branch kernels $3 \times 10^{-4}$ at 1 and at rounding from 2 (§7.1) |
| whether the truth can be reached | a spectral Hessian of the samples is within about $10^{-5}$ at `sigma` 0.5, 60 pixels from the border (§7.2) |
| how to score the shape index | weight by curvedness; shipped route 0.02 to 0.07 weighted, 99th percentile 0.7 to 0.9 at `sigma` up to 1; no exact ties (§7.3) |

**Limits.** Three images, one set of widths and one ridge strength. The
spectral reference assumes a mirror-extended, band-limited image, and needs a
margin of about 60 pixels. The routes keep their own default border modes;
the 60-pixel margin keeps the border out of every score. The scores use the
installed scikit-image, with the branch kernels transcribed. The
Gaussians are isotropic, so elongated structure is built from many round
Gaussians. The sums are test images in their own right. Their resemblance
to the photographs does not matter for testing, and a better fit would not
make them a better standard.

```{code-cell} ipython3
print(f"scikit-image  {ski.__version__}")
print(f"numpy         {np.__version__}")
```
