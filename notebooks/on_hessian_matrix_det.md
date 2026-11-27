---
title: 'On the Hessian determinant'
date: 2026-11-27
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

:::{related}
https://github.com/scikit-image/scikit-image/pull/8436 | Moment-corrected Gaussian-derivative Hessian and the Gaussian-sum gold standard; used in [§3](#sec-routes) and [§6](#sec-corpus)
:::

:::{changelog}
2026-11-27 | claude-code:claude-opus-5-5 | Create workbook
:::

[`skimage.feature.hessian_matrix_det`](https://scikit-image.org/docs/stable/api/skimage.feature.html)
has two routes for 2-D images. The default, `approximate=True`, uses the box
filters of SURF [@bay2006surf_speeded] on an integral image.
`approximate=False` computes the Hessian with `hessian_matrix`, by finite
differences, and takes the determinant of each 2 × 2 matrix. This notebook
measures the speed and the accuracy of both routes, and of a third route that
uses Gaussian derivatives. It asks whether `approximate=False` is a good
default.

Coordinates are in array order: $r$ runs down the rows, $c$ across the
columns.

## 1. Setup

```{code-cell} ipython3
from itertools import combinations_with_replacement
import time

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from nbhelper import show_table
```

The subject under test is the installed scikit-image. `gaussian_sum` fits the
gold-standard images of §6; `on_gaussian_sum_images.md` describes it.

```{code-cell} ipython3
import skimage as ski
from skimage.feature import hessian_matrix_det

import gaussian_sum as gs

print("scikit-image", ski.__version__)
```

```{code-cell} ipython3
# Categorical slots 1-4 of the default palette. validate_palette.js, light
# mode: all checks pass; worst adjacent CVD ΔE 11.8; contrast WARN for slots
# 2-4, so every chart has a legend, direct labels or a table.
ROUTE_COLOURS = {
    "box": "#257de0",
    "FD": "#f47230",
    "GD": "#00b894",
    "FD closed form": "#f2a600",
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

The finite-difference (FD) route smooths at scale `sigma` and then applies the
central difference twice. It is the route that `hessian_matrix_det(...,
approximate=False)` uses.

```{code-cell} ipython3
def fd_hessian(image, sigma, mode="constant", cval=0):
    """Smooth at sigma, then two central differences: `[Hrr, Hrc, Hcc]`."""
    smoothed = ski.filters.gaussian(image, sigma=sigma, mode=mode, cval=cval)
    gradients = np.gradient(smoothed)
    return [np.gradient(gradients[a], axis=b)
            for a, b in combinations_with_replacement(range(image.ndim), 2)]
```

The Gaussian-derivative (GD) route convolves once per element with a sampled
Gaussian-derivative kernel whose discrete moments are restored.
`on_shape_index.md` §4 explains the kernel. `taps` and `gd_hessian` transcribe
`_gaussian_derivative_kernel` and `_hessian_matrix_with_gaussian` on the
scikit-image `hessian-refactor` branch, at commit `8e5a7a287`. The kernels
agree to the last bit for `sigma` from 0.5 to 16 in steps of 0.25, and
`gd_hessian` gives arrays identical to `hessian_matrix(image, sigma,
mode="constant", use_gaussian_derivatives=True)` on that branch, on the fitted
camera of §6 at `sigma` 1, 2, 4 and 8. Those checks ran outside this notebook,
because the installed scikit-image does not have the branch.

```{code-cell} ipython3
def taps(sigma, order, truncate=8):
    """1-D Gaussian derivative kernel, with its discrete moments restored."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    smoothing = np.exp(-(x**2) / (2 * sigma**2))
    smoothing /= smoothing.sum()
    if order == 0:
        return smoothing
    if order == 1:
        kernel = smoothing * (x / sigma**2)
        return kernel / (kernel * x).sum()
    kernel = smoothing * ((x**2 - sigma**2) / sigma**4)
    kernel = kernel - kernel.sum() * smoothing
    center = len(kernel) // 2
    kernel[center] = -(kernel[:center].sum() + kernel[center + 1:].sum())
    kernel /= (kernel * x**2).sum() / 2
    kernel[center] = -(kernel[:center].sum() + kernel[center + 1:].sum())
    return kernel


def gd_hessian(image, sigma, mode="constant", cval=0, truncate=8):
    """One separable pass per element, with the corrected kernels."""
    image = np.asarray(image, dtype=float)
    ndim = image.ndim
    kernels = {(axis, o): taps(sigma, o, truncate)
               for axis in range(ndim) for o in (0, 1, 2)}
    elements = []
    for ax0, ax1 in combinations_with_replacement(range(ndim), 2):
        orders = [0] * ndim
        orders[ax0] += 1
        orders[ax1] += 1
        result = image
        for axis, o in enumerate(orders):
            result = ndi.correlate1d(result, kernels[(axis, o)], axis=axis,
                                     mode=mode, cval=cval)
        elements.append(result)
    return elements
```

For a symmetric 2 × 2 matrix the determinant is
$H_{rr} H_{cc} - H_{rc}^2$:

```{code-cell} ipython3
def det2(H):
    """Closed-form determinant from the upper-diagonal elements `[Hrr, Hrc, Hcc]`."""
    Hrr, Hrc, Hcc = H
    return Hrr * Hcc - Hrc**2
```

`box_sums` sums the image over one rectangle at an offset from every pixel,
with an [integral image](https://en.wikipedia.org/wiki/Summed-area_table).
Outside the image it counts zeros.

```{code-cell} ipython3
def box_sums(image, r0, c0, nr, nc):
    """Sum of `image[r+r0 : r+r0+nr, c+c0 : c+c0+nc]` at every pixel (r, c)."""
    H, W = image.shape
    pad = max(abs(r0), abs(c0), abs(r0 + nr), abs(c0 + nc)) + 1
    padded = np.pad(np.asarray(image, dtype=float), pad)
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1))
    integral[1:, 1:] = padded.cumsum(0).cumsum(1)
    R = np.arange(H)[:, None] + pad + r0
    C = np.arange(W)[None, :] + pad + c0
    return (integral[R + nr, C + nc] - integral[R, C + nc]
            - integral[R + nr, C] + integral[R, C])
```

`box_det` is a vectorised transcription of `_hessian_matrix_det` in
`skimage/feature/_hessian_det_appx_pythran.py`, the `approximate=True` route.
The code calls the second derivative across the columns `dxx` and the one down
the rows `dyy`. In the code, a box with top-left corner `(r, c)` covers rows
`r + 1` to `r + rl` and columns `c + 1` to `c + cl`; the `+ 1` below
reproduces that. `size` sets the box geometry, and `lobe = size // 3` is the
width of one lobe.

```{code-cell} ipython3
def box_terms(image, sigma):
    """The three box responses of `_hessian_matrix_det`: `(dyy, dxy, dxx)`."""
    size = int(3 * sigma)
    s2, lobe = (size - 1) // 2, size // 3
    weight = 1.0 / size / size

    def box(r0, c0, nr, nc):
        return box_sums(image, r0 + 1, c0 + 1, nr, nc)

    dxy = -(box(-lobe, 1, lobe, lobe) + box(1, -lobe, lobe, lobe)
            - box(-lobe, -lobe, lobe, lobe) - box(1, 1, lobe, lobe)) * weight
    dxx = -(box(-lobe + 1, -s2, 2 * lobe - 1, size)
            - 3 * box(-lobe + 1, -(lobe // 2), 2 * lobe - 1, lobe)) * weight
    dyy = -(box(-s2, -lobe + 1, size, 2 * lobe - 1)
            - 3 * box(-(lobe // 2), -lobe + 1, lobe, 2 * lobe - 1)) * weight
    return dyy, dxy, dxx


def box_det(image, sigma):
    """SURF determinant with skimage's box geometry: `dxx dyy - 0.81 dxy²`."""
    dyy, dxy, dxx = box_terms(image, sigma)
    return dxx * dyy - 0.81 * dxy**2
```

`interior` gives the pixels away from the border, and `det_error` compares a
determinant with the truth there.

```{code-cell} ipython3
def interior(sigma):
    """Pixels that an 8-sigma kernel reaches without crossing the frame."""
    reach = int(8 * sigma + 0.5) + 1
    return (slice(reach, -reach),) * 2


def det_error(got, truth, inner):
    """Relative RMS error, gain error and shape error of `got` against `truth`."""
    g, t = got[inner], truth[inner]
    norm2 = np.sum(t * t)
    gain = np.sum(g * t) / norm2
    rms = np.sqrt(np.sum((g - t) ** 2) / norm2)
    shape = np.sqrt(np.sum((g - gain * t) ** 2) / (gain**2 * norm2))
    return rms, gain - 1, shape


def best_time(func, repeats=5):
    """Shortest of `repeats` wall-clock times of `func()`, in milliseconds."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return 1e3 * min(times)
```

The three numbers from `det_error` are:

* the relative RMS error $E = \|D - T\| / \|T\|$, with $D$ the determinant
  under test and $T$ the truth, summed over the interior pixels;
* the gain error $g - 1$, where $g = \langle D, T\rangle / \|T\|^2$ is the one
  scale factor that best maps $T$ onto $D$;
* the shape error $S = \|D - g T\| / \|g T\|$, the error that no scale factor
  removes. $S$ is zero when $D$ is any positive or negative multiple of $T$.

The transcription of the box route must match the installed code:

```{code-cell} ipython3
rng = np.random.default_rng(0)
noise = rng.uniform(size=(96, 96))
rows = []
for sigma in (1, 2, 3, 4, 8):
    inner = (slice(int(3 * sigma) + 2, -int(3 * sigma) - 2),) * 2
    ours = box_det(noise, sigma)
    theirs = hessian_matrix_det(noise, sigma, approximate=True)
    scale = np.abs(theirs[inner]).max()
    border = np.ones(noise.shape, bool)
    border[inner] = False
    rows.append({"sigma": sigma,
                 "max difference / max value, interior":
                     np.abs(ours - theirs)[inner].max() / scale,
                 "max difference / max value, border":
                     np.abs(ours - theirs)[border].max() / scale})
show_table(pd.DataFrame(rows), floatfmt=".1e")
```

The two agree to rounding away from the border, and differ at the border. At
the border, the installed code moves each box corner to the nearest pixel
inside the image, where `box_sums` counts zeros outside the image. All
comparisons below use the interior only.

The FD route with the closed-form determinant must match `approximate=False`:

```{code-cell} ipython3
rows = []
for sigma in (1, 2, 4, 8):
    ours = det2(fd_hessian(noise, sigma))
    theirs = hessian_matrix_det(noise, sigma, approximate=False)
    rows.append({"sigma": sigma,
                 "max difference / max value":
                     np.abs(ours - theirs).max() / np.abs(theirs).max()})
show_table(pd.DataFrame(rows), floatfmt=".1e")
```

## 3. The problem on one blob

A test image of one isotropic Gaussian blob, of width $w$, has an exact
smoothed Hessian. Smoothing at scale $\sigma$ gives a Gaussian of width
$s = \sqrt{w^2 + \sigma^2}$ and peak $A = w^2 / s^2$
[@mobahi2016closed_form, §2.2]. Its second derivatives are

$$
H_{rr} = \Bigl(\frac{r^2}{s^4} - \frac{1}{s^2}\Bigr) G, \qquad
H_{rc} = \frac{r c}{s^4} G, \qquad
H_{cc} = \Bigl(\frac{c^2}{s^4} - \frac{1}{s^2}\Bigr) G,
$$

with $G = A \exp(-(r^2 + c^2) / (2 s^2))$ and $r, c$ measured from the blob
centre.

```{code-cell} ipython3
def blob(n, width):
    """`n` × `n` image of a unit-peak Gaussian of `width`, centred on a pixel."""
    r = np.arange(n)[:, None] - n // 2
    c = np.arange(n)[None, :] - n // 2
    return np.exp(-(r**2 + c**2) / (2 * width**2))


def blob_hessian(n, width, sigma):
    """Exact `[Hrr, Hrc, Hcc]` of `blob(n, width)` smoothed at `sigma`."""
    r = np.arange(n, dtype=float)[:, None] - n // 2
    c = np.arange(n, dtype=float)[None, :] - n // 2
    s2 = width**2 + sigma**2
    G = width**2 / s2 * np.exp(-(r**2 + c**2) / (2 * s2))
    return [(r**2 / s2**2 - 1 / s2) * G, r * c / s2**2 * G,
            (c**2 / s2**2 - 1 / s2) * G]
```

Check `blob_hessian` against GD, which is exact to rounding at `sigma` 2 and
above (§6):

```{code-cell} ipython3
N, WIDTH, SIGMA = 81, 8, 2
small = blob(N, WIDTH)
truth_small = det2(blob_hessian(N, WIDTH, SIGMA))
gd_small = det2(gd_hessian(small, SIGMA))
print("max |GD - exact| / max |exact|:",
      f"{np.abs(gd_small - truth_small).max() / np.abs(truth_small).max():.1e}")
```

The determinant is positive at the blob, where both curvatures have the same
sign. It is negative on a ring around the blob, where the image is a saddle.
The three routes, for a blob of width 8 at `sigma` 2:

```{code-cell} ipython3
small_dets = {
    "exact": truth_small,
    "box": box_det(small, SIGMA),
    "FD": det2(fd_hessian(small, SIGMA)),
    "GD": gd_small,
}
fig, axes = plt.subplots(1, 4, figsize=(9, 2.6), constrained_layout=True)
for ax, (name, D) in zip(axes, small_dets.items()):
    limit = np.abs(D).max()
    ax.imshow(D, cmap=DIVERGING, norm=TwoSlopeNorm(0, -limit, limit))
    peak = np.unravel_index(np.argmax(D), D.shape)
    ax.plot(peak[1], peak[0], "+", color=INK, ms=8)
    ax.plot(N // 2, N // 2, "o", mfc="none", mec=INK, ms=8)
    ax.set_title(f"{name}: peak at {tuple(int(p) - N // 2 for p in peak)}")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"Determinant, blob width {WIDTH}, sigma {SIGMA}. "
             "Circle: blob centre. Cross: maximum. Each panel has its own scale.",
             color=MUTED, fontsize=9)
plt.show()
```

The box result is not rotationally symmetric, and its maximum is away from
the blob centre. FD and GD look like the exact result. The scales differ:

```{code-cell} ipython3
inner_small = interior(SIGMA)
rows = []
for name in ("box", "FD", "GD"):
    D = small_dets[name]
    rms, gain_err, shape = det_error(D, truth_small, inner_small)
    rows.append({"route": name,
                 "value at centre / exact": D[N // 2, N // 2] / truth_small[N // 2, N // 2],
                 "E": rms, "gain error": gain_err, "shape error S": shape})
show_table(pd.DataFrame(rows), floatfmt=".3g")
```

(sec-routes)=
## 4. The three routes

### 4.1 The determinant of the Hessian

At each pixel, the Hessian of the smoothed image is the symmetric matrix

$$
H = \begin{pmatrix} H_{rr} & H_{rc} \\ H_{rc} & H_{cc} \end{pmatrix}.
$$

Its determinant $H_{rr} H_{cc} - H_{rc}^2$ is the product of the two
principal curvatures. It is large and positive at the centre of a bright or
dark blob of about the scale $\sigma$. The
[determinant of the Hessian](https://en.wikipedia.org/wiki/Blob_detection#The_determinant_of_the_Hessian)
is a standard blob detector [@lindeberg1998feature_detection]. A route that
is accurate for each element of $H$ is accurate for the determinant. But the
determinant multiplies two elements, so a gain error $\epsilon$ on each
element is close to a gain error $2\epsilon$ on the determinant.

### 4.2 Box filters

SURF replaces each Gaussian second derivative with a sum of constant boxes
[@bay2006surf_speeded]. An integral image gives the sum over any box in four
look-ups, so the cost per pixel does not depend on the box size. SURF uses
filters of size 9, 15, 21, 27 and so on: three lobes, each an odd number $l$ of
pixels wide. It weights the cross term by 0.9 and takes
$D_{xx} D_{yy} - (0.9 D_{xy})^2$. Bay and colleagues state that the 9 × 9
filters approximate Gaussian second derivatives with $\sigma = 1.2$, which is
$0.4\,l$.

scikit-image sets `size = int(3 * sigma)` and `lobe = size // 3`. Each box
term is a correlation with a kernel $K$: the output at pixel $p$ is
$\sum_d K(d)\, f(p + d)$. For a unit impulse at the centre pixel $q$, the
output at $p$ is $K(q - p)$, so the impulse response, rotated by 180°, is the
kernel:

```{code-cell} ipython3
def box_kernels(sigma, n=41):
    """The kernels `(dyy, dxy, dxx)` of `box_terms` at `sigma`, centred on n // 2."""
    impulse = np.zeros((n, n))
    impulse[n // 2, n // 2] = 1
    return [response[::-1, ::-1] for response in box_terms(impulse, sigma)]


fig, axes = plt.subplots(2, 3, figsize=(7.5, 5), constrained_layout=True)
for row, sigma in zip(axes, (3, 4)):
    for ax, K, name in zip(row, box_kernels(sigma, n=25), ("dyy", "dxy", "dxx")):
        limit = np.abs(K).max()
        ax.imshow(K, cmap=DIVERGING, norm=TwoSlopeNorm(0, -limit, limit))
        ax.axhline(12, color=INK, lw=0.6); ax.axvline(12, color=INK, lw=0.6)
        ax.set_title(f"sigma {sigma} (size {int(3 * sigma)}): {name}")
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Box kernels; the black lines cross at the output pixel",
             color=MUTED, fontsize=9)
plt.show()
```

Two features of these kernels break the SURF layout. The table measures each:
the centre of each kernel, and the widths of the two outer lobes of `dxx`.

```{code-cell} ipython3
def kernel_centre(K):
    """Mean (row, column) offset of the non-zero taps from the output pixel."""
    rows, cols = np.nonzero(K)
    n = K.shape[0]
    return (rows.mean() - n // 2, cols.mean() - n // 2)


def outer_lobes(K):
    """Widths of the negative runs left and right of the positive lobe, on one row."""
    rows = np.nonzero(K)[0]
    line = np.sign(K[(rows.min() + rows.max()) // 2])
    pos = np.nonzero(line > 0)[0]
    neg = np.nonzero(line < 0)[0]
    return int((neg < pos.min()).sum()), int((neg > pos.max()).sum())


rows = []
for sigma in (2, 3, 4, 5, 6, 8):
    dyy, dxy, dxx = box_kernels(sigma, n=61)
    rows.append({"sigma": sigma, "size": int(3 * sigma), "lobe": int(3 * sigma) // 3,
                 "dxx centre (r, c)": tuple(float(v) for v in kernel_centre(dxx)),
                 "dxy centre (r, c)": tuple(float(v) for v in kernel_centre(dxy)),
                 "dxx outer lobes (left, right)": outer_lobes(dxx)})
show_table(pd.DataFrame(rows))
```

First, every kernel is centred one pixel down from the output pixel, and one
pixel or more to the right. The box with top-left corner `(r, c)` starts at
row `r + 1` and column `c + 1`. Second, for an even `size` the two outer lobes
of `dxx` have different widths, and its centre moves a further half pixel to
the right. The code adds 1 to an even `size`, but only after it has
used `size` to set the middle box width and the offset `s2`, so the addition
has no effect. A kernel with unequal outer lobes is not symmetric about its
centre, so it also responds to the first derivative.

### 4.3 What a second-derivative filter must do

A second-derivative filter applied to $f(c) = c^2 / 2$ must give the same
value at every pixel, because $f''$ is 1 everywhere. The `dxx` box term on
that image, at three pixels along a row:

```{code-cell} ipython3
quadratic = np.tile((np.arange(101.0) - 50) ** 2 / 2, (101, 1))
rows = []
for sigma in (3, 4, 5, 6):
    dxx = box_terms(quadratic, sigma)[2]
    rows.append({"sigma": sigma, "size": int(3 * sigma),
                 "column 40": dxx[50, 40], "column 50": dxx[50, 50],
                 "column 60": dxx[50, 60]})
show_table(pd.DataFrame(rows), floatfmt=".4g")
```

For odd sizes the response is the same constant at each pixel. For even sizes
it changes along the row. The response is the
second derivative times a gain that depends on the lobe width $l$. For the
SURF layout, with lobe half-widths $h = (l-1)/2$ and $L = (3l-1)/2$, a row of
the `dxx` kernel holds $+2$ over $|c| \le h$ and $-1$ over $h < |c| \le L$,
on $2l - 1$ rows, with weight $1 / (3l)^2$. Its response to $c^2 / 2$ is:

```{code-cell} ipython3
l, k = sp.symbols("l k", positive=True)
sum_sq = lambda m: m * (m + 1) * (2 * m + 1) / 3  # Σ c² for |c| ≤ m
h, L = (l - 1) / 2, (3 * l - 1) / 2
# Σ_c K(c) c²/2 per row, times rows, times the weight; dxx = -(whole - 3 centre).
gain = sp.factor((2 * l - 1) * (3 * sum_sq(h) - sum_sq(L)) / 2 / (3 * l) ** 2)
print("dxx response to c**2/2:", gain)
print("leading term:", sp.limit(gain / l**2, l, sp.oo), "* l**2")
print("at l = 1, 3:", [gain.subs(l, v) for v in (1, 3)],
      "measured at sigma 3 (l = 3):", f"{box_terms(quadratic, 3)[2][50, 50]:.6f}")
```

The response is $-l(2l-1)/9$ times $f''$, about $-\tfrac{2}{9} l^2 f''$ for
large $l$. The determinant is then about $\tfrac{4}{81} l^4$ times the
determinant of $f$. The box route
therefore gives a determinant in different units from the Hessian determinant,
and the factor depends on `sigma` through `int(3 * sigma) // 3`. For
blob detection only the relative values matter, so SURF does not need the
Hessian units. A user who swaps `approximate=False` for `approximate=True`
gets values on a different scale.

### 4.4 Finite differences and Gaussian derivatives

FD applies the central difference $(\tfrac12, 0, -\tfrac12)$ twice, which is
the second difference $(\tfrac14, 0, -\tfrac12, 0, \tfrac14)$ along one axis.
On a smoothed image of scale $\sigma$ it under-reports the second derivative,
by more at small $\sigma$: the FD gain errors in §5 and §6 are all negative,
and largest at `sigma` 1. GD differentiates the smoothing kernel instead.
`on_shape_index.md` §4 and `hessian_review.md` measure both routes element by
element. The gain error on the determinant is about twice the gain error on
each element (§4.1).

## 5. Side by side on the blob

The blob of §3, at several `sigma`. The truth is `blob_hessian`. The image is
large enough that the blob is far from the border at every `sigma`.

```{code-cell} ipython3
N_BLOB, W_BLOB = 161, 4
blob_image = blob(N_BLOB, W_BLOB)
rows = []
for sigma in (1, 2, 3, 4, 8):
    truth = det2(blob_hessian(N_BLOB, W_BLOB, sigma))
    inner = interior(sigma)
    for name, D in (("box", box_det(blob_image, sigma)),
                    ("FD", det2(fd_hessian(blob_image, sigma))),
                    ("GD", det2(gd_hessian(blob_image, sigma)))):
        rms, gain_err, shape = det_error(D, truth, inner)
        peak = np.unravel_index(np.argmax(D), D.shape)
        rows.append({"sigma": sigma, "route": name, "E": rms,
                     "gain error": gain_err, "shape error S": shape,
                     "peak offset": tuple(int(p) - N_BLOB // 2 for p in peak)})
show_table(pd.DataFrame(rows), floatfmt=".3g")
```

On the blob, FD has a gain error from −0.10 at `sigma` 1 to −0.02 at
`sigma` 8, and a shape error below 0.03. The box route has a shape error from
0.5 to 2.3, and its maximum is off the blob centre at four of the five
`sigma`.

(sec-corpus)=
## 6. How big, on fitted photographs

The corpus is three photographs from `skimage.data`: `camera`, `coins` and
greyscale `astronaut`. `camera` and `astronaut` are 512 × 512, and `coins`
is 303 × 384. `gaussian_sum.fit`
replaces each photograph by a sum of Gaussians that matches it closely. The
smoothed Hessian of that sum is exact in closed form
[@mobahi2016closed_form], so it is a gold standard on photograph-like
content. `on_gaussian_sum_images.md` describes the fit, and
`test_hessian_metrics.py` on the `hessian-refactor` branch uses the same
images.

```{code-cell} ipython3
photographs = {
    "camera": ski.util.img_as_float(ski.data.camera()),
    "coins": ski.util.img_as_float(ski.data.coins()),
    "astronaut": ski.util.img_as_float(ski.color.rgb2gray(ski.data.astronaut())),
}
fitted = {}
for name, photo in photographs.items():
    levels, residual = gs.fit(photo)
    fitted[name] = (levels, gs.render(levels, photo.shape))
    print(f"{name:9s} {photo.shape}, fit RMS residual / image RMS: "
          f"{np.sqrt(np.mean(residual**2) / np.mean(photo**2)):.3f}")
```

```{code-cell} ipython3
CORPUS_SIGMAS = (1, 1.5, 2, 3, 4, 8)
corpus_rows = []
for name, (levels, image) in fitted.items():
    for sigma in CORPUS_SIGMAS:
        truth = det2(gs.hessian(levels, image.shape, sigma))
        inner = interior(sigma)
        for route, D in (("box", box_det(image, sigma)),
                         ("FD", det2(fd_hessian(image, sigma))),
                         ("GD", det2(gd_hessian(image, sigma)))):
            rms, gain_err, shape = det_error(D, truth, inner)
            corpus_rows.append({"image": name, "sigma": sigma, "route": route,
                                "E": rms, "gain error": gain_err,
                                "shape error S": shape})
corpus = pd.DataFrame(corpus_rows)
worst = corpus.groupby(["sigma", "route"])[["E", "shape error S"]].max().unstack("route")
worst.columns = [f"{metric}, {route}" for metric, route in worst.columns]
show_table(worst.reset_index(), floatfmt=".2g")
```

The table gives the largest value over the three images. The smallest and
largest value over the images, for each route:

```{code-cell} ipython3
spread = corpus.groupby(["route", "sigma"])[["E", "shape error S"]].agg(["min", "max"])
spread.columns = [f"{metric} {stat}" for metric, stat in spread.columns]
show_table(spread.reset_index(), floatfmt=".3g")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
for ax, metric in zip(axes, ("E", "shape error S")):
    for route in ("box", "FD", "GD"):
        values = corpus[corpus.route == route].groupby("sigma")[metric].max()
        ax.plot(values.index, values.values, "o-", color=ROUTE_COLOURS[route],
                lw=2, ms=5, label=route)
        ax.annotate(route, (values.index[-1], values.values[-1]),
                    xytext=(5, 0), textcoords="offset points", va="center",
                    color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("sigma")
    ax.set_title(f"{metric}, largest over the three images")
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlim(0.8, 9.5)
axes[0].legend(frameon=False)
plt.show()
```

On this corpus:

* GD has relative RMS error below $5 \times 10^{-4}$ at `sigma` 1, and at
  rounding level from `sigma` 2.
* FD has a relative RMS error of about 0.5 at `sigma` 1 and about 0.02 at
  `sigma` 8. Its shape error is about half to two thirds of $E$; the rest is
  gain.
* The box route has $E$ above 1 at every `sigma` and image. Its shape error,
  which ignores the scale, is above 1 at every `sigma` and image.

### 6.1 Which scale does the box route match?

The box route may approximate the determinant at a different scale from
`sigma`. For each `sigma`, search over the scale $\sigma'$ of the true
determinant for the smallest shape error, on the fitted `camera`:

```{code-cell} ipython3
def best_scale(D, levels, shape, sigma, factors):
    """Smallest shape error of `D` against the true determinant at sigma * factor."""
    inner = interior(sigma)
    results = [(det_error(D, det2(gs.hessian(levels, shape, sigma * f)), inner)[2], f)
               for f in factors]
    return min(results)


camera_levels, camera_image = fitted["camera"]
FACTORS = np.arange(0.3, 1.21, 0.1)
rows = []
for sigma in (2, 3, 4, 8, 16):
    shape_err, factor = best_scale(box_det(camera_image, sigma), camera_levels,
                                   camera_image.shape, sigma, FACTORS)
    at_sigma = det_error(box_det(camera_image, sigma),
                         det2(gs.hessian(camera_levels, camera_image.shape, sigma)),
                         interior(sigma))[2]
    rows.append({"sigma": sigma, "S at sigma": at_sigma,
                 "best sigma' / sigma": factor, "S at best sigma'": shape_err})
show_table(pd.DataFrame(rows), floatfmt=".2f")
```

From `sigma` 4, the box route best matches the determinant at about 0.6
`sigma`; at smaller `sigma` the best scale is larger. The SURF calibration,
$0.4\,l \approx 0.4\,\sigma$, predicts a smaller scale. Even at the best
scale the shape error is above 0.5 at every `sigma` tested.

## 7. Properties that ought to hold

The blob of §3 is rotationally symmetric, so its determinant must be
unchanged by a 180° rotation of the image plane, and its maximum must be at
the blob centre. Measure both, over blob widths and `sigma`:

```{code-cell} ipython3
rows = []
for width in (2, 4, 8):
    image = blob(101, width)
    for sigma in (1, 2, 3, 4, 8):
        for route, D in (("box", box_det(image, sigma)),
                         ("FD", det2(fd_hessian(image, sigma))),
                         ("GD", det2(gd_hessian(image, sigma)))):
            peak = np.unravel_index(np.argmax(D), D.shape)
            rows.append({"route": route, "width": width, "sigma": sigma,
                         "asymmetry": np.abs(D - D[::-1, ::-1]).max() / np.abs(D).max(),
                         "peak off centre": peak != (50, 50)})
props = pd.DataFrame(rows)
summary = props.groupby("route").agg(
    cases=("asymmetry", "size"),
    smallest_asymmetry=("asymmetry", "min"),
    largest_asymmetry=("asymmetry", "max"),
    peak_off_centre=("peak off centre", "sum"))
show_table(summary.reset_index(), floatfmt=".2g")
```

`asymmetry` is the largest difference between the determinant and its 180°
rotation, over the largest absolute value. FD and GD keep both properties in
all 15 cases. The box route breaks the symmetry in every case and has its
maximum off the centre in 12 of 15.

The one-pixel shift of §4.2 would by itself move the centre of symmetry to
pixel (49, 49). Measure the asymmetry about that pixel instead, by `sigma`:

```{code-cell} ipython3
rows = []
for sigma in (1, 2, 3, 4, 5, 8):
    worst_shifted = 0.0
    for width in (2, 4, 8):
        D = box_det(blob(101, width), sigma)[:-2, :-2]  # rows, columns 0..98
        worst_shifted = max(worst_shifted,
                            np.abs(D - D[::-1, ::-1]).max() / np.abs(D).max())
    rows.append({"sigma": sigma, "size": int(3 * sigma),
                 "asymmetry about (49, 49)": worst_shifted})
show_table(pd.DataFrame(rows), floatfmt=".2g")
```

For odd `size` the result is symmetric about (49, 49): the shift is the only
defect. For even `size` it is not symmetric about either pixel, because of the
unequal lobes.

## 8. Timing

The timings use the installed `hessian_matrix_det` for the box route and for
FD with `np.linalg.det`. They use `det2` for FD and GD with the closed-form
determinant. The images are tiles of `camera`. Each time is the shortest of
five runs, on the machine that ran this notebook.

```{code-cell} ipython3
camera = ski.util.img_as_float(ski.data.camera())
SIZES = (128, 256, 512, 1024)
TIME_SIGMAS = (1, 2, 4, 8, 16)
timing_rows = []
for size in SIZES:
    image = np.tile(camera, (2, 2))[:size, :size].copy()
    for sigma in TIME_SIGMAS:
        timing_rows.append({
            "size": size, "sigma": sigma,
            "box": best_time(lambda: hessian_matrix_det(image, sigma, approximate=True)),
            "FD": best_time(lambda: hessian_matrix_det(image, sigma, approximate=False)),
            "FD closed form": best_time(lambda: det2(fd_hessian(image, sigma))),
            "GD": best_time(lambda: det2(gd_hessian(image, sigma))),
        })
timings = pd.DataFrame(timing_rows)
show_table(timings.round(1), floatfmt="g")
```

Times are in milliseconds. "GD" here uses the closed-form determinant.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), constrained_layout=True)
routes = ("box", "FD", "FD closed form", "GD")
at_sigma = timings[timings.sigma == 4]
for route in routes:
    axes[0].plot(at_sigma["size"], at_sigma[route], "o-", lw=2, ms=5,
                 color=ROUTE_COLOURS[route], label=route)
axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
axes[0].set_xticks(SIZES, [str(s) for s in SIZES])
axes[0].set_xlabel("image side, pixels"); axes[0].set_ylabel("time, ms")
axes[0].set_title("sigma 4")
at_size = timings[timings["size"] == 512]
for route in routes:
    axes[1].plot(at_size["sigma"], at_size[route], "o-", lw=2, ms=5,
                 color=ROUTE_COLOURS[route], label=route)
    axes[1].annotate(route, (16, at_size[route].iloc[-1]), xytext=(5, 0),
                     textcoords="offset points", va="center", color=INK)
axes[1].set_xscale("log", base=2); axes[1].set_yscale("log")
axes[1].set_xticks(TIME_SIGMAS, [str(s) for s in TIME_SIGMAS])
axes[1].set_xlim(0.8, 40)
axes[1].set_xlabel("sigma"); axes[1].set_title("512 × 512")
for ax in axes:
    ax.grid(color=GRID, lw=0.6)
axes[0].legend(frameon=False)
plt.show()
```

The ratios to the box route at 512 × 512:

```{code-cell} ipython3
ratios = at_size.set_index("sigma")[["FD", "FD closed form", "GD"]].div(
    at_size.set_index("sigma")["box"], axis=0)
show_table(ratios.reset_index(), floatfmt=".1f")
```

The box route takes the same time at every `sigma`. FD with `np.linalg.det`
is about ten times slower than the box route at small `sigma`. Most of that
time is `np.linalg.det`: FD with the closed form is close to the box route at
`sigma` 1 and 2. GD is slower than FD with the closed form, and the ratio
grows with `sigma`. GD makes six 1-D passes with kernels of radius
$8\sigma$. FD makes two passes with kernels of radius $4\sigma$, then the
differences.

## 9. Ways forward

### 9.1 Closed-form determinant in the `approximate=False` route

`det2` gives the same values as `np.linalg.det` to rounding (§2) and is faster.
The speed-up for the FD route:

```{code-cell} ipython3
speedup = timings.assign(speedup=timings["FD"] / timings["FD closed form"])
show_table(speedup.pivot(index="size", columns="sigma", values="speedup")
           .round(1).reset_index(), floatfmt="g")
```

`np.linalg.det` also works for 3-D and higher images, where the 2 × 2 closed
form does not apply. For 3-D, the
[rule of Sarrus](https://en.wikipedia.org/wiki/Rule_of_Sarrus) gives a closed
form; this notebook does not time it.

### 9.2 Gaussian derivatives in the `approximate=False` route

`hessian_matrix_det` passes `use_gaussian_derivatives=False`. With GD, the
error on the corpus falls from 0.5 to $5 \times 10^{-4}$ at `sigma` 1, and
to rounding from `sigma` 2 (§6). The cost, from §8, is GD time over FD
closed-form time:

```{code-cell} ipython3
gd_cost = timings.assign(ratio=timings["GD"] / timings["FD closed form"])
show_table(gd_cost.pivot(index="size", columns="sigma", values="ratio")
           .round(1).reset_index(), floatfmt="g")
```

### 9.3 A centred box route

The SURF layout has an odd lobe width $l$ and a size of $3l$, with every box
placed symmetrically about the output pixel. `surf_det` implements that
layout:

```{code-cell} ipython3
def surf_det(image, lobe):
    """SURF determinant with an odd `lobe` width, filter size 3 * lobe, centred."""
    l, h, L = lobe, lobe // 2, 3 * lobe // 2
    weight = 1.0 / (3 * l) ** 2
    dxx = box_sums(image, -(l - 1), -L, 2 * l - 1, 3 * l) \
        - 3 * box_sums(image, -(l - 1), -h, 2 * l - 1, l)
    dyy = box_sums(image, -L, -(l - 1), 3 * l, 2 * l - 1) \
        - 3 * box_sums(image, -h, -(l - 1), l, 2 * l - 1)
    dxy = (box_sums(image, -l, -l, l, l) + box_sums(image, 1, 1, l, l)
           - box_sums(image, -l, 1, l, l) - box_sums(image, 1, -l, l, l))
    return (dxx * weight) * (dyy * weight) - 0.81 * (dxy * weight) ** 2
```

The symmetry test of §7, and the best scale on `camera` as in §6.1:

```{code-cell} ipython3
rows = []
for lobe in (3, 5, 7, 9, 15):
    asym, off = 0.0, 0
    for width in (2, 4, 8):
        D = surf_det(blob(101, width), lobe)
        asym = max(asym, np.abs(D - D[::-1, ::-1]).max() / np.abs(D).max())
        off += np.unravel_index(np.argmax(D), D.shape) != (50, 50)
    sigma_guess = 0.55 * lobe
    shape_err, factor = best_scale(surf_det(camera_image, lobe), camera_levels,
                                   camera_image.shape, sigma_guess,
                                   np.arange(0.7, 1.31, 0.05))
    rows.append({"lobe": lobe, "largest asymmetry": asym, "peaks off centre (of 3)": off,
                 "best sigma' / lobe": sigma_guess * factor / lobe,
                 "S at best sigma'": shape_err})
show_table(pd.DataFrame(rows), floatfmt=".2g")
```

The centred layout is symmetric to rounding and peaks at the blob centre. It
matches the determinant at about $0.55\,l$, with a shape error from 0.17 to
0.5 on `camera`. It is still in different units from the Hessian determinant
(§4.3), and it only offers the scales $0.55 \times$ an odd integer.

## 10. What each change costs

* **Default `approximate=False`.** Every 2-D call that does not pass
  `approximate` changes its output. On the corpus, the box result differs from
  the true determinant by more than the determinant itself ($E$ above 1 at
  every `sigma`, §6), and FD by at most 0.51. The box result also has a scale
  factor that depends on `sigma` (§4.3). Any threshold a user tuned on the old output needs retuning. The
  call is about ten times slower at 512 × 512 with `np.linalg.det`, and about
  1 to 5 times slower with the closed form (§8, §9.1). `blob_doh` calls the
  private `_hessian_matrix_det` and does not change.
* **Closed form for 2-D.** No change in output beyond rounding. About 3 to 10
  times faster for the `approximate=False` route (§9.1).
* **GD in the `approximate=False` route.** The output changes by the FD error:
  about 50% at `sigma` 1 and 2% at `sigma` 8 on the corpus (§6). It is slower
  than FD with the closed form by the factors of §9.2.
* **Centred box route.** The `approximate=True` output changes at every
  `sigma`, and `blob_doh` output changes with it. The available scales change.
  This notebook does not measure the effect on `blob_doh` detections.

## 11. What not to do

Do not read "approximate" as "the same quantity, with a small error". The box
route gives values in different units, at a scale of about 0.6 `sigma`, with a
shape error above 0.5 on `camera` even at its best scale (§6.1). It is a blob response, not an
approximation to `hessian_matrix_det(..., approximate=False)`.

Do not judge GD by the FD timing with `np.linalg.det`. The determinant
dominates that time (§8). With the closed form for both, GD costs about 2 to
7 times FD, depending on image size and `sigma` (§9.2). On the corpus it is
exact to rounding from `sigma` 2.

## 12. Summary

```{code-cell} ipython3
def corpus_value(route, sigma, metric="E"):
    """Largest `metric` over the corpus images for `route` at `sigma`."""
    sel = corpus[(corpus.route == route) & (corpus.sigma == sigma)]
    return sel[metric].max()


t512 = timings[timings["size"] == 512].set_index("sigma")
summary_rows = []
for route, time_col in (("box", "box"), ("FD", "FD"), ("FD", "FD closed form"),
                        ("GD", "GD")):
    summary_rows.append({
        "route": time_col,
        "E, sigma 1": corpus_value(route, 1),
        "E, sigma 4": corpus_value(route, 4),
        "S, sigma 4": corpus_value(route, 4, "shape error S"),
        "symmetric": bool(summary.loc[route, "largest_asymmetry"] < 1e-10),
        "ms, 512², sigma 1": t512.loc[1, time_col],
        "ms, 512², sigma 16": t512.loc[16, time_col],
    })
show_table(pd.DataFrame(summary_rows), floatfmt=".2g")
```

* The `approximate=True` route is the fastest, and its time does not depend on
  `sigma`. It is not an approximation to the Hessian determinant at `sigma`:
  it has different units, best matches a scale of about 0.6 `sigma`, and has
  a shape error above 1 at `sigma` on the corpus. Its kernels are shifted by one pixel, and
  for even `int(3 * sigma)` they are not symmetric.
* The `approximate=False` route is about ten times slower at 512 × 512, mostly
  because of `np.linalg.det`. With the closed form it is within a factor of
  about 1 to 5 of the box route.
* GD is exact on the corpus from `sigma` 2, and has error $5 \times 10^{-4}$
  at `sigma` 1. FD has error 0.5 at `sigma` 1 and 0.02 at `sigma` 8.

### Limits

* The accuracy corpus is three fitted photographs of about 512 × 512, at six
  values of `sigma` from 1 to 8. The truth is exact for the fitted sums, not
  for the photographs.
* Timings are from one machine, one run, with tiles of `camera`. Only 2-D
  images were timed. The 3-D route of `hessian_matrix_det` always uses
  `hessian_matrix`, and the box route does not apply there.
* The scikit-image under test is the installed version; GD is a transcription
  of the `hessian-refactor` branch, checked against that branch outside this
  notebook.
* This notebook does not measure the effect of any change on `blob_doh`
  detections.
