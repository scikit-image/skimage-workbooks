---
title: 'On the approximate Hessian determinant'
date: 2026-11-27
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

:::{related}
https://github.com/scikit-image/scikit-image/pull/930 | The pull request that added the box-filter determinant and `blob_doh` (2014); the code it ported is in [§4.2](#sec-port)
https://github.com/scikit-image/scikit-image/commit/32c0b1ca979fcfd201a435f13851666c01247941 | Earlier removal of GPL code from scikit-image (2013); see [§9.3](#sec-licence)
https://github.com/scikit-image/scikit-image/issues/2488 | Notes that SIFT and SURF are patent encumbered; see [§9.3](#sec-licence)
:::

:::{changelog}
2026-11-28 | claude-code:claude-opus-5-5 | Create workbook; compare with mahotas; base the fix on the dlib port
:::

`hessian_matrix_det(image, sigma, approximate=True)` and `blob_doh` compute
the determinant of the Hessian with the box filters of SURF
[@bay2006surf_speeded; @bay2008speededup_robust], on an integral image. The code is
`_hessian_matrix_det` in `skimage/feature/_hessian_det_appx_pythran.py`. This
notebook compares that code with the SURF papers, with an analysis of SURF
[@oyallon2015analysis_surf], and with four other implementations. It lists the
defects, and it tests a fix.

Coordinates are in array order: $r$ runs down the rows, $c$ across the
columns. `on_hessian_matrix_det.md` compares the box route with the
finite-difference and Gaussian-derivative routes of `approximate=False`.

`on_blob_dog.md` first examined these box filters, as part of a comparison of
the three blob detectors, and listed four defects, D1 to D4. This notebook
repeats each of its `blob_doh` measurements and replaces that discussion.
§12.1 lists each conclusion of `on_blob_dog.md`, the section here that tests
it, and whether it holds. References to sections of `on_blob_dog.md` are to
its version of 2026-11-16 (commit `7e3b2b2`), before the `blob_doh` analysis
moved here.

## 1. Setup

```{code-cell} ipython3
import time
import warnings

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from nbhelper import show_table
```

The subject under test is the installed scikit-image. `blob_doh` looks up
`_hessian_matrix_det` in the module `_skimage2.feature.blob`; §10 replaces it
there to test the fix. `gaussian_sum` makes the gold-standard images of §6;
`on_gaussian_sum_images.md` describes it.

```{code-cell} ipython3
import skimage as ski
from skimage.feature import blob_doh, hessian_matrix_det
import _skimage2.feature.blob as blob_module

import gaussian_sum as gs

print("scikit-image", ski.__version__)
```

One comparator runs here: mahotas, whose SURF code is a port of dlib's
(§8). `pyramid` is a private function of mahotas, so this notebook may need
changes when mahotas changes.

```{code-cell} ipython3
# Comparators.
import mahotas
from mahotas.features import _surf as mahotas_surf
from mahotas.features.surf import integral as mahotas_integral

print("mahotas", mahotas.__version__)
```

```{code-cell} ipython3
# Categorical slots 1-4 of the default palette. validate_palette.js, light
# mode: all checks pass; worst adjacent CVD ΔE 11.8; contrast WARN for slots
# 2-4, so every chart has a legend, direct labels or a table.
COLOURS = {
    "current": "#257de0",
    "fixed": "#f47230",
    "Gaussian": "#00b894",
    "OpenCV": "#f2a600",
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

### 2.1 Boxes

A box filter adds the image over a rectangle. An
[integral image](https://en.wikipedia.org/wiki/Summed-area_table) gives that
sum in four look-ups, for any size of rectangle [@crow1984summedarea_tables;
@viola2004robust_realtime]. `box_sums` computes the sum at every pixel. It counts zeros
outside the image, so it is a reference with no border defects.

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

A filter is a list of boxes. Each box is `(r0, c0, nr, nc, weight)`: its top
row and left column relative to the output pixel, its numbers of rows and
columns, and its weight. `kernel` draws the filter as an array $K$. The
output at pixel $p$ is $\sum_d K(d)\, f(p + d)$, with the output pixel at the
centre of the array.

```{code-cell} ipython3
def apply(image, boxes):
    """Weighted sum of the box sums of `boxes` at every pixel."""
    out = np.zeros(np.shape(image))
    for r0, c0, nr, nc, weight in boxes:
        out += weight * box_sums(image, r0, c0, nr, nc)
    return out


def kernel(boxes, n=41):
    """The filter of `boxes` as an `n` × `n` array; output pixel at `n // 2`."""
    K = np.zeros((n, n))
    h = n // 2
    for r0, c0, nr, nc, weight in boxes:
        K[h + r0:h + r0 + nr, h + c0:h + c0 + nc] += weight
    return K


def transpose(boxes):
    """The same filter with rows and columns swapped."""
    return [(c0, r0, nc, nr, w) for r0, c0, nr, nc, w in boxes]


def det_from(image, layout, weight=0.81):
    """SURF determinant `dxx dyy - weight dxy²` for a layout `(dyy, dxy, dxx)`."""
    dyy, dxy, dxx = (apply(image, boxes) for boxes in layout)
    return dxx * dyy - weight * dxy**2
```

### 2.2 Layouts

`skimage_layout` transcribes `_hessian_matrix_det`. The code calls the
derivative across the columns `dxx` and the one down the rows `dyy`. `size`
is `int(3 * sigma)`, and `lobe = size // 3` is the width of one lobe. In the
code, `_integ(img, r, c, rl, cl)` sums rows `r + 1` to `r + rl` and columns
`c + 1` to `c + cl`; `shift = 1` reproduces that (§4.2).

```{code-cell} ipython3
def skimage_layout(sigma, shift=1):
    """Boxes `(dyy, dxy, dxx)` of skimage `_hessian_matrix_det` at `sigma`."""
    size = int(3 * sigma)
    s2, lobe = (size - 1) // 2, size // 3
    w = 1.0 / size / size
    s = shift
    dxy = [(-lobe + s, 1 + s, lobe, lobe, -w), (1 + s, -lobe + s, lobe, lobe, -w),
           (-lobe + s, -lobe + s, lobe, lobe, w), (1 + s, 1 + s, lobe, lobe, w)]
    dxx = [(-lobe + 1 + s, -s2 + s, 2 * lobe - 1, size, -w),
           (-lobe + 1 + s, -(lobe // 2) + s, 2 * lobe - 1, lobe, 3 * w)]
    dyy = [(-s2 + s, -lobe + 1 + s, size, 2 * lobe - 1, -w),
           (-(lobe // 2) + s, -lobe + 1 + s, lobe, 2 * lobe - 1, 3 * w)]
    return dyy, dxy, dxx
```

`surf_layout` is the SURF layout for an odd lobe width $L$, as Oyallon and
Rabin define it [@oyallon2015analysis_surf, eqs. 10–11], with the weight
$1 / (3L)^2$ of OpenSURF [@evans2009notes_opensurf]:

* `dxx` is $+1$ over $3L$ columns and $-3$ over the middle $L$ columns, on
  $2L - 1$ rows. That is lobes of $+1, -2, +1$, each $L$ columns wide.
* `dxy` is four $L \times L$ quadrants, $+1$ on one diagonal and $-1$ on the
  other, with a gap of one row and one column through the output pixel.

```{code-cell} ipython3
def surf_layout(lobe):
    """Boxes `(dyy, dxy, dxx)` of the SURF filters with odd `lobe` width L."""
    assert lobe % 2 == 1
    L = lobe
    w = 1.0 / (3 * L) ** 2
    half_outer, half_inner = (3 * L - 1) // 2, (L - 1) // 2
    dxx = [(-L + 1, -half_outer, 2 * L - 1, 3 * L, w),
           (-L + 1, -half_inner, 2 * L - 1, L, -3 * w)]
    dxy = [(-L, 1, L, L, w), (1, -L, L, L, w), (-L, -L, L, L, -w), (1, 1, L, L, -w)]
    return transpose(dxx), dxy, dxx
```

`opencv_layout` transcribes `calcLayerDetAndTrace` and `resizeHaarPattern`
in OpenCV's
[`surf.cpp`](https://github.com/opencv/opencv_contrib/blob/4.x/modules/xfeatures2d/src/surf.cpp).
OpenCV defines each filter at size 9, scales the box corners by
`size / 9` and rounds them, and divides each box weight by the area of that
box. §8 compares it with `surf_layout`.

```{code-cell} ipython3
def opencv_layout(size):
    """Boxes `(dyy, dxy, dxx)` of OpenCV SURF for filter `size`."""
    dx_s = [(0, 2, 3, 7, 1), (3, 2, 6, 7, -2), (6, 2, 9, 7, 1)]  # x1, y1, x2, y2, w
    dy_s = [(2, 0, 7, 3, 1), (2, 3, 7, 6, -2), (2, 6, 7, 9, 1)]
    dxy_s = [(1, 1, 4, 4, 1), (5, 1, 8, 4, -1), (1, 5, 4, 8, -1), (5, 5, 8, 8, 1)]
    ratio, margin = size / 9, size // 2

    def resize(pattern):
        boxes = []
        for x1, y1, x2, y2, weight in pattern:
            x1, y1, x2, y2 = (int(np.floor(ratio * v + 0.5)) for v in (x1, y1, x2, y2))
            boxes.append((y1 - margin, x1 - margin, y2 - y1, x2 - x1,
                          weight / ((x2 - x1) * (y2 - y1))))
        return boxes

    return resize(dy_s), resize(dxy_s), resize(dx_s)
```

The transcription of the installed code must match it:

```{code-cell} ipython3
rng = np.random.default_rng(0)
noise = rng.uniform(size=(96, 96))
rows = []
for sigma in (1, 1.7, 2, 2.5, 3, 3.7, 4, 5.5, 8):
    margin = int(3 * sigma) + 3
    inner = (slice(margin, -margin),) * 2
    ours = det_from(noise, skimage_layout(sigma))
    theirs = hessian_matrix_det(noise, sigma, approximate=True)
    rows.append({"sigma": sigma,
                 "max difference / max value, interior":
                     np.abs(ours - theirs)[inner].max() / np.abs(theirs[inner]).max()})
show_table(pd.DataFrame(rows), floatfmt=".1e")
```

They agree to rounding away from the border. `skimage_layout` counts zeros
outside the image. `current_det` also copies the border rule of `_integ`,
which clips each box corner to the image (§4.6), and the `max(0, …)` clamp
(§4.5):

```{code-cell} ipython3
def current_det(image, sigma):
    """`_hessian_matrix_det` on `integral_image(image)`, vectorised, border included."""
    I = ski.transform.integral_image(np.asarray(image, dtype=float))
    H, W = I.shape
    size = int(3 * sigma)
    s2, s3 = (size - 1) // 2, size // 3
    w_i = 1.0 / size / size
    r, c = np.indices(I.shape)

    def integ(r, c, rl, cl):
        """`_integ`: clip the corner, then add the extent to the clipped corner."""
        r = np.clip(r, 0, H - 1)
        c = np.clip(c, 0, W - 1)
        r2 = np.clip(r + rl, 0, H - 1)
        c2 = np.clip(c + cl, 0, W - 1)
        return np.maximum(0.0, I[r, c] + I[r2, c2] - I[r, c2] - I[r2, c])

    dxy = -(integ(r - s3, c + 1, s3, s3) + integ(r + 1, c - s3, s3, s3)
            - integ(r - s3, c - s3, s3, s3) - integ(r + 1, c + 1, s3, s3)) * w_i
    dxx = -(integ(r - s3 + 1, c - s2, 2 * s3 - 1, size)
            - 3 * integ(r - s3 + 1, c - s3 // 2, 2 * s3 - 1, s3)) * w_i
    dyy = -(integ(r - s2, c - s3 + 1, size, 2 * s3 - 1)
            - 3 * integ(r - s3 // 2, c - s3 + 1, s3, 2 * s3 - 1)) * w_i
    return dxx * dyy - 0.81 * dxy**2


worst = max(np.abs(current_det(noise, s) - hessian_matrix_det(noise, s)).max()
            / np.abs(hessian_matrix_det(noise, s)).max()
            for s in (1, 1.7, 2.5, 3, 4, 8, 16))
print(f"current_det against the installed code, whole image: {worst:.1e}")
```

`current_det` is the installed code at every pixel, border included.

### 2.3 Measurement

The gold standard is the exact Hessian of a sum of Gaussians, which
`gaussian_sum` gives in closed form [@mobahi2016closed_form]. `det2` takes the
determinant of a 2 × 2 symmetric matrix from its upper-diagonal elements.
`shape_error` is the relative error that remains after the best single scale
factor $g$: $S = \|D - gT\| / \|gT\|$, for the determinant $D$ under test and
the truth $T$.

```{code-cell} ipython3
def det2(H):
    """Closed-form determinant from `[Hrr, Hrc, Hcc]`."""
    Hrr, Hrc, Hcc = H
    return Hrr * Hcc - Hrc**2


def shape_error(got, truth):
    """Best gain `g` of `got` against `truth`, and the shape error `S`."""
    g = np.sum(got * truth) / np.sum(truth * truth)
    return g, np.linalg.norm(got - g * truth) / np.linalg.norm(g * truth)


def disk(n, radius, centre=None):
    """`n` × `n` image, 1 inside a disk of `radius` about `centre`, else 0."""
    centre = (n // 2, n // 2) if centre is None else centre
    r, c = np.indices((n, n))
    return (((r - centre[0]) ** 2 + (c - centre[1]) ** 2) <= radius**2).astype(float)


def peak(D):
    """(row, column) of the maximum of `D`."""
    return tuple(int(v) for v in np.unravel_index(np.argmax(D), D.shape))


def best_time(func, repeats=5):
    """Shortest of `repeats` wall-clock times of `func()`, in milliseconds."""
    func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return 1e3 * min(times)
```

## 3. The problem on one disk

A disk of radius 10, centred on pixel (80, 80), at `sigma` 10. Its
determinant must be symmetric about the centre, with its maximum at the
centre.

```{code-cell} ipython3
N, RADIUS, CENTRE = 161, 10, 80
one_disk = disk(N, RADIUS)
current = hessian_matrix_det(one_disk, 10, approximate=True)
window = (slice(CENTRE - 30, CENTRE + 31),) * 2
fig, ax = plt.subplots(figsize=(3.4, 3.2), constrained_layout=True)
D = current[window]
limit = np.abs(D).max()
ax.imshow(D, cmap=DIVERGING, norm=TwoSlopeNorm(0, -limit, limit))
p = peak(D)
ax.plot(p[1], p[0], "+", color=INK, ms=9)
ax.plot(30, 30, "o", mfc="none", mec=INK, ms=9)
ax.set_title(f"current, sigma 10: maximum at {tuple(v - 30 for v in p)} from centre")
ax.set_xticks([]); ax.set_yticks([])
plt.show()
print("asymmetry under 180° rotation about the centre:",
      f"{np.abs(D - D[::-1, ::-1]).max() / limit:.2f}")
```

The circle is the disk centre and the cross is the maximum. The response is
not symmetric, and its maximum is one pixel up and one to the left of the
centre.

## 4. The mechanisms

### 4.1 The SURF filters

SURF replaces the Gaussian second derivatives $L_{xx}$, $L_{yy}$ and $L_{xy}$
with the box filters $D_{xx}$, $D_{yy}$ and $D_{xy}$ of `surf_layout`. Bay
and colleagues use filters of size $3L$ = 9, 15, 21, 27 and so on, so the lobe
width $L$ is 3, 5, 7, 9. The lobe width is always odd. Between two scales,
they "must increase this size by a minimum of 2 pixels (1 pixel on every side)
in order to keep the size uneven and thus ensure the presence of the central
pixel" [@bay2008speededup_robust, §3.3]. Evans gives the same rule for OpenSURF: "the
lobe size can increase by a minimum of 2", and so "the smallest step size
between consecutive filters is 6" [@evans2009notes_opensurf, §1.2.2]. Their
determinant is

$$
\det(H_\text{approx}) = D_{xx} D_{yy} - (w D_{xy})^2, \qquad w = 0.9
$$

[@bay2008speededup_robust, eq. 3].

**Scale.** "The 9 × 9 box filters … are approximations of a Gaussian with
$\sigma = 1.2$" [@bay2008speededup_robust, §3.2], and Evans gives
$\sigma_\text{approx} = \text{filter size} \times 1.2 / 9$
[@evans2009notes_opensurf, §1.2.2]. That is $\sigma = 0.4\,L$. Neither source says how
the value 1.2 was found. Oyallon and Rabin show that other ways of matching a
box to a Gaussian give other factors, and conclude that "defining a relation
between the box parameters … and the Gaussian scale variable σ seems quite
arbitrary" [@oyallon2015analysis_surf, §3.1]. §6 measures the factor.

**Weight.** The weight $w$ balances the Frobenius norms of the box filters
against those of the Gaussian derivatives, at $L = 3$ and $\sigma = 1.2$
[@bay2008speededup_robust, eq. 4]. Bay and colleagues add that "for theoretical
correctness, the weighting changes depending on the scale. In practice, we
keep this factor constant" [@bay2008speededup_robust, §3.2]. Oyallon and Rabin derive
$w(L) = \sqrt{(2L-1)/(2L)}$, which is 0.913 at $L = 3$ and tends to 1
[@oyallon2015analysis_surf, eq. 26]. §6.1 tests the constant weight.

**Normalisation.** The sources describe the normalisation in two ways. Bay
and colleagues write that "the filter responses are normalised with respect
to their size. This guarantees a constant Frobenius norm for any filter size"
[@bay2008speededup_robust, §3.2]. Evans writes that "scale-normalisation is achieved by
dividing the response by the area" [@evans2009notes_opensurf, §1.5.2], and OpenSURF
divides by $(3L)^2$. Division by the area does not give a constant Frobenius
norm:

```{code-cell} ipython3
show_table(pd.DataFrame([
    {"L": L, "Frobenius norm of dxx / (3L)^2":
         np.linalg.norm(kernel(surf_layout(L)[2], 61)),
     "times L": L * np.linalg.norm(kernel(surf_layout(L)[2], 61))}
    for L in (3, 5, 7, 9, 13)]), floatfmt=".4f")
```

The norm falls as $1/L$. Oyallon and Rabin make the same point: a constant
norm "means that box filters responses should be simply divided by the scale
parameter L" [@oyallon2015analysis_surf, §3.3.3]. But their determinant, eq. 24,
divides by $L^4$, which is division of each filter by $L^2$, as in OpenSURF.
They justify $L^2$ by analogy with the $\gamma$-normalised Gaussian derivative
[@oyallon2015analysis_surf, §4.1]. The determinant of the normalised Hessian is
$t^{2\gamma}(L_{xx}L_{yy} - L_{xy}^2)$ with $t = \sigma^2$, and $\gamma = 1$
gives perfect scale invariance [@lindeberg1998feature_detection].

Division by the area gives that normalisation. A $D_{xx}$ filter applied to
$f(c) = c^2 / 2$, whose second derivative is 1, gives
$L(2L-1)/9 \approx \tfrac{2}{9} L^2$ (`on_hessian_matrix_det.md` §4.3). The
Gaussian derivative with $\gamma = 1$, $\sigma^2 L_{xx}$, gives $\sigma^2$. So
a box determinant is proportional to $\sigma^4 \det H$, and a blob detector
can compare it across scales. §6 measures the constant of proportion. The
scikit-image code follows OpenSURF, and this notebook keeps the division by
the area.

(sec-port)=
### 4.2 The port from OpenSURF, and the shift

`_hessian_matrix_det` follows `FastHessian::buildResponseLayer` in OpenSURF
[@evans2009notes_opensurf] line for line, with the same variable names, offsets and the
0.81 weight. The OpenSURF source is in the
[`gussmith23/opensurf`](https://github.com/gussmith23/opensurf) repository on
GitHub, as
[`src/fasthessian.cpp`](https://github.com/gussmith23/opensurf/blob/master/src/fasthessian.cpp)
and
[`include/integral.h`](https://github.com/gussmith23/opensurf/blob/master/include/integral.h).
Its file headers state: "This library is distributed under the GNU GPL". In OpenSURF, a box with top-left pixel
`(row, col)` covers rows `row` to `row + rows - 1`. Its `BoxIntegral`
converts that to integral-image indices:

```c++
// The subtraction by one for row/col is because row/col is inclusive.
int r1 = std::min(row,          img->height) - 1;
int c1 = std::min(col,          img->width)  - 1;
int r2 = std::min(row + rows,   img->height) - 1;
int c2 = std::min(col + cols,   img->width)  - 1;
```

The scikit-image `_integ` has no subtraction of one:

```python
r = _clip(r, 0, img.shape[0] - 1)
c = _clip(c, 0, img.shape[1] - 1)
r2 = _clip(r + rl, 0, img.shape[0] - 1)
c2 = _clip(c + cl, 0, img.shape[1] - 1)
ans = img[r, c] + img[r2, c2] - img[r, c2] - img[r2, c]
```

SURF defines the integral image with inclusive sums,
$I_\Sigma(x, y) = \sum_{i \le x} \sum_{j \le y} I(i, j)$
[@bay2008speededup_robust, eq. 1], and so does `integral_image`:
$I[r, c] = \sum_{r' \le r,\, c' \le c} f[r', c']$. The sum over rows $r_0$ to
$r_1$ therefore needs $I[r_1] - I[r_0 - 1]$. That is the reason for the
subtraction in OpenSURF. In `_integ`,
$I[r_2, c_2] - I[r, c_2] - I[r_2, c] + I[r, c]$ sums rows $r + 1$ to $r_2$
and columns $c + 1$ to $c_2$. Every box is one row down and one column to the
right of the OpenSURF box. Measure the centre of each kernel of
`skimage_layout`, and compare it with the SURF kernel:

```{code-cell} ipython3
def kernel_centre(K):
    """Mean (row, column) of |K|, relative to the output pixel."""
    a = np.abs(K)
    r, c = np.indices(K.shape) - K.shape[0] // 2
    return (round(float((r * a).sum() / a.sum()), 3),
            round(float((c * a).sum() / a.sum()), 3))


rows = []
for sigma, lobe in ((3, 3), (5, 5), (7, 7)):
    ours = [kernel(b) for b in skimage_layout(sigma)]
    surf = [kernel(b) for b in surf_layout(lobe)]
    rows.append({"sigma": sigma, "size": int(3 * sigma),
                 "dxx centre": kernel_centre(ours[2]),
                 "dxy centre": kernel_centre(ours[1]),
                 "SURF dxx centre": kernel_centre(surf[2]),
                 "same as SURF moved by (1, 1)": all(
                     np.allclose(np.roll(S, (1, 1), (0, 1)), -K)
                     for S, K in zip(surf, ours))})
show_table(pd.DataFrame(rows))
```

At these `sigma` the scikit-image filters are the SURF filters, with the sign
reversed and moved by one pixel in each direction. The sign does not change
the determinant. The move puts each output one pixel up and to the left of
the image structure that makes it: a feature at pixel $p$ shows at
$p - (1, 1)$.

(sec-doh-geometry)=
### 4.3 Sizes from `int(3 * sigma)`

scikit-image takes `size = int(3 * sigma)` and `lobe = size // 3`. SURF needs
`size` $= 3L$ with $L$ odd. Three things go wrong at other sizes:

* when `size` is not a multiple of 3, the outer box is wider than $3L$, so the
  outer lobes are wider than the middle lobe, and the weights do not sum to
  zero;
* when `lobe` is even, the middle lobe of width $L$ cannot be centred on a
  pixel;
* when `size` is even, the outer box cannot be centred on a pixel either. The
  code adds 1 to an even `size`, but after it has set the box widths, so that
  line has no effect.

A second-derivative filter must give zero for a constant image and for a
linear ramp, and the same value at every pixel for $c^2/2$. Test the `dxx`
filter for those three images, over `sigma`:

```{code-cell} ipython3
n = 121
cols = np.tile(np.arange(n, dtype=float) - n // 2, (n, 1))
tests = {"constant": np.ones((n, n)), "ramp": cols, "quadratic": cols**2 / 2}


def dxx_checks(layout):
    """Responses of the `dxx` filter of `layout` to the three test images."""
    dxx = layout[2]
    res = {name: apply(im, dxx)[n // 2, n // 2 - 5:n // 2 + 6] for name, im in tests.items()}
    return {"constant": np.abs(res["constant"]).max(),
            "ramp": np.abs(res["ramp"]).max(),
            "quadratic, spread": np.ptp(res["quadratic"])}


rows = []
for sigma in np.round(np.arange(1, 8.01, 1 / 3), 2):
    size = int(3 * sigma)
    rows.append({"sigma": sigma, "size": size, "lobe": size // 3,
                 "SURF size": size == 3 * (size // 3) and (size // 3) % 2 == 1,
                 **dxx_checks(skimage_layout(sigma))})
sizes = pd.DataFrame(rows)
show_table(sizes, floatfmt=".3g")
```

The columns give the largest absolute response to the constant and the ramp,
and the spread of the response to the quadratic over 11 pixels along a row.
All three are zero when `size` is a SURF size. At every other `sigma` in the
table, at least one is non-zero:

```{code-cell} ipython3
fails = (sizes[["constant", "ramp", "quadratic, spread"]] > 1e-9).any(axis=1)
print("rows with a SURF size:", int(sizes["SURF size"].sum()), "of", len(sizes))
print("SURF size and passes all three:", bool((~fails == sizes["SURF size"]).all()))
```

The SURF sizes 3, 9, 15, 21, … come from `sigma` in $[1, 4/3)$, $[3, 10/3)$,
$[5, 16/3)$ and so on: one sixth of the range of `sigma` from 1 up.
`blob_doh` with its default `min_sigma=1`, `max_sigma=30` and `num_sigma=10`
uses these `sigma`:

```{code-cell} ipython3
default_sigmas = np.linspace(1, 30, 10)
show_table(pd.DataFrame({
    "sigma": default_sigmas,
    "size": [int(3 * s) for s in default_sigmas],
    "lobe": [int(3 * s) // 3 for s in default_sigmas],
    "SURF size": [int(3 * s) % 3 == 0 and (int(3 * s) // 3) % 2 == 1
                  for s in default_sigmas]}), floatfmt=".4g")
```

#### Symmetry is not enough

`on_blob_dog.md` §5 tests the filters for symmetry. The middle box of width
`size` starts at `c - s2`, so it is centred on the output pixel only when
`size` is odd. The inner box of width `lobe` starts at `c - lobe // 2`, so it is
centred only when `lobe` is odd. It concludes that the usable sizes are the odd
sizes with an odd lobe: 3, 5, 9, 11, 15, 17, …, "odd and not 1 mod 6".

The symmetry argument is correct, but a symmetric filter is not always a
second-derivative filter. Measure both properties for every `size` from 3 to
24: the asymmetry of the response to a symmetric Gaussian blob, with and
without the one-pixel shift, and the response of `dxx` to a constant image:

```{code-cell} ipython3
blob_n = 121
br, bc = np.indices((blob_n, blob_n)) - blob_n // 2
sym_blob = np.exp(-(br**2 + bc**2) / (2 * 6.0**2))


def asymmetry_of(layout):
    """Largest change under a 180° rotation about the blob centre, over the maximum."""
    D = det_from(sym_blob, layout)
    return np.abs(D - D[::-1, ::-1]).max() / np.abs(D).max()


rows = []
for size in range(3, 25):
    sigma = size / 3 + 1e-9  # so that int(3 * sigma) == size
    lobe = size // 3
    rows.append({"size": size, "size mod 6": size % 6, "lobe": lobe,
                 "asymmetry, current": asymmetry_of(skimage_layout(sigma)),
                 "asymmetry, shift removed": asymmetry_of(skimage_layout(sigma, shift=0)),
                 "dxx response to a constant":
                     abs(apply(np.ones((41, 41)), skimage_layout(sigma)[2])[20, 20]),
                 "SURF size": size == 3 * lobe and lobe % 2 == 1})
geometry = pd.DataFrame(rows)
show_table(geometry, floatfmt=".2g")
```

With the shift, no size is symmetric. Without it, the response is symmetric
for every odd `size` with an odd `lobe`, as `on_blob_dog.md` says, and not
symmetric for the others. But the sizes 5, 11, 17 and 23, with `size mod 6`
equal to 5, have a middle box two pixels wider than $3L$. Their weights do not
sum to zero, so they respond to a constant image:

```{code-cell} ipython3
symmetric = geometry["asymmetry, shift removed"] < 1e-9
zero_sum = geometry["dxx response to a constant"] < 1e-12
print("symmetric sizes:        ", geometry["size"][symmetric].tolist())
print("symmetric and zero-sum: ", geometry["size"][symmetric & zero_sum].tolist())
print("SURF sizes:             ", geometry["size"][geometry["SURF size"]].tolist())
```

The filter is a second-derivative filter only at the SURF sizes $3L$ with $L$
odd. The rows of `dxx` through the output pixel, without the shift, show the
three cases:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.4), sharey=True,
                         constrained_layout=True)
for ax, size in zip(axes, (8, 7, 11, 9)):
    K = kernel(skimage_layout(size / 3 + 1e-9, shift=0)[2], n=31)
    row = K[15] * size**2
    x = np.arange(31) - 15
    keep = np.abs(x) <= 8
    ax.bar(x[keep], -row[keep], width=0.85, color=COLOURS["current"])
    ax.axvline(0, color=INK, lw=0.8, ls=":")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_title(f"size {size}, lobe {size // 3}: sum {-row.sum() + 0:+.0f}")
    ax.set_xticks([-8, -4, 0, 4, 8])
axes[0].set_ylabel("weight × size²")
fig.suptitle("Middle row of dxx, shift removed, signs as in SURF. "
             "Dotted line: output pixel.",
             color=MUTED, fontsize=9)
plt.show()
```

At `size` 8 and 7 the row is not symmetric about the output pixel. At `size`
11 it is symmetric, but its weights sum to $+2$ per row: it is a smoothing
filter plus a second-derivative filter. At `size` 9 it is the SURF filter.

### 4.4 The response across `sigma`

`blob_doh` picks the scale of a blob from the maximum of the response over
`sigma`. With scale normalisation, the response at the centre of a disk rises
to one maximum and falls. The weight $1/\text{size}^2$ normalises only the
SURF sizes: at other sizes the box sums and the weight do not match. The
centre response for the disk of §3, on a fine grid of `sigma`. "Fixed" is the
SURF layout with the odd lobe nearest to `sigma` (§9):

```{code-cell} ipython3
def lobe_for(sigma):
    """The odd lobe width nearest to `sigma`: 1 for sigma < 2, 3 for [2, 4), ..."""
    return 2 * int(sigma / 2) + 1


def surf_det(image, sigma):
    """SURF determinant, centred, with the odd lobe nearest to `sigma`."""
    return det_from(image, surf_layout(lobe_for(sigma)))


sweep = np.arange(1, 20.01, 0.25)
# The current code puts the response to the disk centre at (79, 79), §4.2.
response = {
    "current": np.array([hessian_matrix_det(one_disk, s)[CENTRE - 1, CENTRE - 1]
                         for s in sweep]),
    "fixed": np.array([surf_det(one_disk, s)[CENTRE, CENTRE] for s in sweep]),
}
fig, ax = plt.subplots(figsize=(6.5, 3), constrained_layout=True)
for name, values in response.items():
    ax.plot(sweep, values / values.max(), color=COLOURS[name], lw=2, label=name)
ax.set_xlabel("sigma")
ax.set_ylabel("centre response / its maximum")
ax.set_title(f"Disk of radius {RADIUS}: response at the disk centre")
ax.grid(color=GRID, lw=0.6)
ax.set_xlim(1, 20)
ax.legend(frameon=False, loc="lower right")
plt.show()


def local_maxima(v):
    """Number of local maxima in a sequence, with each run of equal values as one."""
    v = v[np.r_[True, np.diff(v) != 0]]
    return int(((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:])).sum())


for name, values in response.items():
    print(f"{name:8s} local maxima over sigma: {local_maxima(values)}, "
          f"largest at sigma {sweep[np.argmax(values)]}")
```

The current response has a saw-tooth over `sigma`, with several local maxima.
The fixed response changes in steps, because the lobe width changes in steps
of 2. Its maximum is one step, at lobe 11. Both have their largest value at about the disk
radius, which is the relation `blob_doh` documents: "The radius of each blob
is approximately `sigma`".

### 4.5 The clamp to zero

`_integ` returns `max(0.0, ans)`, as OpenSURF does. A box sum of a
non-negative image is never negative, so the clamp has no effect there. A box
sum of an image with negative values can be negative, and the clamp sets it to
zero. A second derivative does not change when a constant is added to the
image. Test that, away from the border, at SURF sizes:

```{code-cell} ipython3
patch = ski.util.img_as_float(ski.data.camera())[100:228, 100:228]
rows = []
for sigma in (3, 5, 7):
    margin = 3 * int(sigma) + 2
    inner = (slice(margin, -margin),) * 2
    reference = hessian_matrix_det(patch, sigma)[inner]
    row = {"sigma": sigma}
    for offset in (1, -0.5, -1):
        moved = hessian_matrix_det(patch + offset, sigma)[inner]
        row[f"offset {offset:+g}"] = (np.abs(moved - reference).max()
                                      / np.abs(reference).max())
    rows.append(row)
show_table(pd.DataFrame(rows), floatfmt=".2g")
```

The table gives the largest change over the largest value. The patch is in
$[0, 1]$. Adding 1 changes nothing beyond rounding. Subtracting 0.5 changes
the result by a large fraction of its size; subtracting 1 makes every value
zero. `hessian_matrix_det` is a public function with no check on the sign of
its input, and `blob_doh(..., prescale='none')` passes a float image through
unchanged.

(sec-doh-border)=
### 4.6 The border

`_integ` clips the top-left corner of each box to the image first, and then
adds the extent to the clipped corner:

```python
r = _clip(r, 0, img.shape[0] - 1)
r2 = _clip(r + rl, 0, img.shape[0] - 1)
```

A request for corner row `r` and `rl` rows asks for image rows `r + 1` to
`r + rl` (§4.2). The table follows that request near the top and the bottom of
an image of 8 rows, for `rl = 4`:

```{code-cell} ipython3
def rows_summed(r, rl, height):
    """Image rows that `_integ` sums for corner row `r` and extent `rl`."""
    r1 = min(max(r, 0), height - 1)
    r2 = min(max(r1 + rl, 0), height - 1)
    return list(range(r1 + 1, r2 + 1))


height, rl = 8, 4
show_table(pd.DataFrame([
    {"corner row r": r,
     "rows asked for, in the image": [x for x in range(r + 1, r + rl + 1)
                                      if 0 <= x < height],
     "rows summed": rows_summed(r, rl, height)}
    for r in (-4, -3, -2, -1, 0, 3, 4, 5, 6, 7)]))
```

Two different things happen at the two ends:

* at the top (and left), a box that starts above the image keeps its full
  extent and slides down, to start at row 1. Row 0 is never summed. The sum is
  over rows the filter did not ask for;
* at the bottom (and right), a box that runs past the image stops at the last
  row. It sums exactly the rows of the request that are inside the image, so
  it gives the response for an image that is zero outside.

`on_blob_dog.md` §8 describes the slide and the shrink. It also says that the
shrunk box "no longer matches the area actually summed", and that neither edge
gives "what any `mode` would give". The second half of the table shows that
the bottom and right edges give zero padding, the rule of `mode='constant'`.
Measure it on `camera` at half size, against `skimage_layout`, which has the
same shift and counts zeros outside:

```{code-cell} ipython3
photo = ski.util.img_as_float(ski.data.camera())[::2, ::2]
rows = []
for sigma in (2, 3, 4, 8, 16):
    band = int(3 * sigma) + 2
    current = current_det(photo, sigma)
    zero_padded = det_from(photo, skimage_layout(sigma))
    p = band + 2
    edge_padded = det_from(np.pad(photo, p, mode="edge"),
                           skimage_layout(sigma))[p:-p, p:-p]
    scale = np.abs(current[band:-band, band:-band]).max()
    diff_zero = np.abs(current - zero_padded) / scale
    diff_edge = np.abs(current - edge_padded) / scale
    inner = (slice(band, None), slice(band, None))  # away from the top and left
    rows.append({"sigma": sigma, "band": band,
                 "vs zero outside, top and left": max(diff_zero[:band].max(),
                                                      diff_zero[:, :band].max()),
                 "vs zero outside, bottom and right": diff_zero[inner].max(),
                 "vs edge padding, bottom and right": diff_edge[inner].max()})
show_table(pd.DataFrame(rows), floatfmt=".2g")
```

The values are the largest difference over the largest interior value. At the
bottom and right the current code is zero padding to rounding. At the top and
left it differs from zero padding by 35% to 130% of the interior maximum. Against edge padding, the reference of `on_blob_dog.md` §8, the bottom
and right also differ, because zero padding is not edge padding. So the defect
is at the top and left only, and it comes from the order of the two `_clip`
calls, as the shift of §4.2 comes from the missing subtraction of one.

SURF does not define a value at the border. OpenSURF sets "a border for each
octave such that the largest filter in the octave remains entirely over the
image", because "edge effects can yield anomolous results"
[@evans2009notes_opensurf, §1.5.2]. dlib does the same (§8). `hessian_matrix_det`
returns a value at every pixel, so it must choose a border rule. Two fixes
give a consistent rule:

* zero outside the image, `mode='constant'`: the fix of §9.1 clips each box
  to the image before it sums it, at the cost of four comparisons per box;
* pad the image once under a chosen `mode`, run the box filters, and crop, as
  `on_blob_dog.md` §8 proposes. This offers every `mode` of `hessian_matrix`.
  The pad must cover the reach of the filter, `size - (size - 1) // 2` pixels
  for the current layout, plus one for the shift of §4.2. Its cost, with the installed code on the 256 × 256
  image:

```{code-cell} ipython3
def reach(sigma):
    """Pixels the current filter reaches past the output pixel."""
    size = int(3 * sigma)
    return size - (size - 1) // 2 + 1


def pad_once(image, sigma, mode="edge"):
    """Approximate determinant with one boundary extension, then crop."""
    p = reach(sigma)
    return hessian_matrix_det(np.pad(image, p, mode=mode), sigma)[p:-p, p:-p]


rows = []
for sigma in (2, 4, 8, 16):
    t_now = best_time(lambda: hessian_matrix_det(photo, sigma))
    t_pad = best_time(lambda: pad_once(photo, sigma))
    rows.append({"sigma": sigma, "pad": reach(sigma), "current, ms": t_now,
                 "pad once, ms": t_pad, "ratio": t_pad / t_now})
show_table(pd.DataFrame(rows), floatfmt=".3g")
```

The pad costs extra in proportion to the padded area, which grows with
`sigma`.

### 4.7 `sigma` below 1

For `sigma` below 1, `lobe` is 0 and the boxes are empty. Below 1/3, `size`
is 0 and the weight $1/\text{size}^2$ divides by zero:

```{code-cell} ipython3
with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for sigma in (0.3, 0.5, 0.9):
        D = hessian_matrix_det(patch, sigma)
        print(f"sigma {sigma}: all finite {np.isfinite(D).all()}, "
              f"max |value| {np.nanmax(np.abs(D)):.1e}")
```

(sec-doh-quantised)=
### 4.8 The quantised scale axis

`size = int(3 * sigma)` is the only use of `sigma`, so `sigma` values with the
same `size` give the same output:

```{code-cell} ipython3
base = hessian_matrix_det(photo, 3.0)
show_table(pd.DataFrame([
    {"sigma": sigma, "size": int(3 * sigma),
     "same output as sigma 3": bool(np.array_equal(hessian_matrix_det(photo, sigma), base))}
    for sigma in (3.0, 3.2, 3.32, 3.34, 3.67, 4.0)]))
```

`on_blob_dog.md` §6 concludes that a `num_sigma` finer than steps of 1/3
"buys duplicate work and nothing else". It does more than that. `blob_doh`
finds peaks with `peak_local_max`, which keeps a pixel that equals the maximum
of its 3 × 3 × 3 neighbourhood, so a tie counts as a peak. Where two adjacent
scale planes are identical and the response falls with scale, the second of
the two ties with its neighbour and is higher than the plane beyond it, so it
is a peak. The disk of §3 at three values of `num_sigma`:

```{code-cell} ipython3
rows = []
for num_sigma in (10, 28, 85):
    sigmas = np.linspace(2, 30, num_sigma)
    sizes_used = [int(3 * s) for s in sigmas]
    cube = np.stack([hessian_matrix_det(one_disk, s) for s in sigmas], axis=-1)
    column = cube[CENTRE - 1, CENTRE - 1]
    ball = ndi.maximum_filter(cube, size=3, mode="constant", cval=-np.inf)
    peaks = np.flatnonzero((cube == ball)[CENTRE - 1, CENTRE - 1] & (column > 0.005))
    found = blob_doh(one_disk, min_sigma=2, max_sigma=30, num_sigma=num_sigma,
                     threshold=0.005)
    rows.append({"num_sigma": num_sigma,
                 "duplicate planes": num_sigma - len(set(sizes_used)),
                 "scale peaks at the centre (sigma)": [round(float(sigmas[k]), 2) for k in peaks],
                 "blob_doh sigma": [round(float(b[2]), 2) for b in found]})
show_table(pd.DataFrame(rows))
```

With no duplicate planes there is one scale peak at the disk centre, near the
radius 10. With 9 duplicate planes there are 15, from `sigma` 2.67 to 20.67.
`blob_doh` removes the smaller of two overlapping blobs, so it reports the
largest of them. A fine `num_sigma` therefore changes the answer. Any fix that
quantises the scale more coarsely, as §9.1 does, makes this worse unless
`blob_doh` removes duplicate planes (§9.1).

## 5. Side by side on the disk

The disk of §3 at `sigma` 10, with the current code and with `surf_det`:

```{code-cell} ipython3
fixed = surf_det(one_disk, 10)
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), constrained_layout=True)
for ax, (name, D_full) in zip(axes, (("current", current), ("fixed", fixed))):
    D = D_full[window]
    limit = np.abs(D).max()
    ax.imshow(D, cmap=DIVERGING, norm=TwoSlopeNorm(0, -limit, limit))
    p = peak(D)
    ax.plot(p[1], p[0], "+", color=INK, ms=9)
    ax.plot(30, 30, "o", mfc="none", mec=INK, ms=9)
    asym = np.abs(D - D[::-1, ::-1]).max() / limit
    ax.set_title(f"{name}: maximum at {tuple(v - 30 for v in p)}, "
                 f"asymmetry {asym:.2g}")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Circle: disk centre. Cross: maximum. Each panel has its own scale.",
             color=MUTED, fontsize=9)
plt.show()
```

The fixed response is symmetric about the disk centre, and its maximum is at
the centre.

## 6. How big, on fitted photographs

The corpus is `camera` and `coins` from `skimage.data`, replaced by sums of
Gaussians with `gaussian_sum.fit`. For each `sigma`, search over the scale
$\sigma'$ of the true determinant for the smallest shape error $S$. The box
route may match a Gaussian scale other than `sigma`; $S$ measures how close it
gets at its best scale, whatever the units.

```{code-cell} ipython3
fitted = {}
for name in ("camera", "coins"):
    photo = ski.util.img_as_float(getattr(ski.data, name)())
    levels, _ = gs.fit(photo)
    fitted[name] = (levels, gs.render(levels, photo.shape))

FACTORS = np.round(np.arange(0.35, 0.86, 0.05), 2)


def best_match(D, levels, shape, lobe, margin):
    """Smallest shape error of `D` over true determinants at lobe * FACTORS."""
    inner = (slice(margin, -margin),) * 2
    results = []
    for f in FACTORS:
        truth = det2(gs.hessian(levels, shape, f * lobe))[inner]
        g, S = shape_error(D[inner], truth)
        results.append((S, f, g / (f * lobe) ** 4))
    return min(results)


rows = []
for name, (levels, image) in fitted.items():
    for sigma in (3, 3.7, 4, 5, 6, 7, 8):
        lobe = int(3 * sigma) // 3
        margin = int(3 * sigma) + int(8 * 0.85 * lobe) + 3
        cur = best_match(det_from(image, skimage_layout(sigma, shift=0)),
                         levels, image.shape, lobe, margin)
        fix = best_match(surf_det(image, sigma), levels, image.shape,
                         lobe_for(sigma), margin)
        rows.append({"image": name, "sigma": sigma, "size": int(3 * sigma),
                     "S current": cur[0], "sigma'/lobe, current": cur[1],
                     "fixed lobe": lobe_for(sigma), "S fixed": fix[0],
                     "sigma'/lobe, fixed": fix[1], "gain / sigma'^4, fixed": fix[2]})
corpus = pd.DataFrame(rows)
show_table(corpus, floatfmt=".3g")
```

"Current" here is the current layout without the one-pixel shift, so that the
table measures the layout defects of §4.3 alone. At the SURF sizes (`sigma`
3, 5, 7) the two layouts are the same, and so are their errors. At the other
sizes the current layout has a larger $S$ than the fixed one in every row,
from 1.4 to 120 times larger. The fixed lobe is not always the current lobe:
at `sigma` 4 the current lobe is 4 and the fixed lobe is 5.

The fixed filters best match the Gaussian determinant at $\sigma'$ from
$0.55\,L$ to $0.6\,L$ on this corpus, not at the $0.4\,L$ of the SURF paper.
Their output is $0.43$ to $0.51$ times $\sigma'^4 \det H$: the
scale-normalised determinant times a constant. The shape error at the best
scale, 0.17 to 0.47, is the error of the box approximation itself. Bay and
colleagues see a related effect: Hessian-based detectors have "a lower
repeatability score for angles around odd multiples of $\pi/4$"
[@bay2008speededup_robust, Fig. 3], and the repeatability is best near multiples of
$\pi/2$, "due to the square shape of the filter" [@bay2008speededup_robust, §3.2].

### 6.1 The weight 0.81

Fit both coefficients of $a\,D_{xx}D_{yy} - b\,D_{xy}^2$ to the true
determinant at the best scale, by least squares. The ratio $b/a$ is the best
$w^2$:

```{code-cell} ipython3
levels, image = fitted["camera"]
rows = []
for lobe in (3, 5, 7, 9, 13):
    dyy, dxy, dxx = (apply(image, b) for b in surf_layout(lobe))
    margin = 3 * lobe + int(8 * 0.55 * lobe) + 3
    inner = (slice(margin, -margin),) * 2
    truth = det2(gs.hessian(levels, image.shape, 0.55 * lobe))[inner].ravel()
    P, Q = (dxx * dyy)[inner].ravel(), (dxy**2)[inner].ravel()
    (a, b), *_ = np.linalg.lstsq(np.c_[P, -Q], truth, rcond=None)
    fit = a * P - b * Q
    rows.append({"lobe": lobe, "best w^2": b / a,
                 "Oyallon-Rabin w(L)^2": (2 * lobe - 1) / (2 * lobe),
                 "S, w^2 = 0.81": shape_error(P - 0.81 * Q, truth)[1],
                 "S, best w^2": np.linalg.norm(fit - truth) / np.linalg.norm(truth)})
show_table(pd.DataFrame(rows), floatfmt=".3f")
```

The best $w^2$ is near 0.81 at $L = 3$ and rises with $L$, as $w(L)^2$ does.
The best weight lowers $S$ by at most 0.06, so the fixed weight 0.81 is not a
defect. This agrees with Bay and colleagues, for whom a constant weight "did
not have a significant impact on the results in our experiments"
[@bay2008speededup_robust, §3.2].

(sec-doh-scale)=
### 6.2 Scale selection on blobs and disks

`blob_doh` takes the `sigma` at which the response peaks over scale as the
blob size. `on_blob_dog.md` §7 compares that choice with the exact
scale-normalised determinant, $\sigma^4 \det H$, on Gaussian blobs, and finds
that the box route "reads high". For a Gaussian blob of standard deviation
$w$, the exact determinant with $\gamma = 1$ peaks at $\sigma = w$
[@lindeberg1998feature_detection]. The exact route here uses the
moment-corrected Gaussian-derivative kernels of `on_hessian_matrix_det.md`
§2. The table gives the `sigma` of the largest response at the blob centre,
on a grid of step 0.25, for Gaussian blobs and for disks:

```{code-cell} ipython3
def taps(sigma, order, truncate=8):
    """Moment-corrected 1-D Gaussian derivative kernel (on_hessian_matrix_det.md §2)."""
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
    return k / ((k * x**2).sum() / 2)


def exact_centre(image, sigma, centre):
    """sigma^4 det H at `centre` only, with Gaussian-derivative kernels, zero outside."""
    t = {o: taps(sigma, o) for o in (0, 1, 2)}
    h = len(t[0]) // 2
    padded = np.pad(image, h)
    window = padded[centre[0]:centre[0] + 2 * h + 1, centre[1]:centre[1] + 2 * h + 1]

    def element(a, b):
        return t[a] @ window @ t[b]

    return sigma**4 * (element(2, 0) * element(0, 2) - element(1, 1) ** 2)


def lobe_for_gaussian(sigma):
    """Odd lobe nearest sigma / 0.55: the §9.2 alternative."""
    return 2 * int(sigma / 0.55 / 2) + 1


M = 201
mr, mcol = np.indices((M, M)) - M // 2
rows = []
for kind, size_ in (("Gaussian", 1.5), ("Gaussian", 3), ("Gaussian", 5), ("Gaussian", 8),
                    ("disk", 4), ("disk", 10), ("disk", 16), ("disk", 24)):
    if kind == "Gaussian":
        image = np.exp(-(mr**2 + mcol**2) / (2 * size_**2))
    else:
        image = ((mr**2 + mcol**2) <= size_**2).astype(float)
    sig = np.arange(1, 3 * size_ + 2.01, 0.25)
    centre = (M // 2, M // 2)
    shifted_centre = (M // 2 - 1, M // 2 - 1)  # where the current code puts it
    pick = lambda values: float(sig[int(np.argmax(values))])
    rows.append({
        "blob": f"{kind} {size_}",
        "exact": pick([exact_centre(image, s, centre) for s in sig]),
        "current": pick([current_det(image, s)[shifted_centre] for s in sig]),
        "fixed, lobe near sigma": pick([surf_det(image, s)[centre] for s in sig]),
        "fixed, lobe near sigma/0.55": pick(
            [det_from(image, surf_layout(lobe_for_gaussian(s)))[centre] for s in sig]),
    })
show_table(pd.DataFrame(rows), floatfmt=".3g")
```

The "fixed" columns change in steps, because the lobe does; the table gives
the first `sigma` of the best step. Three results:

* The exact determinant picks $\sigma = w$ for the Gaussian blobs, and about
  $0.7 R$ for a disk of radius $R$.
* The current code and the fixed code with the lobe near `sigma` pick a
  larger `sigma` than the exact determinant, on both kinds of blob: 1.3 to 2
  times $w$ on the Gaussian blobs. This is the "reads high" of
  `on_blob_dog.md` §7. The lobe near `sigma` matches a Gaussian of about
  $0.55\,\sigma$ on photographs (§6), so `sigma` is larger than the Gaussian
  scale it stands for. With the lobe near $\sigma / 0.55$ the choice falls to
  0.75 to 0.9 times $w$, below the exact value.
* On disks, the fixed code with the lobe near `sigma` picks $\sigma = R$, the
  `blob_doh` docstring relation "radius ≈ `sigma`". The current code picks
  $R + 1$.

The best lobe for a disk, with the response interpolated between lobes by a
parabola, as SURF does for its scale [@bay2008speededup_robust, §3.4]:

```{code-cell} ipython3
lobes = np.arange(1, 41, 2)
rows = []
for radius in (6, 10, 16, 24):
    image = ((mr**2 + mcol**2) <= radius**2).astype(float)
    response = np.array([det_from(image, surf_layout(L))[M // 2, M // 2] for L in lobes])
    k = int(np.argmax(response))
    y0, y1, y2 = response[k - 1:k + 2]
    rows.append({"radius": radius, "best odd lobe": int(lobes[k]),
                 "interpolated lobe": lobes[k] + (y0 - y2) / (y0 - 2 * y1 + y2)})
show_table(pd.DataFrame(rows), floatfmt=".3g")
```

The interpolated best lobe is the radius plus 1.4 to 1.6 pixels. So on disks
the fixed filters report a radius that is high by about 1.5 pixels, which is
25% at radius 6 and 6% at radius 24. `on_blob_dog.md` D3 proposes to
"recalibrate the constant" in `size = int(3 * sigma)`. A new constant scales
`sigma`; it does not remove an offset of constant size. Which constant is
right depends on what `sigma` is to mean (§9.2): a disk radius, or the
standard deviation of an equivalent Gaussian.

## 7. Properties that ought to hold

Five promises for a determinant of the Hessian at one scale. The test image
is a Gaussian blob of standard deviation $0.55\,\sigma$, centred on a pixel;
a disk can give several pixels with the same largest value.

* **constant**: a constant added to the image changes nothing away from the
  border (§4.5);
* **ramp**: `dxx` gives zero for a linear ramp (§4.3);
* **centred**: the maximum is at the blob centre;
* **symmetric**: the response does not change under a 180° rotation about
  the blob centre;
* **transpose**: transposing the image transposes the response.

```{code-cell} ipython3
def properties(det_func, layout_func, sigmas):
    """Fraction of `sigmas` at which each promise holds, to 1e-9."""
    counts = dict.fromkeys(("constant", "ramp", "centred", "symmetric", "transpose"), 0)
    for sigma in sigmas:
        r, c = np.indices((N, N)) - CENTRE
        image = np.exp(-(r**2 + c**2) / (2 * (0.55 * sigma) ** 2))
        D = det_func(image, sigma)
        w = (slice(CENTRE - 40, CENTRE + 41),) * 2
        counts["centred"] += peak(D) == (CENTRE, CENTRE)
        counts["symmetric"] += np.allclose(D[w], D[w][::-1, ::-1], atol=1e-9 * np.abs(D).max())
        counts["transpose"] += np.allclose(det_func(image.T, sigma), D.T)
        inner = (slice(45, -45),) * 2
        moved = det_func(image + 1, sigma)
        counts["constant"] += np.allclose(moved[inner], D[inner], atol=1e-9 * np.abs(D).max())
        counts["ramp"] += np.abs(apply(tests["ramp"], layout_func(sigma)[2])[n // 2, n // 2]) < 1e-9
    return {k: f"{v} / {len(sigmas)}" for k, v in counts.items()}


SIGMAS = np.round(np.arange(1, 12.01, 0.5), 2)
show_table(pd.DataFrame([
    {"route": "current", **properties(lambda im, s: hessian_matrix_det(im, s),
                                      skimage_layout, SIGMAS)},
    {"route": "fixed", **properties(surf_det, lambda s: surf_layout(lobe_for(s)),
                                    SIGMAS)},
]))
```

`SIGMAS` runs from 1 to 12 in steps of 0.5. The current code keeps the
transpose property, because the shift is the same in both directions. It is
centred and symmetric at none of the `sigma`. It passes the constant test at
the integer `sigma`, where `size` is a multiple of 3, and the ramp test only
at the SURF sizes. The fixed code keeps all five at every `sigma`.

(sec-doh-comparators)=
## 8. Comparators

Four other implementations of the SURF determinant. All four are read from
their source, and §8.1 runs mahotas:

* **OpenSURF** [@evans2009notes_opensurf]:
  [`fasthessian.cpp`](https://github.com/gussmith23/opensurf/blob/master/src/fasthessian.cpp).
  Filter sizes 9, 15, 21, … ; `surf_layout`; weight $1/(3L)^2$; zero outside
  the image; the `max(0, …)` clamp.
* **dlib**:
  [`hessian_pyramid.h`](https://github.com/davisking/dlib/blob/master/dlib/image_keypoint/hessian_pyramid.h).
  Lobe widths $2^{o+1}(i+1) + 1$, all odd; boxes from `centered_rect`, with
  the `dxy` quadrants at offsets $\pm(\lfloor L/2 \rfloor + 1)$, which is
  `surf_layout`; weight $1/(3L)^2$; computes only pixels at least a border
  away from the edge.
* **mahotas**:
  [`_surf.cpp`](https://github.com/luispedro/mahotas/blob/master/mahotas/features/_surf.cpp),
  a port of dlib. Its `sum_rect` subtracts one from each integral-image index,
  as OpenSURF does, so its boxes are not shifted.
* **OpenCV**: `opencv_layout` (§2.2).

None of them shifts the boxes. OpenSURF, dlib and mahotas use only odd lobe
widths. OpenCV differs from `surf_layout` in two ways.

First, it divides each box by its own area, so `dxx` is $9L/(2L-1)$ times,
and `dxy` 9 times, the `surf_layout` filter.

Second, its boxes are the 9 × 9 boxes scaled by `size / 9`, so for `size` 21
and up its lobes are shorter than $2L - 1$ and its `dxy` gap is wider than one
pixel. The SURF paper supports both readings of the box shape. Its Fig. 5
shows the 15 × 15 filter with lobes of $2L - 1 = 9$, and the text says that
"for dimensions different from $l_0$ (e.g. the width of the central band for
the vertical filter in Fig. 5), rescaling the mask introduces rounding-off
errors" [@bay2008speededup_robust, §3.3]. Rescaling is what OpenCV does. The two
readings agree at sizes 9 and 15:

```{code-cell} ipython3
rows = []
for lobe in (3, 5, 7, 9):
    cv = [kernel(b, 61) for b in opencv_layout(3 * lobe)]
    surf = [kernel(b, 61) for b in surf_layout(lobe)]
    same_support = all(np.array_equal(A != 0, B != 0) for A, B in zip(cv, surf))
    ratio = {name: np.unique(np.round(A[B != 0] / B[B != 0], 4)).tolist()
             for name, A, B in zip(("dyy", "dxy", "dxx"), cv, surf)}
    rows.append({"size": 3 * lobe, "same boxes as surf_layout": same_support,
                 "dxx ratio": ratio["dxx"] if same_support else "-",
                 "9L/(2L-1)": 9 * lobe / (2 * lobe - 1),
                 "dxy ratio": ratio["dxy"] if same_support else "-",
                 "dxx rows": int((cv[2] != 0).any(axis=1).sum()), "2L-1": 2 * lobe - 1})
show_table(pd.DataFrame(rows), floatfmt=".4g")
```

With the per-box division, OpenCV's `0.81` weights `dxy` relative to `dxx` by
a different amount from OpenSURF's. In OpenSURF units, OpenCV computes
$(9L/(2L-1))^2 \bigl(D_{xx}D_{yy} - 0.81\,((2L-1)/L)^2 D_{xy}^2\bigr)$, an
effective $w^2$ of 2.25 at $L = 3$. On the fitted `camera`, at the best scale:

```{code-cell} ipython3
levels, image = fitted["camera"]
rows = []
for lobe in (3, 5, 7, 9):
    margin = 3 * lobe + int(8 * 0.85 * lobe) + 3
    for name, D in (("SURF layout", det_from(image, surf_layout(lobe))),
                    ("OpenCV", det_from(image, opencv_layout(3 * lobe)))):
        S, f, _ = best_match(D, levels, image.shape, lobe, margin)
        rows.append({"size": 3 * lobe, "implementation": name,
                     "best sigma'/lobe": f, "S": S})
show_table(pd.DataFrame(rows), floatfmt=".3g")
```

On this image OpenCV gives a larger shape error than the SURF layout at every
size. At sizes 9 and 15 the boxes are the same, so there the difference comes
from the weighting alone. The 0.81 weight belongs with one common weight for
all boxes. This is a property of OpenCV, not of scikit-image.

### 8.1 mahotas, run

mahotas is the only one of the four whose determinant can be called from
Python with a standard install. The PyPI wheels of OpenCV leave out SURF,
which is in the `nonfree` module. dlib's Python bindings do not expose the
Hessian pyramid, and OpenSURF has no Python bindings.

`mahotas_surf.pyramid(integral, octaves, intervals, step)` returns, for each
octave, one plane per interval. In octave 0 the lobe of interval $i$ is
$2(i + 1) + 1$, and the step is one pixel. Each plane differs from the
SURF determinant in three ways, all in `_surf.cpp`:

* the weight of $D_{xy}^2$ is 0.36 ($0.6^2$), not 0.81;
* a negative determinant becomes zero;
* the result is multiplied by the sign of $D_{xx} + D_{yy}$.

mahotas also leaves zero every pixel closer to the edge than
$\lceil 3 (L_{\max} + 2) / 2 \rceil$, where $L_{\max}$ is the largest lobe
it computes in the octave. Its `get_border_size` takes the lobe of one more
interval than it computes. `mahotas_like` applies the same three changes to a layout, so the
comparison is inside that margin. The image is noise on a frame that is not
square, so that a swap of rows and columns would show:

```{code-cell} ipython3
def mahotas_like(image, layout):
    """A layout's determinant with mahotas's weight, clamp and sign."""
    dyy, dxy, dxx = (apply(image, boxes) for boxes in layout)
    det = np.maximum(dxx * dyy - 0.36 * dxy**2, 0)
    return np.where(dxx + dyy < 0, -1.0, 1.0) * det


noise = rng.uniform(size=(96, 104))
INTERVALS = 6
planes = mahotas_surf.pyramid(mahotas_integral(noise.copy()), 1, INTERVALS, 1)[0]
largest = 2 * INTERVALS + 1
margin = int(np.ceil(3 * (largest + 2) / 2))  # get_border_size in _surf.cpp
inner = (slice(margin, -margin),) * 2
rows = []
for i, plane in enumerate(planes):
    lobe = 2 * (i + 1) + 1
    row = {"lobe": lobe,
           "mahotas zero outside the margin": bool(
               np.all(plane[~np.pad(np.ones_like(plane[inner], bool), margin)] == 0))}
    for name, layout in (("SURF layout", surf_layout(lobe)),
                         ("current", skimage_layout(lobe))):
        ref = mahotas_like(noise, layout)
        row[f"max |mahotas − {name}| / max"] = (
            np.abs(plane[inner] - ref[inner]).max() / np.abs(ref[inner]).max())
    rows.append(row)
show_table(pd.DataFrame(rows), floatfmt=".2g")
```

Inside the margin, mahotas agrees with `surf_layout` to rounding error at
every lobe. It disagrees with the current layout of scikit-image at the
same lobe, which is the layout of `surf_layout` moved by one pixel (§4.2).
So an implementation that came from dlib, not from OpenSURF, has the same
boxes as the fix of §9.1, centred on the same pixel. This checks the
interior only: mahotas computes nothing near the edge, and so does not
test the border (§4.6).

## 9. Ways forward

(sec-doh-fix)=
### 9.1 The fix

The fix replaces `_hessian_det_appx_pythran.py` with a port of dlib's
`hessian_pyramid` (§8), under the Boost Software License. That is the code
on the `surf-fixes` branch of scikit-image. Against the current code it
changes three things:

1. Each box is named by its centre and its height and width, with dlib's
   `centered_rect`, and summed with dlib's `get_sum_of_area`. The boxes are
   then centred on the output pixel, which removes the shift of §4.2. The box
   sum first clips the rectangle to the image, so pixels outside the image
   count as zero (§4.6). There is no `max(0.0, …)` clamp (§4.5).
2. The lobe is the odd integer nearest to `sigma`, and the filters are
   `3 * lobe` wide. This keeps the current scale of the SURF sizes (`lobe` 3
   for `sigma` 3), and so the `blob_doh` relation "radius ≈ `sigma`" (§4.4).
3. The dead `size += 1` goes.

The port differs from dlib in five ways: it computes one lobe from `sigma`,
not a pyramid; it computes every pixel, where dlib leaves a border; it clips
each box to the image, so pixels outside count as zero, where dlib requires
the box to be inside the image; and it keeps the determinant as it is,
without dlib's clamp of negative values to zero and without dlib's sign of
the Laplacian. `on_surf_dlib_port.md`
compares the port with dlib and OpenSURF call by call.

The port is plain Python, in the form that Pythran compiles. This notebook
loads it from `hessian_det_appx_dlib_port.py`, a copy of the branch file,
and checks the copy by its hash:

```{code-cell} ipython3
import hashlib
import inspect
import pathlib

PORT_SOURCE = pathlib.Path("hessian_det_appx_dlib_port.py").read_text()
assert hashlib.sha256(PORT_SOURCE.encode()).hexdigest() == (
    "9c151e192f48378494b432cd6ef26c27e14216469466ac05ce151c71147c6b58")
port = {"__name__": "port"}
exec(compile(PORT_SOURCE, "hessian_det_appx_dlib_port.py", "exec"), port)
```

The box functions, as dlib's `centered_rect` and `get_sum_of_area`:

```{code-cell} ipython3
for name in ("_centered_rect", "_sum_of_area"):
    print(inspect.getsource(port[name]))
```

`_hessian_matrix_det` takes the integral image, as the current code does. It
must match `surf_det`, at every pixel including the border, on an image with
negative values:

```{code-cell} ipython3
signed = rng.normal(size=(40, 48))
integral = ski.transform.integral_image(signed)
for sigma in (0.5, 1, 2.5, 3, 4.2, 7):
    loop = port["_hessian_matrix_det"](integral, sigma)
    vector = surf_det(signed, sigma)
    print(f"sigma {sigma}: lobe {lobe_for(sigma)}, max difference / max value "
          f"{np.abs(loop - vector).max() / np.abs(vector).max():.1e}")
```

The loop does the same number of box sums per pixel as the current code,
eight. Each box sum adds four comparisons to clip the rectangle. This
notebook does not build the Pythran module, so it does not time the fix.

The odd lobes make the scale axis coarser: a step of 2 in `sigma` per lobe.
`blob_doh` must therefore compute each lobe once and report the lobe as the
scale, or duplicate planes give false scale peaks (§4.8). As a function, with
the steps of `blob_doh` otherwise unchanged:

```{code-cell} ipython3
from skimage.feature import peak_local_max


def blob_doh_lobes(image, min_sigma=1, max_sigma=30, num_sigma=10,
                   threshold=0.01, overlap=0.5):
    """`blob_doh` with the fixed filters, one plane per distinct lobe."""
    sigmas = np.linspace(min_sigma, max_sigma, num_sigma)
    lobes = np.unique([lobe_for(s) for s in sigmas])
    cube = np.stack([det_from(image, surf_layout(L)) for L in lobes], axis=-1)
    peaks = peak_local_max(cube, threshold_abs=threshold, exclude_border=False,
                           footprint=np.ones((3, 3, 3)))
    if peaks.size == 0:
        return np.empty((0, 3))
    peaks = peaks.astype(float)
    peaks[:, -1] = lobes[peaks[:, -1].astype(int)]
    return blob_module._prune_blobs(peaks, overlap)
```

§10.3 tests it.

(sec-doh-sigma)=
### 9.2 What `sigma` means

The fixed filters match a Gaussian determinant at about $0.55\,L$ (§6), so
`sigma` is not the standard deviation of a Gaussian, as the docstrings of
`hessian_matrix_det` and `blob_doh` say. Two choices:

* keep the lobe nearest to `sigma`, as in §9.1, and document that `sigma` sets
  the lobe width, which corresponds to a Gaussian of about $0.55\,\sigma$;
* or take the lobe nearest to $\sigma / 0.55$, so that `sigma` is close to a
  Gaussian standard deviation. That changes every `sigma` that `blob_doh`
  returns, by a factor of about 1.8, and the relation to blob radius.

The first keeps `blob_doh` results; the second matches the docstring.

(sec-licence)=
### 9.3 The licence of the source

The fix of §9.1, and any other change to this code, raises a question about
where the current code comes from. This section gives the record. It is a
reading of that record, not legal advice.

#### What the licences say

scikit-image is under the BSD 3-clause licence. `LICENSE.txt` has no entry
for `_hessian_det_appx_pythran.py`, so the blanket entry `Files: *`, BSD-3-Clause,
covers it. The guide for core developers says that "New contributions should
be available under the same license as or be compatible with scikit-image's
license", and that a contributor who is not the copyright holder should "ask
the original authors for approval and include their names in `LICENSE.txt`"
(`doc/source/development/core_developer.md`, item 5).

OpenSURF is under the GPL. The header of each source file states: "This
library is distributed under the GNU GPL"
([`gussmith23/opensurf`](https://github.com/gussmith23/opensurf)). The GPL is
not compatible with distribution under BSD terms alone.

The other implementations of §8 have permissive licences: dlib is under the
Boost Software License, and mahotas, a port of dlib, states "License: MIT" for
its own code. OpenCV's SURF is in the `nonfree` part of `opencv_contrib`.

#### How close the code is to OpenSURF

The table puts `FastHessian::buildResponseLayer` and `BoxIntegral` from
OpenSURF beside `_hessian_matrix_det` and `_integ` from scikit-image.

| OpenSURF (C++) | scikit-image (Python) |
| --- | --- |
| `int b = (rl->filter - 1) / 2;` | `s2 = (size - 1) // 2` |
| `int l = rl->filter / 3;` | `s3 = size // 3` |
| `int w = rl->filter;` | `w = size` |
| `float inverse_area = 1.f/(w*w);` | `w_i = 1.0 / size / size` |
| `BoxIntegral(img, r - l + 1, c - b, 2*l - 1, w)` | `_integ(img, r - s3 + 1, c - s2, 2 * s3 - 1, w)` |
| `BoxIntegral(img, r - l + 1, c - l / 2, 2*l - 1, l)*3` | `_integ(img, r - s3 + 1, c - s3 // 2, 2 * s3 - 1, s3)`, then `mid - 3 * side` |
| `Dxy`: boxes at `(r - l, c + 1)`, `(r + 1, c - l)`, `(r - l, c - l)`, `(r + 1, c + 1)`, each `l × l` | boxes `bl`, `tr`, `tl`, `br` at the same four offsets |
| `Dxx * Dyy - 0.81f * Dxy * Dxy` | `dxx * dyy - 0.81 * (dxy * dxy)` |
| `return std::max(0.f, A - B - C + D);` | `return max(0.0, ans)` |

The two agree in every offset, in the order of the arguments and in the
`max(0, …)` clamp. Both box-sum functions take a top-left corner and a number
of rows and columns. Three things differ:

* `_integ` does not subtract one from the start indices (§4.2);
* `_integ` moves box corners into the image, where `BoxIntegral` counts zeros
  outside (§4.6);
* scikit-image negates `dxx`, `dyy` and `dxy`, which does not change the
  determinant, and sets `size` from `sigma` instead of from a list of filter
  sizes (§4.3).

The first two differences are the defects of §4.2 and §4.6. They are what a
translation of `BoxIntegral` gives if the translator leaves out the line
"The subtraction by one for row/col is because row/col is inclusive".

#### The opposing reading

The method is public. Bay and colleagues define the filters
[@bay2008speededup_robust], Evans describes the OpenSURF layout in prose and
equations [@evans2009notes_opensurf], and Oyallon and Rabin give the box
domains as equations [@oyallon2015analysis_surf, eqs. 10–11]. The lobe
offsets follow from that description. dlib and
mahotas compute the same filters with permissive licences. Copyright covers
the expression of a program, not the method, and a second implementation of
the same method can look much like the first. The PR of 2014 cites only the
SURF paper, and says the code approximates the determinant "as described
here".

Against that reading: the scikit-image code keeps OpenSURF's arrangement of
`b`, `l`, `w` and `inverse_area`, its argument order for each box, the
`max(0, …)` clamp, which is not in any of the papers, and the weight written as
`0.81` rather than $0.9^2$. That is the shape of a line-by-line translation.

#### What the project record shows

A search on 2026-11-27 found no discussion of the question:

* no commit message in the scikit-image history mentions OpenSURF;
* no file in the tree mentions OpenSURF or Evans in this connection;
* the commits by the original author, from `f456245d5` (2014-03-14) on,
  and the later commits to the file (2015 to 2026, including the Pythran
  conversion in #3226), add no attribution or licence note;
* the comments, review comments and reviews of #930, #3226 and #1414 do not
  mention a licence, the GPL, OpenSURF, Evans or copyright;
* GitHub search of scikit-image issues and PRs for "opensurf", "SURF GPL",
  "SURF license", "SURF patent", "hessian_det_appx license" and
  "blob_doh license" finds nothing.

The project has acted on GPL code before. Commit
[`32c0b1ca9`](https://github.com/scikit-image/scikit-image/commit/32c0b1ca979fcfd201a435f13851666c01247941)
(2013) is "Rewrite histogram operations to remove GPL license on Perreault's
code". The record also has a note on patents: the old `skimage/io/sift.py`
says that SIFT and SURF "are patent encumbered", and issue #2488 (2017) points
to it. No one connected that note to `blob_doh` or `hessian_matrix_det`. This
notebook does not examine the patent.

#### Consequences for the fix

An earlier version of the fix in §9.1 kept the structure of OpenSURF, and the
author of this notebook read the OpenSURF source. Two ways to settle the
question:

* a rewrite from the papers only, by someone who has not read the OpenSURF
  source, stated as such in the PR. `surf_layout` gives the filters as boxes
  from the equations of Oyallon and Rabin, and `box_sums` gives a box sum
  with zero outside the image. Both can serve as the specification, with the
  tests of §7;
* a port of the dlib layout, under the Boost Software License, with dlib
  named in `LICENSE.txt`, as the developer guide asks.

The `surf-fixes` branch of scikit-image, and §9.1 now, take the second way.
`on_surf_dlib_port.md` shows that the port makes dlib's box calls, in
dlib's order, and keeps none of the forms of the OpenSURF code. Its author
had read the OpenSURF source, so it is a port of dlib, not a clean-room
implementation. Four of its five changes from dlib give OpenSURF's
behaviour: every pixel computed, zero outside the image, no clamp of the
determinant, and no sign of the Laplacian (`on_surf_dlib_port.md` §8).
These keep what `hessian_matrix_det` already returns, a signed determinant
at every pixel.

## 10. What each fix costs

### 10.1 `blob_doh` on disks of known centre

The disks of `test_blob_doh` in scikit-image, of radius 20, 30, 40 and 50,
on a 512 × 512 image. `use_det` puts a determinant function in place of
`_hessian_matrix_det` inside `blob_doh`:

```{code-cell} ipython3
def use_det(det_func):
    """A `_hessian_matrix_det` replacement that calls `det_func(image, sigma)`."""
    def from_integral(integral, sigma):
        image = np.diff(np.diff(np.pad(integral, ((1, 0), (1, 0))), axis=0), axis=1)
        return det_func(image, sigma)
    return from_integral


def run_blob_doh(det_func, image, **kwargs):
    """`blob_doh` with `det_func`, or with the installed code if None."""
    original = blob_module._hessian_matrix_det
    if det_func is not None:
        blob_module._hessian_matrix_det = use_det(det_func)
    try:
        return blob_doh(image, **kwargs)
    finally:
        blob_module._hessian_matrix_det = original


def shift_only(image, sigma):
    """The current layout without the one-pixel shift."""
    return det_from(image, skimage_layout(sigma, shift=0))


DISKS = [((400, 130), 20), ((460, 50), 30), ((100, 300), 40), ((200, 350), 50)]
test_image = np.ones((512, 512), dtype=np.uint8)
for (r, c), radius in DISKS:
    test_image[disk(512, radius, (r, c)) > 0] = 255

rows = []
for name, func in (("current", None), ("shift fix only", shift_only), ("fixed", surf_det)):
    blobs = run_blob_doh(func, test_image, min_sigma=1, max_sigma=60,
                         num_sigma=10, threshold=0.05)
    for (r, c), radius in DISKS:
        b = blobs[np.argmin(np.hypot(blobs[:, 0] - r, blobs[:, 1] - c))]
        rows.append({"route": name, "radius": radius,
                     "row error": int(b[0] - r), "column error": int(b[1] - c),
                     "sigma": round(float(b[2]), 2)})
show_table(pd.DataFrame(rows))
```

For the disks of radius 20, 30 and 40, the current code finds each centre one
pixel up and to the left, and the shift fix finds it exactly. The full fix
finds two of the three exactly and the third one pixel up and to the left.
For the disk of radius 50, every route is 2 to 3 pixels off. The test in
scikit-image allows 4 pixels, so it does not see the shift. The `sigma` of
each blob, and so its radius, is the same for all three routes.

### 10.2 `blob_doh` on `coins`

The docstring example of `blob_doh`:

```{code-cell} ipython3
coins = ski.data.coins()
before = run_blob_doh(None, coins)
rows = []
for name, func in (("shift fix only", shift_only), ("fixed", surf_det)):
    after = run_blob_doh(func, coins)
    nearest = np.array([after[np.argmin(np.hypot(*(after[:, :2] - b[:2]).T))]
                        for b in before])
    moved = nearest[:, :2] - before[:, :2]
    rows.append({"route": name, "blobs before": len(before), "blobs after": len(after),
                 "moved by exactly (1, 1)": int((moved == 1).all(axis=1).sum()),
                 "same sigma": int((nearest[:, 2] == before[:, 2]).sum()),
                 "largest move (pixels)": float(np.hypot(*moved.T).max())})
show_table(pd.DataFrame(rows), floatfmt=".3g")
```

The shift fix alone moves 16 of the 17 blobs by one pixel down and to the
right, with the same `sigma`. The other blob moves by 5.8 pixels and changes
`sigma`. The full fix also changes the filters at the non-SURF sizes, which
here are 8 of the 10 `sigma` of `blob_doh` (§4.3). It finds 18 blobs instead
of 17, keeps the `sigma` of 11 of the 17, and moves blob centres by up to 4.1
pixels.

(sec-doh-discs)=
### 10.3 The disks of `on_blob_dog.md`

`on_blob_dog.md` §3 reports radii 5% to 33% too large for four disks, with
`num_sigma=30` from 2 to 30, and §5 reports every centre at (−1, −1). The
same image, with the routes of §9.1. "Fixed, all planes" puts `surf_det` into
`blob_doh` unchanged; "fixed, one plane per lobe" is `blob_doh_lobes`:

```{code-cell} ipython3
DISCS = [(80, 80, 6), (80, 220, 10), (220, 80, 16), (220, 260, 24)]
discs = np.zeros((400, 400))
for r0, c0, radius in DISCS:
    discs[disk(400, radius, (r0, c0)) > 0] = 1.0
settings = dict(min_sigma=2, max_sigma=30, num_sigma=30, threshold=0.005)
routes = {
    "current": run_blob_doh(None, discs, **settings),
    "shift fix only": run_blob_doh(shift_only, discs, **settings),
    "fixed, all planes": run_blob_doh(surf_det, discs, **settings),
    "fixed, one plane per lobe": blob_doh_lobes(discs, **settings),
}
rows = []
for name, blobs in routes.items():
    for r0, c0, radius in DISCS:
        b = blobs[np.argmin(np.hypot(blobs[:, 0] - r0, blobs[:, 1] - c0))]
        rows.append({"route": name, "radius": radius,
                     "offset": (int(b[0] - r0), int(b[1] - c0)),
                     "sigma": round(float(b[2]), 2),
                     "sigma / radius - 1": f"{b[2] / radius - 1:+.0%}"})
show_table(pd.DataFrame(rows))
```

The shift fix puts every centre right and leaves the radii as they were. With
all planes, the fixed filters give much larger `sigma` than the radius,
because of the false scale peaks of §4.8. With one plane per lobe, every
centre is right and the reported `sigma` is the odd lobe just above the
radius.

### 10.4 Other costs

* `hessian_matrix_det(..., approximate=True)` changes at every pixel. At the
  SURF sizes the values move by one pixel; at other sizes the values change.
* For `sigma` below 2 the fixed lobe is 1, a 3 × 3 filter. The current code
  gives NaN or zeros there (§4.7).
* The test `test_hessian_matrix_det` in scikit-image checks that a 5 × 5
  image gives a determinant near zero at `sigma` 5. It does not depend on the
  layout.

(sec-doh-usage)=
### 10.5 Who calls the box route

A fix changes the output for every caller of the box route (§10.4). This
section records who those callers are, outside scikit-image. It is a record
of a search on 2026-11-28, not a computation, so no cell re-derives it.

The search was GitHub code search for `hessian_matrix_det`, split by file
size and type so that each query returned fewer than 1000 results. It found
2523 files. Of these, 2381 were copies of scikit-image itself (paths under
`skimage/`, `site-packages/` and the like), type stubs, documentation, HTML
and packaging files. The other 142 Python files and notebooks were fetched
and their calls parsed. A call uses the box route when it passes
`approximate=True`, or when it does not pass `approximate`, because `True`
is the default.

Calls of the public `hessian_matrix_det` that use the box route:

| repository | files | calls | `approximate` | `sigma` |
| --- | --- | --- | --- | --- |
| [Cocopyth/AMFtrack](https://github.com/Cocopyth/AMFtrack/blob/d613e7e7f0410cb7a5256a962f6a52aaa9d39a13/amftrack/pipeline/functions/image_processing/extract_skel.py) | 4 (1 module, 3 notebooks) | 5 | default | 15, 20 |
| [Cocopyth/AMFTravellingWave](https://github.com/Cocopyth/AMFTravellingWave/blob/1fe7dc3062e0157812665a6fe335527afb7b6eb0/amftrack/pipeline/functions/image_processing/extract_skel.py) | 1 | 1 | default | 20 |
| [TTPO100AJIEX/CVTDA](https://github.com/TTPO100AJIEX/CVTDA/blob/24649e23af84ce318b019fbd895eb7d09e0610bf/experiments/cifar10_dataset/14__experiment.ipynb) | 2 notebooks | 3 | default | default, 1 |
| [nonplayer47361/opencv_tuning](https://github.com/nonplayer47361/opencv_tuning/blob/0e57d684ffef838667ed7789f53f675729dc8f35/src/bayesian_optimizer.py) | 2 | 2 | default | a tuned parameter, default 1.0 |
| [Richard-Yan-UCB/datasci281_final_project](https://github.com/Richard-Yan-UCB/datasci281_final_project/blob/6bcba89df30952a75ec65501f9b445e5e44d7a8f/utils.py) | 1 | 2 | default | variables |
| [paulhfu/RLForSeg](https://github.com/paulhfu/RLForSeg/blob/4ff4b830357b97d5e7f3839f6540552cbad11046/utils/affinities.py) | 1 | 1 | default | a variable |
| [DrThiruCSomasundaram/PhDdataanalysis](https://github.com/DrThiruCSomasundaram/PhDdataanalysis/blob/66b8d2353032bb4be738f9139ce65e2dbb95d4a5/Image%20editor.py) | 1 | 1 | default | 1 |
| [missingbulb/RaunTruth](https://github.com/missingbulb/RaunTruth/blob/603c2d63e82fbd0a4e8522e2fc3eaeb5ccb77193/dev/tools/trick_pool/tricks.py) | 1 | 1 | default | `max(0.5, ...)` |
| [sdfunni/SingleOrigin](https://github.com/sdfunni/SingleOrigin/blob/38540b0188ca5b44947e79fe0ffdb3bd002892cf/SingleOrigin/utils/peakfit.py) | 1 | 1 | default | a loop variable |
| [saumy-github/mirra-mvp](https://github.com/saumy-github/mirra-mvp/blob/1655249e9eb8f7dc68de02efad22b33ca2c52b7c/product_ingestion/cv_shape.py) | 1 | 1 | default | 2.0 |
| [jaaumeadrover/BSc-Computer-Science-UIB](https://github.com/jaaumeadrover/BSc-Computer-Science-UIB/blob/8ece4d4e44095f294056b47a50e9fa5332b90e08/04_year/03_deep_learning/assignments/landscape_img_classification/testing/hessian_matrix_det.ipynb) | 1 notebook | 1 | `True`, in a parameter dict | 1 |

That is 19 calls in 16 files of 11 repositories, from 10 owners. Only one
call states `approximate=True`; the others take the default. One call
outside scikit-image passes `approximate=False`
([furuse-kazufumi/fullseye](https://github.com/furuse-kazufumi/fullseye/blob/7fa652573fb95d8ef053f8bccd9554c372720ab2/backends.py)).
Several callers use `sigma` of 1 or 2, where the box filters are least
accurate (§4.3, §4.7). The RaunTruth call can pass `sigma` 0.5, where the
current code returns zeros (§4.7).

Three projects call the private `_hessian_matrix_det` directly, to build a
cube of their own as `blob_doh` does:

* [EPFL-VILAB/omnidata](https://github.com/EPFL-VILAB/omnidata/blob/f69ff3aedf983cb34c490a0afd0d29fdd83f6a1c/omnidata_annotator/scripts/create_keypoints_2d_images.py),
  `create_keypoints_2d_images.py`. The same file is in four other
  repositories that copy omnidata.
* [BMCV/SuperDSM](https://github.com/BMCV/SuperDSM/blob/f92eb57eddd8ed14e5372e1d243d67998ddb6046/superdsm/pipeline.py),
  `pipeline.py`, as `skimage.feature.blob._hessian_matrix_det`.
* [mKamiAsim/pixelnexus](https://github.com/mKamiAsim/pixelnexus/blob/533823cd355135485b87922979b030c73fd52f2b/engine/src/pixelnexus_engine/filters/blobs.py),
  `filters/blobs.py`.

These callers see the fix too, and a change to the private name or its
arguments breaks them.

The count leaves out:

* the scikit-image benchmark, and copies of it and of `test_corner.py` in
  other repositories;
* a fork of scikit-image (bipxx/scikit-imagexx);
* five repositories that call `_hessian_matrix_det` only from a copy of the
  scikit-image `blob.py`;
* 97 files that name `hessian_matrix_det` without calling it, in imports,
  lists of names or text; 86 of them are in the two Cocopyth repositories;
* one notebook that shows only the traceback of a `blob_doh` call.

The counts are a lower bound. GitHub code search is incomplete: the totals
it reported for one query changed while the results were being read, from
342 to 416 files and from 533 to 778 files in two size bands. It searches
default branches, and it leaves out most forks. The search does not cover
callers of `blob_doh`, which use the box route through it.

## 11. What not to do

* **Do not replace the box route with `approximate=False` inside
  `blob_doh`.** `approximate=False` returns $\det H$ without scale
  normalisation. A blob detector needs $\sigma^4 \det H$ to compare scales
  (§4.1), so a direct swap changes which scale wins.
* **Do not treat $0.4\,L$ as exact.** It is a convention of the SURF paper.
  On this corpus the filters match $0.55\,L$ better (§6).
* **Do not change the 0.81 weight.** The best weight is close to it, and the
  gain in accuracy is small (§6.1).
* **Do not quantise the scale without removing duplicate planes.** Identical
  planes make false scale peaks in `blob_doh` (§4.8, §10.3).
* **Do not take "odd size and odd lobe" as the condition for a valid filter.**
  Sizes 5, 11, 17 and 23 meet it, and respond to a constant image (§4.3).
* **Do not fix only the border or the clamp.** The shift and the non-SURF
  sizes change every interior pixel; the border defect changes only a band
  of `lobe + 1` pixels.

(sec-doh-summary)=
## 12. Summary

| defect | where | effect | measured in |
| --- | --- | --- | --- |
| boxes moved by (1, 1) | `_integ` lacks the `- 1` of OpenSURF's `BoxIntegral` | every output is one pixel up and to the left of its cause; `blob_doh` centres off by (−1, −1) | §4.2, §10.1 |
| `size = int(3 * sigma)` not a SURF size | `_hessian_matrix_det` | responds to a constant and to a ramp; not centred; for 5/6 of the `sigma` range | §4.3, §7 |
| normalisation $1/\text{size}^2$ at non-SURF sizes | `_hessian_matrix_det` | saw-tooth response over `sigma`; several local maxima for one disk | §4.4 |
| `size += 1` after use | `_hessian_matrix_det` | no effect; dead code | §4.3 |
| `max(0.0, ans)` clamp | `_integ`, from OpenSURF | wrong or zero output for images with negative values | §4.5 |
| corner clipped before the extent is added | `_integ` | at the top and left, boxes slide inward and row 0 and column 0 are never summed; the bottom and right are zero padding | §4.6 |
| identical scale planes | `blob_doh` with a fine `num_sigma`; any coarser quantisation | false scale peaks; `blob_doh` reports too large a `sigma` | §4.8, §10.3 |
| `sigma < 1` | `_hessian_matrix_det` | zeros; NaN below 1/3 | §4.7 |
| `sigma` called a Gaussian standard deviation | docstrings | the filters match a Gaussian of about $0.55\,\sigma$; SURF states $0.4\,L$; scale selection picks a larger `sigma` than the exact determinant | §4.1, §6, §6.2, §9.2 |

The fix of §9.1, with one plane per lobe in `blob_doh`, removes all but the
last, which is a decision about the meaning of `sigma`.

Separately, the current code is a close translation of GPL-licensed OpenSURF
code, with no attribution and no record of a licence check (§9.3).

(sec-doh-blob-dog)=
### 12.1 The conclusions of `on_blob_dog.md`

| `on_blob_dog.md` (version `7e3b2b2`) | conclusion | verdict | here |
| --- | --- | --- | --- |
| §5, D1 | every blob one pixel up and left; `_integ` sums one pixel down and right of the corner it names | holds | §4.2, §10.1, §10.3 |
| §5, D2 | the dead `size += 1`; filters off-centre at even `size` and at `size` 1 mod 6 | holds | §4.3 |
| §5, D2 fix | use odd sizes that are not 1 mod 6: 3, 5, 9, 11, … | does not hold: sizes 5, 11, 17, 23 respond to a constant; only $3L$ with $L$ odd is valid | §4.3 |
| §6 | `sigma` with the same `int(3 * sigma)` give identical planes; nothing below `sigma` 1 | holds; NaN below 1/3 | §4.7, §4.8 |
| §6, §12 | a fine `num_sigma` "buys duplicate work and nothing else" | does not hold: duplicate planes make false scale peaks | §4.8 |
| §7, D3 | the box route picks too large a scale; recalibrate the constant | the bias holds; on disks the fixed filters give radius + 1.5 px, an offset that no constant removes; the constant is the choice of §9.2 | §6.2, §9.2 |
| §8, D4 | at the border, boxes slide at the top and left and shrink at the bottom and right | holds | §4.6 |
| §8, D4 | the shrunk box is mis-normalised; neither edge matches any `mode` | does not hold: the bottom and right are zero padding, `mode='constant'` | §4.6 |
| §8, D4 fix | pad once under a `mode`, then crop | holds, as one of two fixes | §4.6, §9.1 |
| §9, §12 | exact Gaussian kernels would fix the defects but lose the flat cost | holds; `on_hessian_matrix_det.md` §8 measures the cost | §11 |
| §12 | D1 to D4 are independent and can go in any order | D1 and D4 are independent; D2 and D3 are one decision, and D2 needs the duplicate-plane fix | §9.1, §10.3 |

### Limits

* The scale factor 0.55 and the shape errors come from two fitted
  photographs, `camera` and `coins`, for lobes 3 to 13.
* The comparators are read from source and transcribed. The notebook runs
  none of the other libraries, and does not test their border handling.
* The notebook does not build the patched Pythran module, and so does not
  time it.
* `blob_doh` is tested on the disks of its own test and on `coins` with
  default parameters.
* The count of callers in §10.5 comes from one GitHub code search, which is
  incomplete, and does not cover callers of `blob_doh`.
