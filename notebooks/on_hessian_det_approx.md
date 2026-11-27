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
2026-11-27 | claude-code:claude-opus-5-5 | Create workbook
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

## 1. Setup

```{code-cell} ipython3
import warnings

import numpy as np
import pandas as pd
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

They agree to rounding away from the border. §4.6 measures the border.

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

### 4.6 The border

At the border, `_integ` moves each box corner to the nearest pixel of the
integral image. A box that starts above row 0 then starts at integral row 0,
which leaves out image row 0. OpenSURF treats the image as zero outside.
Compare the current code with its own transcription, which counts zeros
outside (`box_sums`), by distance from the edge:

```{code-cell} ipython3
sigma = 3
ours = det_from(noise, skimage_layout(sigma))
theirs = hessian_matrix_det(noise, sigma)
distance = np.minimum.reduce(np.indices(noise.shape).tolist()
                             + (np.array(noise.shape)[:, None, None] - 1
                                - np.indices(noise.shape)).tolist())
scale = np.abs(theirs).max()
show_table(pd.DataFrame([{"distance from edge": d,
                          "max difference / max value":
                              np.abs(ours - theirs)[distance == d].max() / scale}
                         for d in range(8)]), floatfmt=".2g")
```

The difference is non-zero only within `lobe + 1` pixels of the edge. This is
a defect of the same origin as §4.2. It changes only pixels near the edge.

SURF does not define a value at the border. OpenSURF sets "a border for each
octave such that the largest filter in the octave remains entirely over the
image", because "edge effects can yield anomolous results"
[@evans2009notes_opensurf, §1.5.2]. dlib does the same (§8). `hessian_matrix_det`
returns a value at every pixel, so it must choose a border rule. Zero outside
the image, as in `box_sums`, is the rule of OpenSURF's `BoxIntegral` and of
`hessian_matrix(..., mode='constant')`.

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

## 8. Comparators

Four other implementations of the SURF determinant, read from their source:

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

## 9. Ways forward

### 9.1 The fix

The fix changes three things in `_hessian_det_appx_pythran.py`:

1. `_integ` subtracts one from each start index, as OpenSURF's `BoxIntegral`
   does, and counts zeros outside the image instead of moving the corners.
   It drops the `max(0.0, …)` clamp.
2. The lobe is the odd integer nearest to `sigma`, and `size = 3 * lobe`.
   This keeps the current scale of the SURF sizes (`lobe` 3 for `sigma` 3),
   and so the `blob_doh` relation "radius ≈ `sigma`" (§4.4).
3. The dead `size += 1` goes.

As Python, in the form that Pythran compiles. The function takes the
integral image, as now:

```{code-cell} ipython3
def _integ_fixed(img, r, c, rl, cl):
    """Sum over rows r .. r+rl-1 and columns c .. c+cl-1; zero outside."""
    r1 = min(r, img.shape[0]) - 1
    c1 = min(c, img.shape[1]) - 1
    r2 = min(r + rl, img.shape[0]) - 1
    c2 = min(c + cl, img.shape[1]) - 1
    a = img[r1, c1] if r1 >= 0 and c1 >= 0 else 0.0
    b = img[r1, c2] if r1 >= 0 and c2 >= 0 else 0.0
    d = img[r2, c1] if r2 >= 0 and c1 >= 0 else 0.0
    e = img[r2, c2] if r2 >= 0 and c2 >= 0 else 0.0
    return a - b - d + e


def _hessian_matrix_det_fixed(img, sigma):
    """SURF determinant of Hessian from the integral image `img`."""
    lobe = 2 * int(sigma / 2) + 1  # odd, nearest to sigma
    size = 3 * lobe
    border = (size - 1) // 2
    half = lobe // 2
    w_i = 1.0 / size / size
    height, width = img.shape
    out = np.empty_like(img, dtype=np.float64)
    for r in range(height):
        for c in range(width):
            dxx = (_integ_fixed(img, r - lobe + 1, c - border, 2 * lobe - 1, size)
                   - 3 * _integ_fixed(img, r - lobe + 1, c - half, 2 * lobe - 1, lobe))
            dyy = (_integ_fixed(img, r - border, c - lobe + 1, size, 2 * lobe - 1)
                   - 3 * _integ_fixed(img, r - half, c - lobe + 1, lobe, 2 * lobe - 1))
            dxy = (_integ_fixed(img, r - lobe, c + 1, lobe, lobe)
                   + _integ_fixed(img, r + 1, c - lobe, lobe, lobe)
                   - _integ_fixed(img, r - lobe, c - lobe, lobe, lobe)
                   - _integ_fixed(img, r + 1, c + 1, lobe, lobe))
            out[r, c] = (dxx * dyy - 0.81 * dxy * dxy) * w_i * w_i
    return out
```

It must match `surf_det`, at every pixel including the border, on an image
with negative values:

```{code-cell} ipython3
signed = rng.normal(size=(40, 48))
integral = ski.transform.integral_image(signed)
for sigma in (0.5, 1, 2.5, 3, 4.2, 7):
    loop = _hessian_matrix_det_fixed(integral, sigma)
    vector = surf_det(signed, sigma)
    print(f"sigma {sigma}: lobe {lobe_for(sigma)}, max difference / max value "
          f"{np.abs(loop - vector).max() / np.abs(vector).max():.1e}")
```

The loop does the same number of look-ups per pixel as the current code, plus
four comparisons in each `_integ_fixed`. This notebook does not build the
Pythran module, so it does not time the fix.

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

The fix of §9.1 keeps the structure of OpenSURF, and the author of this
notebook read the OpenSURF source. It does not settle the question. Two ways
to settle it:

* a rewrite from the papers only, by someone who has not read the OpenSURF
  source, stated as such in the PR. `surf_layout` gives the filters as boxes
  from the equations of Oyallon and Rabin, and `box_sums` gives a box sum
  with zero outside the image. Both can serve as the specification, with the
  tests of §7;
* a port of the dlib layout, under the Boost Software License, with dlib
  named in `LICENSE.txt`, as the developer guide asks.

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

### 10.3 Other costs

* `hessian_matrix_det(..., approximate=True)` changes at every pixel. At the
  SURF sizes the values move by one pixel; at other sizes the values change.
* For `sigma` below 2 the fixed lobe is 1, a 3 × 3 filter. The current code
  gives NaN or zeros there (§4.7).
* The test `test_hessian_matrix_det` in scikit-image checks that a 5 × 5
  image gives a determinant near zero at `sigma` 5. It does not depend on the
  layout.

## 11. What not to do

* **Do not replace the box route with `approximate=False` inside
  `blob_doh`.** `approximate=False` returns $\det H$ without scale
  normalisation. A blob detector needs $\sigma^4 \det H$ to compare scales
  (§4.1), so a direct swap changes which scale wins.
* **Do not treat $0.4\,L$ as exact.** It is a convention of the SURF paper.
  On this corpus the filters match $0.55\,L$ better (§6).
* **Do not change the 0.81 weight.** The best weight is close to it, and the
  gain in accuracy is small (§6.1).
* **Do not fix only the border or the clamp.** The shift and the non-SURF
  sizes change every interior pixel; the border defect changes only a band
  of `lobe + 1` pixels.

## 12. Summary

| defect | where | effect | measured in |
| --- | --- | --- | --- |
| boxes moved by (1, 1) | `_integ` lacks the `- 1` of OpenSURF's `BoxIntegral` | every output is one pixel up and to the left of its cause; `blob_doh` centres off by (−1, −1) | §4.2, §10.1 |
| `size = int(3 * sigma)` not a SURF size | `_hessian_matrix_det` | responds to a constant and to a ramp; not centred; for 5/6 of the `sigma` range | §4.3, §7 |
| normalisation $1/\text{size}^2$ at non-SURF sizes | `_hessian_matrix_det` | saw-tooth response over `sigma`; several local maxima for one disk | §4.4 |
| `size += 1` after use | `_hessian_matrix_det` | no effect; dead code | §4.3 |
| `max(0.0, ans)` clamp | `_integ`, from OpenSURF | wrong or zero output for images with negative values | §4.5 |
| corners moved into the image | `_integ` | leaves out row 0 and column 0 near the edge | §4.6 |
| `sigma < 1` | `_hessian_matrix_det` | zeros; NaN below 1/3 | §4.7 |
| `sigma` called a Gaussian standard deviation | docstrings | the filters match a Gaussian of about $0.55\,\sigma$; SURF states $0.4\,L$ | §4.1, §6, §9.2 |

The fix of §9.1 removes all but the last, which is a decision about the
meaning of `sigma`.

Separately, the current code is a close translation of GPL-licensed OpenSURF
code, with no attribution and no record of a licence check (§9.3).

### Limits

* The scale factor 0.55 and the shape errors come from two fitted
  photographs, `camera` and `coins`, for lobes 3 to 13.
* The comparators are read from source and transcribed. The notebook runs
  none of the other libraries, and does not test their border handling.
* The notebook does not build the patched Pythran module, and so does not
  time it.
* `blob_doh` is tested on the disks of its own test and on `coins` with
  default parameters.
