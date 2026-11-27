---
title: 'On the blob detectors'
date: 2026-09-14
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

:::{changelog}
2026-11-27 | claude-code:claude-opus-5-5 | Move the `blob_doh` analysis to `on_hessian_det_approx.md`; correct D2, D4 and the duplicate-plane note
2026-11-16 | Matthew Brett | Refactor references
2026-09-22 | Stefan van der Walt, claude-code:claude-opus-5 | Add title and creation date; quote the title
2026-09-15 | Stefan van der Walt, claude-code:claude-sonnet-5 | Port to MyST
2026-09-14 | Matthew Brett | Create workbook
:::

`skimage.feature` offers three blob detectors — `blob_dog`, `blob_log` and
`blob_doh` — with matching signatures and a shared output format. They are
presented as three routes to one answer, differing in speed.

They differ in more than speed. Measured against discs of known radius, one is
accurate to a few per cent, one reports radii a quarter too small, and one
reports them up to a third too large. This notebook explains where each
difference comes from, and which are defects rather than documented
approximations.

Coordinates are in array order throughout: the first index runs down the
picture, the second runs right.

```{code-cell} ipython3
import time

import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
```

```{code-cell} ipython3
import skimage as ski
from skimage.feature import blob_dog, blob_log, blob_doh, hessian_matrix_det
```

```{code-cell} ipython3
# Comparator.
import cv2
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette, validated all-pairs.
C_ONE = "#2a78d6"
C_TWO = "#eb6834"
C_THREE = "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd5"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f4", C_ONE])

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "figure.facecolor": "white"}
)


def bare(ax, title=None):
    """Strip an axis down to the data."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    return ax


def recede(ax, title=None):
    """Keep the ticks, but make the frame recede."""
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    if title is not None:
        ax.set_title(title)
    return ax
```

## 1. A test image with known answers

Four discs, well separated, with radii spanning the useful range.

```{code-cell} ipython3
N = 400
rows, cols = np.indices((N, N))
TRUTH = [(80, 80, 6), (80, 220, 10), (220, 80, 16), (220, 260, 24)]

discs = np.zeros((N, N))
for r0, c0, radius in TRUTH:
    discs[(rows - r0) ** 2 + (cols - c0) ** 2 <= radius**2] = 1.0

fig, ax = plt.subplots(figsize=(3.2, 3.2))
bare(ax, "four discs, radii 6, 10, 16 and 24")
ax.imshow(discs, cmap="gray")
fig.tight_layout()
```

## 2. What a blob detector does

Each detector builds a **scale-normalised response** — a filter applied at many
scales, scaled so that responses at different scales are comparable — and then
looks for maxima in the three-dimensional stack of position and scale. The
position of a maximum gives the blob's centre, and its scale gives the size.

That normalisation is the whole trick. A plain second derivative gets smaller as
`sigma` grows, so without it the smallest scale always wins. Multiplying by the
right power of `sigma` cancels that, and the power differs by filter: `sigma**2`
for a Laplacian, `sigma**4` for a Hessian determinant.

The three detectors differ in which filter they use.

| | filter | documented radius |
| --- | --- | --- |
| `blob_log` | Laplacian of Gaussian, `sigma**2 * div(grad(G)) * f` | `sqrt(2) * sigma` |
| `blob_dog` | difference of two Gaussians, approximating the above | `sqrt(2) * sigma` |
| `blob_doh` | determinant of the Hessian, by box filters | `sigma` |

The radius conventions really do differ, and they are documented that way. Every
comparison below uses each detector's own convention, taken from its docstring.

```{code-cell} ipython3
CONVENTION = {"blob_dog": np.sqrt(2), "blob_log": np.sqrt(2), "blob_doh": 1.0}
```

## 3. What each one reports

```{code-cell} ipython3
found = {
    "blob_dog": blob_dog(discs, min_sigma=2, max_sigma=30, threshold=0.05),
    "blob_log": blob_log(discs, min_sigma=2, max_sigma=30, num_sigma=30,
                         threshold=0.05),
    "blob_doh": blob_doh(discs, min_sigma=2, max_sigma=30, num_sigma=30,
                         threshold=0.005),
}


def matched(name, r0, c0, window=400):
    """The detection nearest a known centre, as a radius in pixels."""
    near = [b for b in found[name] if (b[0] - r0) ** 2 + (b[1] - c0) ** 2 < window]
    if not near:
        return np.nan
    best = min(near, key=lambda b: (b[0] - r0) ** 2 + (b[1] - c0) ** 2)
    return best[2] * CONVENTION[name]


print(f"{'true radius':>12}" + "".join(f"{n:>22}" for n in found))
for r0, c0, radius in TRUTH:
    line = f"{radius:>12}"
    for name in found:
        got = matched(name, r0, c0)
        line += f"{got:>13.1f} ({(got - radius) / radius:+.0%})"
    print(line)
print()
print("blobs found, against four present:",
      {n: len(b) for n, b in found.items()})
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
for ax, (name, blobs) in zip(axes, found.items()):
    bare(ax, name)
    ax.imshow(discs, cmap="gray")
    for r0, c0, radius in TRUTH:
        ax.add_patch(plt.Circle((c0, r0), radius, fill=False, color=C_THREE,
                                lw=1.2, ls=":"))
    for b in blobs:
        ax.add_patch(plt.Circle((b[1], b[0]), b[2] * CONVENTION[name],
                                fill=False, color=C_TWO, lw=1.4))
fig.suptitle("dotted green: the true disc.   orange: what the detector reports",
             y=1.03)
fig.tight_layout()
```

`blob_log` traces the discs closely. `blob_dog` sits consistently inside them and
`blob_doh` consistently outside. Neither of those is noise: both biases are
systematic and repeat at every radius. [`on_hessian_det_approx.md` §10.3](on_hessian_det_approx.md#sec-doh-discs)
repeats this `blob_doh` measurement and traces each part of the bias.

## 4. `blob_dog`: a documented approximation

The difference of two Gaussians at `sigma` and `k * sigma` approximates the
scale-normalised Laplacian, and the approximation improves as `k` approaches 1.
`blob_dog` reports the *smaller* of the pair, so the reported `sigma` sits below
the scale the response actually peaked at.

```{code-cell} ipython3
print(f"{'sigma_ratio k':>15}{'reported radius for the r=16 disc':>36}")
for ratio in (1.6, 1.4, 1.2, 1.1, 1.05):
    got = blob_dog(discs, min_sigma=2, max_sigma=30, sigma_ratio=ratio,
                   threshold=0.05)
    near = [b for b in got if (b[0] - 220) ** 2 + (b[1] - 80) ** 2 < 400]
    radius = near[0][2] * np.sqrt(2) if near else np.nan
    print(f"{ratio:>15}{radius:>25.1f} ({(radius - 16) / 16:+.0%})")
```

The bias falls from -28% at the default to about -2%, which identifies it: it
is the `sigma_ratio` approximation, behaving as the theory says it should. It is
not monotone in `k`, because changing `sigma_ratio` also changes which discrete
scales get computed. `sigma_ratio`
defaults to 1.6, the value
Lowe uses in SIFT [@lowe2004distinctive_image], chosen to
trade accuracy for the number of scales that must be computed.

So `blob_dog`'s error is a parameter the caller controls, documented in the
signature. It is a deliberate approximation, not a defect. OpenCV's `SIFT`
makes the same trade for the same reason.

## 5. `blob_doh`: box filters over an integral image

`blob_doh` never computes a Gaussian derivative. It builds an **integral
image**, in which every pixel holds the sum of everything above and to the
left, so the sum over any rectangle costs four lookups regardless of its size.
It then approximates each second derivative with a few rectangles. That is
SURF [@bay2006surf_speeded], and its point is that the cost does not grow with
`sigma`.

[`on_hessian_det_approx.md`](on_hessian_det_approx.md) is the full analysis of these box filters, their defects and a
fix. It replaces the analysis that this section held before. The defects, with
the section that measures each:

| defect | effect on `blob_doh` | in `on_hessian_det_approx.md` |
| --- | --- | --- |
| D1: `_integ` sums each box one pixel down and right of its corner | every centre one pixel up and left | [§4.2](on_hessian_det_approx.md#sec-port) |
| D2: `size = int(3 * sigma)` is a SURF size, $3L$ with $L$ odd, for one sixth of the `sigma` range | filters off-centre, or not zero-sum, so they respond to a constant | [§4.3](on_hessian_det_approx.md#sec-doh-geometry) |
| D3: the scale that `blob_doh` picks | radii too large; on disks the fixed filters give radius + 1.5 px | [§6.2](on_hessian_det_approx.md#sec-doh-scale) |
| D4: `_integ` clips the corner before it adds the extent | boxes slide inward at the top and left; the bottom and right are zero padding | [§4.6](on_hessian_det_approx.md#sec-doh-border) |
| identical scale planes | false scale peaks when `num_sigma` is fine | [§4.8](on_hessian_det_approx.md#sec-doh-quantised) |

The earlier version of this section came to three conclusions that do not
hold; [§12.1](on_hessian_det_approx.md#sec-doh-blob-dog) of that notebook gives the evidence:

* "odd and not 1 mod 6" sizes are symmetric, but sizes 5, 11, 17 and 23 do
  not sum to zero, so only the SURF sizes $3L$, with $L$ odd, are valid;
* at the bottom and right edges the current code is exact zero padding, not a
  mis-normalised filter;
* duplicate scale planes do more than duplicate work: they change the `sigma`
  that `blob_doh` reports.

The positions of the three detectors on the four discs:

```{code-cell} ipython3
DETECTORS = {"blob_dog": lambda im: blob_dog(im, min_sigma=2, max_sigma=30,
                                             threshold=0.05),
             "blob_log": lambda im: blob_log(im, min_sigma=2, max_sigma=30,
                                             num_sigma=30, threshold=0.05),
             "blob_doh": lambda im: blob_doh(im, min_sigma=2, max_sigma=30,
                                             num_sigma=30, threshold=0.005)}

print("offset of the reported centre from the true centre, in pixels")
print(f"{'true radius':>12}" + "".join(f"{n:>20}" for n in DETECTORS))
for r0, c0, radius in TRUTH:
    line = f"{radius:>12}"
    for name, run in DETECTORS.items():
        near = [b for b in run(discs)
                if (b[0] - r0) ** 2 + (b[1] - c0) ** 2 < 400]
        if not near:
            line += f"{'missed':>20}"
            continue
        b = min(near, key=lambda b: (b[0] - r0) ** 2 + (b[1] - c0) ** 2)
        line += f"{f'{b[0] - r0:+.0f}, {b[1] - c0:+.0f}':>20}"
    print(line)
```

`blob_dog` and `blob_log` land on the centre exactly. `blob_doh` is one pixel
up and one pixel left, at every radius (D1).

The sections below compare the box route with an exact determinant for cost.
These are the helpers they use:

```{code-cell} ipython3
photo = ski.util.img_as_float(ski.data.camera())[::2, ::2]


def gaussian_taps(sigma, order, trunc=8):
    lw = int(trunc * sigma + 0.5)
    x = np.arange(-lw, lw + 1).astype(float)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    if order == 0:
        return x, g
    if order == 1:
        return x, g * (x / sigma**2)
    return x, g * ((x**2 - sigma**2) / sigma**4)


def corrected_taps(sigma, order, trunc=8):
    """Fix C of `on_hessian.md`: repair the discrete moments."""
    x, g = gaussian_taps(sigma, 0, trunc)
    _, k = gaussian_taps(sigma, order, trunc)
    if order == 0:
        return x, g
    if order == 1:
        return x, k / (k * x).sum()
    k = k - k.sum() * g
    return x, k / ((k * x**2).sum() / 2)


def exact_doh(image, sigma, mode="nearest", trunc=8):
    """Scale-normalised determinant of the Hessian, exact kernels."""
    taps = {o: corrected_taps(sigma, o, trunc)[1] for o in (0, 1, 2)}
    sep = lambda a, b: ndi.correlate1d(
        ndi.correlate1d(image, taps[a], axis=0, mode=mode), taps[b], axis=1, mode=mode)
    hrr, hrc, hcc = sep(2, 0), sep(1, 1), sep(0, 2)
    return (sigma**4) * (hrr * hcc - hrc**2)


def box_doh(image, sigma):
    """The box-filter determinant: `hessian_matrix_det(..., approximate=True)`."""
    return hessian_matrix_det(image, sigma=sigma, approximate=True)


def best_of(f, n=5):
    """Minimum of `n` timed calls, after one warmup call."""
    f()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        times.append(time.perf_counter() - t0)
    return min(times)
```

## 6. What the approximation buys

```{code-cell} ipython3
print(f"{'sigma':>6}{'box filters':>14}{'exact':>11}{'ratio':>9}")
for sigma in (2.0, 4.0, 8.0, 16.0):
    t_box = best_of(lambda: box_doh(photo, sigma))
    t_exact = best_of(lambda: exact_doh(photo, sigma))
    print(f"{sigma:>6}{t_box * 1e3:>11.2f} ms{t_exact * 1e3:>8.2f} ms"
          f"{t_exact / t_box:>8.1f}x")
```

This is the trade, and it is a real one. The box cost is flat: the same
rectangle count whatever `sigma` is. The exact cost grows linearly with `sigma`,
because the kernel does. At `sigma = 16` the gap is about 35-fold and it
keeps widening.

So replacing the box filters with exact kernels would fix the scale bias and the
border, and destroy the reason `blob_doh` exists. The box defects have a fix
that keeps the flat cost ([`on_hessian_det_approx.md` §9.1](on_hessian_det_approx.md#sec-doh-fix)). `blob_log` is already the
accurate Gaussian-based detector; a `blob_doh` that computed exact Gaussian
derivatives would be a slower `blob_log` under a different name.

## 7. Do the `hessian_matrix` fixes reach it?

No. `on_hessian.md` finds a border defect in `hessian_matrix` that reaches
`frangi`, `sato`, `meijering` and `hessian`, and proposes fixes A and C. None of
them touches `blob_doh`, which calls `_hessian_matrix_det` directly.

`hessian_matrix_det` is the function to watch, because it has two paths.

```{code-cell} ipython3
print(f"{'sigma':>6}{'approximate=True':>20}{'approximate=False':>21}")
for sigma in (1.0, 2.0, 4.0):
    pad = 2 * int(8 * sigma + 0.5) + 1
    big = np.pad(photo, pad, mode="edge")
    row = ""
    for approximate in (True, False):
        direct = hessian_matrix_det(photo, sigma=sigma, approximate=approximate)
        reference = hessian_matrix_det(big, sigma=sigma,
                                       approximate=approximate)[pad:-pad, pad:-pad]
        gap = np.abs(direct - reference)
        row += f"{gap.max() / max(np.abs(reference).max(), 1e-12):>19.1%}"
    print(f"{sigma:>6}{row}")
```

`approximate=False` routes through `hessian_matrix`, so it inherits the border
defect and fixes A and C repair it. `approximate=True` is the box route, with a
border problem of its own that no fix in `on_hessian.md` addresses
([`on_hessian_det_approx.md` §4.6](on_hessian_det_approx.md#sec-doh-border)).

## 8. How other libraries do it

OpenCV has no direct equivalent of this trio. `cv2.SimpleBlobDetector` is not a
scale-space method at all — it thresholds the image at a series of levels and
groups the connected components, so it finds regions rather than scales.

```{code-cell} ipython3
params = cv2.SimpleBlobDetector_Params()
params.filterByArea, params.minArea, params.maxArea = True, 50, 5000
params.filterByCircularity = params.filterByConvexity = False
params.filterByInertia = params.filterByColor = False
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect((discs * 255).astype(np.uint8))
print(f"{'true radius':>12}{'SimpleBlobDetector':>22}")
for r0, c0, radius in TRUTH:
    near = [k for k in keypoints
            if (k.pt[1] - r0) ** 2 + (k.pt[0] - c0) ** 2 < 400]
    got = near[0].size / 2 if near else np.nan
    print(f"{radius:>12}{got:>13.1f} ({(got - radius) / radius:+.0%})")
```

It recovers the radii of solid discs well, because a disc is exactly what it
looks for. It has no scale-normalised response, so it cannot rank a blob's
strength across scale, and it does not generalise to blobs that are not
threshold-separable from their surroundings.

`cv2.SIFT` uses a difference of Gaussians, like `blob_dog`, with the same
`sigma_ratio` trade for the same reason. The genuine SURF implementation, which
is what `blob_doh` reproduces, sits in `opencv-contrib` behind a build flag and
is not available here. [`on_hessian_det_approx.md` §8](on_hessian_det_approx.md#sec-doh-comparators)
compares the SURF code of OpenSURF, dlib, mahotas and OpenCV from their source.

## 9. What to do

**`blob_dog`.** Nothing. Its bias is the documented `sigma_ratio` approximation,
it shrinks when the caller asks for it to, and it matches what SIFT does.

**`blob_log`.** Nothing. It is the accurate detector of the three.

**`blob_doh`.** Fix the box filters, as
[`on_hessian_det_approx.md` §9.1](on_hessian_det_approx.md#sec-doh-fix) sets out: index the integral image correctly
(D1), use only the SURF sizes $3L$ with $L$ odd (D2), choose a border rule
(D4), and compute one scale plane per lobe. What `sigma` should mean (D3) is a
separate decision ([§9.2](on_hessian_det_approx.md#sec-doh-sigma)). [§12](on_hessian_det_approx.md#sec-doh-summary) of that
notebook lists every defect, and [§12.1](on_hessian_det_approx.md#sec-doh-blob-dog) checks each
conclusion that this notebook made before.

### Would exact kernels be better, given `on_hessian` fixes A and C?

Worth asking, because that notebook's fixes change the arithmetic on the other
side of the comparison. Fix A computes each Hessian element in one
`gaussian_filter` call; Fix C repairs the discrete moments of the second
derivative kernel and, in doing so, removes the need for `truncate = 100` at
small `sigma`. Together they make the exact route both correct at the border
and much faster than it was — the `exact_doh` used in section 6 above is
already built on them.

It still is not the answer for `blob_doh`, and the timing table in section 6 is
why: even with A and C the exact route costs about 3.6x at `sigma = 2` and 35x at
`sigma = 16` on the machine that ran it, because its kernels grow with `sigma` while the box filters do
not. The gap widens without limit. A `blob_doh` built on exact kernels would fix
D1 to D4 at a stroke and would be a slower `blob_log` wearing a different
name.

Where A and C *do* land is `hessian_matrix_det(..., approximate=False)`, which
routes through `hessian_matrix` and inherits both. That path needs nothing from
this notebook. The split to hold onto is: `approximate=False` is fixed by the
`on_hessian` work, `approximate=True` and `blob_doh` need the box fixes of
`on_hessian_det_approx.md`, and no fix crosses between them.

Measured with scikit-image from this working tree, OpenCV 5.0.0, on 400x400
synthetic discs and the 256x256 `camera` photograph.
