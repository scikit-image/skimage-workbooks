---
title: 'On alternatives to the box filters in blob_doh'
date: 2026-11-28
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

:::{changelog}
2026-11-28 | claude-code:claude-opus-5-5 | Create workbook
:::

`blob_doh` finds blobs at maxima of the determinant of the Hessian over
position and scale. It computes the determinant with the box filters of SURF
on an integral image [@bay2006surf_speeded], because their cost does not grow
with `sigma`. `on_hessian_det_approx.md` lists the defects of that code and a
fix. This notebook asks a different question: can `blob_doh` compute the
determinant from Gaussian smoothing instead, by finite differences (FD) or
with Gaussian-derivative kernels (GD), at an acceptable cost? It measures the
cost, the accuracy on disks of known size, the properties that a
determinant-of-Hessian detector must keep, and the agreement of each route
with an exact detector on fitted photographs.

Coordinates are in array order: $r$ runs down the rows, $c$ across the
columns. Every time in this notebook comes from one run on one machine.
Compare times only within one table.

## 1. Setup

```{code-cell} ipython3
import time

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt

from nbhelper import show_table
```

The subject under test is the installed scikit-image. `blob_doh` and
`blob_log` are the detectors it ships; `_prune_blobs` is the step that
`blob_doh` uses to remove overlapping blobs.

```{code-cell} ipython3
import skimage as ski
from skimage.feature import blob_doh, blob_log, hessian_matrix_det, peak_local_max
from _skimage2.feature.blob import _prune_blobs

print("scikit-image", ski.__version__)
```

```{code-cell} ipython3
# Categorical slots 1-4 of the default palette. validate_palette.js, light
# mode: all checks pass; worst adjacent CVD ΔE 11.8; contrast WARN for slots
# 2-4, so every chart has a legend or a table.
COLOURS = {
    "box, current": "#257de0",
    "FD": "#f47230",
    "GD": "#00b894",
    "box, fixed": "#f2a600",
}
INK, MUTED, GRID = "#191919", "#666666", "#dedcd5"

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "axes.edgecolor": GRID,
     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
     "figure.facecolor": "white"}
)
```

## 2. Helpers

### 2.1 Gaussian determinants

The scale-normalised determinant of the Hessian is
$\sigma^4 (H_{rr} H_{cc} - H_{rc}^2)$, where $H$ is the Hessian of the image
smoothed at scale $\sigma$ [@lindeberg1998feature_detection]. `fd_plane`
computes $H$ by FD, as `hessian_matrix(..., use_gaussian_derivatives=False)`
does: smooth, then apply `np.gradient` twice. It takes the determinant in
closed form. `on_hessian_matrix_det.md` §8 shows that the closed form is many
times faster than `np.linalg.det`, which `hessian_matrix_det(...,
approximate=False)` uses.

```{code-cell} ipython3
def fd_from_smoothed(smoothed, sigma):
    """sigma^4 det H from an image already smoothed at `sigma`, by FD."""
    gr, gc = np.gradient(smoothed)
    grr, grc = np.gradient(gr)
    gcc = np.gradient(gc, axis=1)
    return sigma**4 * (grr * gcc - grc**2)


def fd_plane(image, sigma):
    """Scale-normalised determinant at `sigma`: Gaussian smoothing, then FD."""
    return fd_from_smoothed(ndi.gaussian_filter(image, sigma, mode="nearest"), sigma)


def fd_planes(image, sigmas, cascade=False):
    """One `fd_plane` per sigma; with `cascade`, smooth each from the previous."""
    planes, smoothed, previous = [], image, 0.0
    for sigma in sigmas:
        if cascade:
            step = np.sqrt(sigma**2 - previous**2)
            smoothed = ndi.gaussian_filter(smoothed, step, mode="nearest")
            previous = sigma
        else:
            smoothed = ndi.gaussian_filter(image, sigma, mode="nearest")
        planes.append(fd_from_smoothed(smoothed, sigma))
    return np.stack(planes, axis=-1)
```

GD convolves with sampled Gaussian-derivative kernels whose discrete moments
are restored. `taps` transcribes the kernel of the scikit-image
`hessian-refactor` branch, as `on_hessian_matrix_det.md` §2 does. `on_hessian_matrix_det.md`
§6 shows that GD is exact to rounding from `sigma` 2 on fitted photographs,
where FD has a relative error of 0.5 at `sigma` 1 and 0.06 at `sigma` 4.

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
    centre = len(kernel) // 2
    kernel[centre] = -(kernel[:centre].sum() + kernel[centre + 1:].sum())
    kernel /= (kernel * x**2).sum() / 2
    kernel[centre] = -(kernel[:centre].sum() + kernel[centre + 1:].sum())
    return kernel


def gd_plane(image, sigma):
    """Scale-normalised determinant at `sigma`, with Gaussian-derivative kernels."""
    k = {o: taps(sigma, o) for o in (0, 1, 2)}

    def element(order_r, order_c):
        out = ndi.correlate1d(image, k[order_r], axis=0, mode="nearest")
        return ndi.correlate1d(out, k[order_c], axis=1, mode="nearest")

    return sigma**4 * (element(2, 0) * element(0, 2) - element(1, 1) ** 2)


def hybrid_planes(image, sigmas, switch=4):
    """GD below `switch`, where FD is least accurate; FD at and above it."""
    return np.stack([gd_plane(image, s) if s < switch else fd_plane(image, s)
                     for s in sigmas], axis=-1)


def hybrid_cascade_planes(image, sigmas, switch=4):
    """GD below `switch`; cascade FD at and above it, starting from the image."""
    small = [s for s in sigmas if s < switch]
    large = [s for s in sigmas if s >= switch]
    planes = [gd_plane(image, s) for s in small]
    if large:
        planes += list(np.moveaxis(fd_planes(image, large, cascade=True), -1, 0))
    return np.stack(planes, axis=-1)


def log_plane(image, sigma):
    """Scale-normalised Laplacian of Gaussian, the response of `blob_log`."""
    return -sigma**2 * ndi.gaussian_laplace(image, sigma, mode="nearest")
```

### 2.2 Box determinants

The current box route is the installed `hessian_matrix_det(...,
approximate=True)`, the function that `blob_doh` calls. The fixed box route
is the SURF layout with an odd lobe, centred, with zero outside the image, from
`on_hessian_det_approx.md` §2 and §9.1. `blob_doh` reports `sigma` as the
blob radius; for the fixed route the scale is the lobe width.

```{code-cell} ipython3
def box_sums(image, r0, c0, nr, nc):
    """Sum of `image[r+r0 : r+r0+nr, c+c0 : c+c0+nc]` at every pixel, zero outside."""
    H, W = image.shape
    pad = max(abs(r0), abs(c0), abs(r0 + nr), abs(c0 + nc)) + 1
    integral = np.zeros((H + 2 * pad + 1, W + 2 * pad + 1))
    integral[1:, 1:] = np.pad(image, pad).cumsum(0).cumsum(1)
    R = np.arange(H)[:, None] + pad + r0
    C = np.arange(W)[None, :] + pad + c0
    return (integral[R + nr, C + nc] - integral[R, C + nc]
            - integral[R + nr, C] + integral[R, C])


def fixed_box_plane(image, lobe):
    """SURF determinant for odd `lobe` L, weight 1/(3L)^2, zero outside."""
    L, w = lobe, 1.0 / (3 * lobe) ** 2
    b, h = (3 * L - 1) // 2, (L - 1) // 2
    dxx = (box_sums(image, -L + 1, -b, 2 * L - 1, 3 * L)
           - 3 * box_sums(image, -L + 1, -h, 2 * L - 1, L)) * w
    dyy = (box_sums(image, -b, -L + 1, 3 * L, 2 * L - 1)
           - 3 * box_sums(image, -h, -L + 1, L, 2 * L - 1)) * w
    dxy = (box_sums(image, -L, 1, L, L) + box_sums(image, 1, -L, L, L)
           - box_sums(image, -L, -L, L, L) - box_sums(image, 1, 1, L, L)) * w
    return dxx * dyy - 0.81 * dxy**2


def lobe_for(sigma):
    """The odd lobe width nearest to `sigma`."""
    return 2 * int(sigma / 2) + 1


def current_box_plane(image, sigma):
    """NumPy transcription of the current layout, without its border clipping."""
    size = int(3 * sigma)
    s2, l, w = (size - 1) // 2, size // 3, 1.0 / size**2
    dxy = (box_sums(image, -l + 1, -l + 1, l, l) + box_sums(image, 2, 2, l, l)
           - box_sums(image, -l + 1, 2, l, l) - box_sums(image, 2, -l + 1, l, l)) * w
    dxx = (3 * box_sums(image, -l + 2, -(l // 2) + 1, 2 * l - 1, l)
           - box_sums(image, -l + 2, -s2 + 1, 2 * l - 1, size)) * w
    dyy = (3 * box_sums(image, -(l // 2) + 1, -l + 2, l, 2 * l - 1)
           - box_sums(image, -s2 + 1, -l + 2, size, 2 * l - 1)) * w
    return dxx * dyy - 0.81 * dxy**2
```

`current_box_plane` exists only to time the current layout in the same NumPy
code as the fixed one (§6). Both do eight box sums per pixel.

### 2.3 Detection

All routes share the peak search and pruning of `blob_doh`: maxima of the
cube in a 3 × 3 × 3 neighbourhood above `threshold`, then removal of the
smaller of two overlapping blobs. `detect` takes a cube and the scale of each
plane.

```{code-cell} ipython3
def detect(cube, scales, threshold, overlap=0.5):
    """`blob_doh` peak search and pruning on a ready-made cube."""
    peaks = peak_local_max(cube, threshold_abs=threshold, exclude_border=False,
                           footprint=np.ones((3, 3, 3)))
    if peaks.size == 0:
        return np.empty((0, 3))
    peaks = peaks.astype(float)
    peaks[:, -1] = np.asarray(scales)[peaks[:, -1].astype(int)]
    return _prune_blobs(peaks, overlap)


def detect_route(route, image, sigmas, threshold):
    """Blobs `(r, c, radius)` for one route; each route's own radius convention."""
    if route == "box, current":
        blobs = blob_doh(image, min_sigma=sigmas[0], max_sigma=sigmas[-1],
                         num_sigma=len(sigmas), threshold=threshold)
        return blobs  # radius = sigma, as the docstring says
    if route == "box, fixed":
        lobes = np.unique([lobe_for(s) for s in sigmas])
        cube = np.stack([fixed_box_plane(image, L) for L in lobes], axis=-1)
        return detect(cube, lobes, threshold)  # radius = lobe
    if route == "blob_log":
        blobs = blob_log(image, min_sigma=sigmas[0], max_sigma=sigmas[-1],
                         num_sigma=len(sigmas), threshold=threshold)
    else:
        make = {"FD": lambda: fd_planes(image, sigmas),
                "FD cascade": lambda: fd_planes(image, sigmas, cascade=True),
                "hybrid": lambda: hybrid_planes(image, sigmas),
                "hybrid cascade": lambda: hybrid_cascade_planes(image, sigmas)}[route]
        blobs = detect(make(), sigmas, threshold)
    blobs = blobs.copy()
    blobs[:, 2] *= np.sqrt(2)  # radius = sqrt(2) sigma for a Gaussian scale
    return blobs


def best_time(func, repeats=3):
    """Shortest of `repeats` wall-clock times of `func()`, in ms, after a warm-up."""
    func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return 1e3 * min(times)


def strongest(cube, scales, count, overlap=0.5):
    """Pruned blobs from the fewest strongest peaks that leave at least `count` blobs."""
    peaks = peak_local_max(cube, threshold_abs=1e-9, exclude_border=False,
                           footprint=np.ones((3, 3, 3)))
    peaks = peaks[np.argsort(cube[tuple(peaks.T)])[::-1]].astype(float)
    peaks[:, -1] = np.asarray(scales)[peaks[:, -1].astype(int)]
    lo, hi = count, min(len(peaks), 20 * count)
    while lo < hi:  # bisect on the number of peaks kept
        mid = (lo + hi) // 2
        if len(_prune_blobs(peaks[:mid].copy(), overlap)) >= count:
            hi = mid
        else:
            lo = mid + 1
    return _prune_blobs(peaks[:lo].copy(), overlap)
```

`strongest` replaces the threshold for routes whose response is in other
units: it keeps the strongest peaks until pruning leaves `count` blobs.

The radius conventions differ. `blob_doh` documents radius ≈ `sigma`, which
fits the current box route (`on_hessian_det_approx.md` §6.2). For a Gaussian
scale, the scale-normalised determinant on a disk of radius $R$ peaks at about
$\sigma = 0.7 R$ (`on_hessian_det_approx.md` §6.2), so radius ≈ $\sqrt{2}\,\sigma$,
the convention of `blob_log`. §5 tests both conventions.

### 2.4 An exact gold standard

`gaussian_sum` fits a photograph with a sum of Gaussians, whose smoothed
Hessian has a closed form at every scale [@mobahi2016closed_form];
`on_gaussian_sum_images.md` describes the fit. `fitted_canvas` renders the
fitted sum with a margin of `pad` pixels on every side. The sum decays to
near zero inside the margin, so a filter sees the whole function and the
border mode has no effect. `exact_planes` gives the exact scale-normalised
determinant of that sum. The peak search of `detect` on the exact planes
gives the gold-standard blobs.

```{code-cell} ipython3
from dataclasses import replace

from scipy.optimize import linear_sum_assignment

import gaussian_sum as gs


def fitted_canvas(name, pad=80):
    """Gaussian-sum fit of a `skimage.data` photograph, rendered with a margin of `pad`."""
    photo = ski.util.img_as_float(getattr(ski.data, name)())
    levels, _ = gs.fit(photo)
    levels = [replace(level, row_centres=level.row_centres + pad,
                      col_centres=level.col_centres + pad) for level in levels]
    shape = (photo.shape[0] + 2 * pad, photo.shape[1] + 2 * pad)
    return levels, shape, gs.render(levels, shape)


def exact_planes(levels, shape, sigmas):
    """Exact sigma^4 det H of the Gaussian sum, one plane per sigma."""
    planes = []
    for sigma in sigmas:
        hrr, hrc, hcc = gs.hessian(levels, shape, sigma)
        planes.append(sigma**4 * (hrr * hcc - hrc**2))
    return np.stack(planes, axis=-1)


def match(gold, found):
    """One-to-one pairs (gold index, found index) within max(2, sigma / 2) pixels."""
    distance = np.hypot(gold[:, None, 0] - found[None, :, 0],
                        gold[:, None, 1] - found[None, :, 1])
    allowed = distance <= np.maximum(2, gold[:, 2] / 2)[:, None]
    g, f = linear_sum_assignment(np.where(allowed, distance, 1e6))
    keep = allowed[g, f]
    return g[keep], f[keep]
```

`match` pairs each gold blob with at most one found blob, nearest first. The
tolerance is half the gold `sigma`, and at least 2 pixels.

## 3. The problem on one disk

A disk of radius 10, centred on pixel (80, 80), blurred slightly so that it
has no flat plateau of tied maxima. The response at the disk centre, over
`sigma`, for the current box route, FD and GD. Each curve is divided by its
own maximum:

```{code-cell} ipython3
N, CENTRE, RADIUS = 161, 80, 10
rr, cc = np.indices((N, N)) - CENTRE
one_disk = ndi.gaussian_filter(((rr**2 + cc**2) <= RADIUS**2).astype(float), 1)
sweep = np.arange(1, 20.01, 0.25)
centre_response = {
    # The current code puts the response to the centre at (79, 79).
    "box, current": [hessian_matrix_det(one_disk, s)[CENTRE - 1, CENTRE - 1] for s in sweep],
    "FD": [fd_plane(one_disk, s)[CENTRE, CENTRE] for s in sweep],
    "GD": [gd_plane(one_disk, s)[CENTRE, CENTRE] for s in sweep],
}
fig, ax = plt.subplots(figsize=(6.5, 3), constrained_layout=True)
for name, values in centre_response.items():
    values = np.array(values)
    ax.plot(sweep, values / values.max(), color=COLOURS[name], lw=2, label=name)
ax.axvline(RADIUS / np.sqrt(2), color=INK, lw=0.8, ls=":")
ax.axvline(RADIUS, color=INK, lw=0.8, ls="--")
ax.set_xlabel("sigma")
ax.set_ylabel("centre response / its maximum")
ax.set_title("Dotted: radius / sqrt(2). Dashed: radius.")
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False)
plt.show()
for name, values in centre_response.items():
    print(f"{name:13s} peak at sigma {sweep[int(np.argmax(values))]:5.2f}")
```

The FD and GD curves are smooth, with one maximum near $R / \sqrt{2}$. The box
curve is a saw-tooth with its largest value near $R$
(`on_hessian_det_approx.md` §4.4 explains the saw-tooth).

## 4. The mechanisms

### 4.1 Cost per scale

The box route does about 30 look-ups per pixel at any `sigma`. FD does one
separable Gaussian smoothing, whose kernel has about $8\sigma + 1$ taps, then
a fixed number of differences. GD does six separable passes with kernels of
about $16\sigma + 1$ taps. Times for one scale on `camera`, 512 × 512, in
milliseconds:

```{code-cell} ipython3
camera = ski.util.img_as_float(ski.data.camera())
rows = []
for sigma in (1, 4, 8, 16, 30):
    rows.append({
        "sigma": sigma,
        "box, current": best_time(lambda: hessian_matrix_det(camera, sigma)),
        "approximate=False": best_time(
            lambda: hessian_matrix_det(camera, sigma, approximate=False)),
        "FD, closed form": best_time(lambda: fd_plane(camera, sigma)),
        "GD": best_time(lambda: gd_plane(camera, sigma)),
    })
per_scale = pd.DataFrame(rows)
show_table(per_scale, floatfmt=".1f")
```

The box time is the same at every `sigma`. FD with the closed form costs the
same as the box route at `sigma` 1, and its cost grows with `sigma`.
`approximate=False` costs much more at every `sigma`;
`on_hessian_matrix_det.md` §8 traces most of that cost to `np.linalg.det`. GD costs several times FD.

### 4.2 Cascade smoothing

The Gaussian has the semigroup property: smoothing at $\sigma_a$ and then at
$\sigma_b$ is smoothing at $\sqrt{\sigma_a^2 + \sigma_b^2}$
[@lindeberg1998feature_detection]. So each scale can start from the previous
one, with a kernel of width $\sqrt{\sigma_k^2 - \sigma_{k-1}^2}$, which is
smaller than $\sigma_k$. For sampled kernels the property holds only
approximately, and the `mode` is applied once per step. Compare the cascade
with direct smoothing, away from the border, on a test image of 12 disks:

```{code-cell} ipython3
H, W = 512, 640
DISK_RADII = np.linspace(5, 30, 12)
DISK_CENTRES = [(64 + 128 * (i // 4), 80 + 160 * (i % 4)) for i in range(12)]
r_grid, c_grid = np.indices((H, W))
disks = np.zeros((H, W))
for (r0, c0), radius in zip(DISK_CENTRES, DISK_RADII):
    disks[(r_grid - r0) ** 2 + (c_grid - c0) ** 2 <= radius**2] = 1.0
disks = ndi.gaussian_filter(disks, 1)

SIGMAS = np.linspace(2, 30, 40)
direct = fd_planes(disks, SIGMAS)
cascade = fd_planes(disks, SIGMAS, cascade=True)
inner = (slice(60, -60), slice(60, -60))
show_table(pd.DataFrame([
    {"sigma": SIGMAS[k],
     "max |cascade - direct| / max |direct|, interior":
         np.abs(cascade[..., k] - direct[..., k])[inner].max()
         / np.abs(direct[..., k][inner]).max()}
    for k in (1, 5, 10, 20, 39)]), floatfmt=".2g")
```

The difference grows with the number of steps, to about $2 \times 10^{-3}$ of
the response after 40 scales. That is small beside the FD error itself
(§4.3), but it is not zero.

### 4.3 FD at small `sigma`

FD under-reports second derivatives at small `sigma`: on fitted photographs
its gain error on the determinant is −0.10 at `sigma` 1 and −0.02 at `sigma` 8
(`on_hessian_matrix_det.md` §5). A scale-normalised detector compares
responses across scales, so an error that depends on `sigma` moves the chosen
scale. GD has no such error from `sigma` 2. The hybrid route uses GD below
`sigma` 4, where it is cheap and FD is least accurate, and FD from 4 up. §7
measures the effect on the response.

## 5. Side by side on disks

The 12 disks of §4.2, with radii from 5 to 30. `sigma` runs over 40 values
from 2 to 30 for every route; the current box route uses 2 to 40, because its
`sigma` is a radius, not a Gaussian scale. The threshold is 0.005 for the box
and determinant routes, and 0.05 for `blob_log`, whose response has other
units.

```{code-cell} ipython3
ROUTES = ("box, current", "box, fixed", "FD", "FD cascade", "hybrid", "hybrid cascade",
          "blob_log")
SIGMAS_BOX = np.linspace(2, 40, 40)
found = {}
for route in ROUTES:
    sigmas = SIGMAS_BOX if route.startswith("box") else SIGMAS
    threshold = 0.05 if route == "blob_log" else 0.005
    found[route] = detect_route(route, disks, sigmas, threshold)

rows = []
for route, blobs in found.items():
    for (r0, c0), radius in zip(DISK_CENTRES, DISK_RADII):
        b = blobs[np.argmin(np.hypot(blobs[:, 0] - r0, blobs[:, 1] - c0))]
        rows.append({"route": route, "radius": radius,
                     "offset": float(np.hypot(b[0] - r0, b[1] - c0)),
                     "radius error": b[2] / radius - 1})
disk_results = pd.DataFrame(rows)
summary = disk_results.groupby("route", sort=False).agg(
    blobs=("radius", "size"),
    largest_offset=("offset", "max"),
    smallest_radius_error=("radius error", "min"),
    largest_radius_error=("radius error", "max"))
summary.insert(0, "blobs found", [len(found[r]) for r in summary.index])
show_table(summary.reset_index(), floatfmt=".3g")
```

`blobs` is the number of disks matched; `blobs found` is the number the route
reported. `largest_offset` is the largest distance, in pixels, between a disk
centre and the nearest reported blob. The radius error is the reported radius
over the true radius, minus 1. The four routes with a Gaussian scale give the
same range of errors. Check whether they choose the same `sigma` for each
disk:

```{code-cell} ipython3
gaussian_routes = ("FD", "FD cascade", "hybrid", "hybrid cascade", "blob_log")
chosen = {route: disk_results[disk_results.route == route]["radius error"].to_numpy()
          for route in gaussian_routes}
for route in gaussian_routes[1:]:
    print(f"{route:11s} same scale as FD for "
          f"{int(np.isclose(chosen[route], chosen['FD']).sum())} of 12 disks")
```

On these disks:

* every route finds all 12 disks and no other blob;
* the current box route puts centres up to 1.4 pixels off, the one-pixel
  shift in both directions; every other route puts them exactly;
* the box routes report radii 3% to 57% too large, the largest errors at the
  smallest disks;
* the routes with a Gaussian scale report radii within −5% to +4%, with
  radius $= \sqrt{2}\,\sigma$. FD, the hybrid and `blob_log` choose the same
  `sigma` for all 12 disks, and the FD cascade and the hybrid cascade for 11
  of 12.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
for route in ("box, current", "box, fixed", "FD"):
    part = disk_results[disk_results.route == route]
    ax.plot(part["radius"], 100 * part["radius error"], "o-", ms=5, lw=2,
            color=COLOURS[route], label=route)
part = disk_results[disk_results.route == "hybrid"]
ax.plot(part["radius"], 100 * part["radius error"], "s--", ms=5, lw=1.5,
        color=COLOURS["GD"], label="hybrid (GD below sigma 4)")
ax.axhline(0, color=INK, lw=0.8)
ax.set_xlabel("disk radius (pixels)")
ax.set_ylabel("radius error (%)")
ax.set_title("Reported radius against the true radius")
ax.grid(color=GRID, lw=0.6)
ax.legend(frameon=False)
plt.show()
```

## 6. How big: cost of the whole detector

Times for the whole detector, with the default `sigma` range of `blob_doh`, 10
values from 1 to 30, on two photographs. The fixed box route is written in
NumPy here, so this table leaves it out; the last cell of this section
compares it with the current layout in the same NumPy code.

```{code-cell} ipython3
DEFAULT_SIGMAS = np.linspace(1, 30, 10)
images = {"coins (303 × 384)": ski.util.img_as_float(ski.data.coins()),
          "camera (512 × 512)": camera}
rows = []
for name, image in images.items():
    row = {"image": name,
           "box, current": best_time(lambda: blob_doh(image))}
    for route, make in (("FD", lambda: fd_planes(image, DEFAULT_SIGMAS)),
                        ("FD cascade", lambda: fd_planes(image, DEFAULT_SIGMAS, True)),
                        ("hybrid", lambda: hybrid_planes(image, DEFAULT_SIGMAS)),
                        ("hybrid cascade",
                         lambda: hybrid_cascade_planes(image, DEFAULT_SIGMAS))):
        row[route] = best_time(lambda: detect(make(), DEFAULT_SIGMAS, 0.01))
    row["blob_log"] = best_time(lambda: blob_log(image, 1, 30, 10, threshold=0.1))
    rows.append(row)
detector_times = pd.DataFrame(rows)
show_table(detector_times, floatfmt=".0f")
ratios = detector_times.set_index("image").div(
    detector_times.set_index("image")["box, current"], axis=0)
show_table(ratios.reset_index(), floatfmt=".2f")
```

The first table is in milliseconds; the second divides each time by the
current box time. The FD detector costs 3.2 to 3.4 times the current box
detector, the cascade about 2 times, and the hybrid a little more than FD.
The hybrid cascade costs about 2.1 to 2.2 times, a little more than the
cascade. `blob_log` costs 8 to 10 times. Where the time goes, for FD on `camera`:

```{code-cell} ipython3
fd_cube = fd_planes(camera, DEFAULT_SIGMAS)
show_table(pd.DataFrame([{
    "FD planes, direct": best_time(lambda: fd_planes(camera, DEFAULT_SIGMAS)),
    "FD planes, cascade": best_time(lambda: fd_planes(camera, DEFAULT_SIGMAS, True)),
    "peak search and pruning": best_time(lambda: detect(fd_cube, DEFAULT_SIGMAS, 0.01)),
    "box planes, current": best_time(
        lambda: [hessian_matrix_det(camera, s) for s in DEFAULT_SIGMAS]),
}]), floatfmt=".0f")
```

The fixed box route does the same work as the current one: eight box sums per
pixel. Time one plane on `camera`, in milliseconds, for the two layouts in
the same NumPy code, and for the installed compiled code. The last column
checks that `current_box_plane` computes the installed response, away from
the border:

```{code-cell} ipython3
inner_camera = (slice(70, -70), slice(70, -70))
rows = []
for sigma in (3, 10, 20):
    installed = hessian_matrix_det(camera, sigma)
    rows.append({
        "sigma": sigma,
        "current, NumPy": best_time(lambda: current_box_plane(camera, sigma)),
        "fixed, NumPy": best_time(lambda: fixed_box_plane(camera, lobe_for(sigma))),
        "current, installed": best_time(lambda: hessian_matrix_det(camera, sigma)),
        "max |NumPy - installed| / max |installed|, interior":
            np.abs(current_box_plane(camera, sigma) - installed)[inner_camera].max()
            / np.abs(installed).max(),
    })
box_times = pd.DataFrame(rows)
box_times.insert(3, "fixed / current, NumPy",
                 box_times["fixed, NumPy"] / box_times["current, NumPy"])
show_table(box_times, floatfmt=".3g")
```

In the same code the fixed layout costs the same as the current one, so a
compiled fixed route would cost the same as the installed box route.

## 7. Properties that ought to hold

### 7.1 One response for one shape at every size

A scale-normalised detector with $\gamma = 1$ gives the same peak response
for the same shape at any size [@lindeberg1998feature_detection]. A Gaussian
blob of unit peak and width $w$, smoothed at $\sigma$, is a Gaussian of
width $s = \sqrt{w^2 + \sigma^2}$ and peak $w^2 / s^2$
[@mobahi2016closed_form]. At its centre $H_{rr} = H_{cc} = -w^2 / s^4$ and
$H_{rc} = 0$, so

$$
\sigma^4 \det H = \frac{\sigma^4 w^4}{(w^2 + \sigma^2)^4},
$$

which is largest at $\sigma = w$, with the value $1/16$ for every $w$. The
scale-normalised Laplacian of `blob_log` at the centre is
$-\sigma^2 (H_{rr} + H_{cc}) = 2 \sigma^2 w^2 / (w^2 + \sigma^2)^2$, which is
also largest at $\sigma = w$, with the value $1/2$. The peak over `sigma` of
the response at the blob centre, for blobs of several widths:

```{code-cell} ipython3
rows = []
n_blob = 201
rb, cb = np.indices((n_blob, n_blob)) - n_blob // 2
for width in (1.5, 2, 3, 4, 6, 8, 12):
    image = np.exp(-(rb**2 + cb**2) / (2 * width**2))
    grid = np.arange(0.5, 2.5 * width + 0.01, 0.05 * width)
    centre = (n_blob // 2, n_blob // 2)
    rows.append({
        "width": width,
        "box, current": max(hessian_matrix_det(image, s)[centre[0] - 1, centre[1] - 1]
                            for s in np.arange(1, 4 * width + 2, 0.25)),
        "box, fixed": max(fixed_box_plane(image, L)[centre]
                          for L in range(1, int(4 * width) + 4, 2)),
        "FD": max(fd_plane(image, s)[centre] for s in grid),
        "FD cascade": fd_planes(image, grid, cascade=True)[centre].max(),
        "GD": max(gd_plane(image, s)[centre] for s in grid),
        "hybrid": hybrid_planes(image, grid)[centre].max(),
        "hybrid cascade": hybrid_cascade_planes(image, grid)[centre].max(),
        "Laplacian": max(log_plane(image, s)[centre] for s in grid),
    })
peaks = pd.DataFrame(rows)
show_table(peaks, floatfmt=".4f")
spread = peaks.drop(columns="width").max() / peaks.drop(columns="width").min() - 1
show_table(spread.rename("largest / smallest - 1").reset_index(), floatfmt=".3f")
```

The box routes are in other units; only the spread within each column
matters for them.

GD gives $1/16$ at every width, to 0.1%. FD falls short at small widths: its
peak is 33% low at width 1.5 and 0.6% low at width 12, a spread of 49%. A
detector that compares responses across scales therefore under-rates small
blobs with FD, and the threshold removes them first (§7.3 counts them). The
hybrid route uses GD below `sigma` 4, which covers the widths where the FD
error is largest; its spread is 2.9%, from the FD part at width 6. The
hybrid cascade gives the same spread: its FD part starts at `sigma` 4, where
the cascade steps are wide. The box
routes vary by 16% over these widths, and the Laplacian by 4%, all of it at
width 1.5.

The cascade overshoots: its peak at width 1.5 is 0.112, 80% above $1/16$.
Here the grid of `sigma` has steps of $0.05\,w$, so the first cascade steps
smooth by `sigma` of about 0.3. A sampled Gaussian kernel that narrow has a
variance well below $\sigma^2$. The variances of the steps add, so the image
is smoothed less than the cascade assumes. Sum the variances of the step
kernels that `ndi.gaussian_filter` uses, up to the last `sigma` of the grid
at or below $w$, and compare with the square of that `sigma`:

```{code-cell} ipython3
def kernel_variance(sigma):
    """Variance of the sampled kernel that `ndi.gaussian_filter1d` applies."""
    impulse = np.zeros(81)
    impulse[40] = 1.0
    x = np.arange(81) - 40
    return (ndi.gaussian_filter1d(impulse, sigma) * x**2).sum()


rows = []
for width in (1.5, 3, 12):
    grid = np.arange(0.5, 2.5 * width + 0.01, 0.05 * width)
    grid = grid[grid <= width + 1e-9]
    steps = np.sqrt(np.diff(np.concatenate([[0.0], grid**2])))
    rows.append({"grid": f"width {width}", "last sigma": grid[-1],
                 "smallest step": steps.min(),
                 "applied variance / last sigma^2":
                     sum(kernel_variance(s) for s in steps) / grid[-1] ** 2})
steps = np.sqrt(np.diff(np.concatenate([[0.0], SIGMAS**2])))
rows.append({"grid": "SIGMAS of §5", "last sigma": SIGMAS[-1],
             "smallest step": steps.min(),
             "applied variance / last sigma^2":
                 sum(kernel_variance(s) for s in steps) / SIGMAS[-1] ** 2})
show_table(pd.DataFrame(rows), floatfmt=".3g")
```

At width 1.5, with steps from 0.28, the cascade applies about half the
variance it should. At width 3, with steps from 0.42, it applies 3% too
little; at width 12 and with the steps of §5, 0.1% too little. A cascade
needs steps wide enough for the sampled kernel to keep its variance: here
0.28 is too narrow and 0.5 is wide enough. The cascade in §5, §6 and §7.3
uses steps of 1.8 and more.

### 7.2 No response to edges and lines

The determinant is the product of the two principal curvatures. Along a
straight edge or a long line one curvature is zero, so the determinant is
near zero. The Laplacian, the sum of the curvatures, does respond. This is the
reason to have a determinant detector beside `blob_log`
[@lindeberg1998feature_detection]. The largest response in the interior of
a step edge, a line 3 pixels wide, the same two on the diagonal, and a
constant image, over the largest response near the centre of a disk of
radius 10, at `sigma` 7:

```{code-cell} ipython3
n = 201
edge = np.zeros((n, n))
edge[:, n // 2:] = 1.0
line = np.zeros((n, n))
line[:, n // 2 - 1:n // 2 + 2] = 1.0
constant = np.ones((n, n))
r_d, c_d = np.indices((n, n)) - n // 2
diagonal_edge = (r_d > c_d).astype(float)
diagonal_line = (np.abs(r_d - c_d) <= 2).astype(float)  # about 3 pixels wide
disk_10 = ndi.gaussian_filter(((r_d**2 + c_d**2) <= 100).astype(float), 1)
inside = (slice(50, -50), slice(50, -50))
near_centre = (slice(n // 2 - 3, n // 2 + 4),) * 2
sigma = 7

rows = []
for name, func in (("FD determinant", lambda im: fd_plane(im, sigma)),
                   ("FD cascade", lambda im: fd_planes(im, np.linspace(2, sigma, 6),
                                                       cascade=True)[..., -1]),
                   ("GD determinant", lambda im: gd_plane(im, sigma)),
                   ("box, current", lambda im: hessian_matrix_det(im, sigma * np.sqrt(2))),
                   ("box, fixed", lambda im: fixed_box_plane(im, lobe_for(sigma * np.sqrt(2)))),
                   ("Laplacian", lambda im: log_plane(im, sigma))):
    # The largest response within 3 pixels of the disk centre; the current
    # box route puts it one pixel up and left.
    reference = np.abs(func(disk_10)[near_centre]).max()
    rows.append({"operator": name,
                 "edge / disk": np.abs(func(edge)[inside]).max() / reference,
                 "line / disk": np.abs(func(line)[inside]).max() / reference,
                 "diagonal edge / disk": np.abs(func(diagonal_edge)[inside]).max() / reference,
                 "diagonal line / disk": np.abs(func(diagonal_line)[inside]).max() / reference,
                 "constant / disk": np.abs(func(constant)[inside]).max() / reference})
show_table(pd.DataFrame(rows), floatfmt=".2g")
```

The box routes use `sigma` $\sqrt{2}\,\sigma$, or the lobe nearest to it,
so that their scale matches a disk of radius 10 (§5). The hybrid route is GD
below `sigma` 4 and FD above, so the GD and FD rows cover it.

The FD, cascade and GD determinants give zero, to rounding, for edges and
lines in every direction tested. The fixed box route gives zero along the
axes, and 8% of the disk response along the diagonal, because its filters
are square. The current box route responds along the axes too, 14% for an
edge, and 4% to a constant image. At `sigma` 9.9 its filter size is 29, not a
multiple of 3, so its filters do not sum to zero
(`on_hessian_det_approx.md` §4.3).
The Laplacian gives 23% to 34%. A determinant detector built on Gaussian
smoothing keeps the property that separates it from `blob_log`, and keeps it
better than the box filters.

### 7.3 Agreement with an exact detector

The tests so far use shapes chosen by hand. The gold standard of §2.4 tests
the whole detector on photographs. Fit `coins` and `camera` with sums of
Gaussians. Run the peak search and pruning of `blob_doh` on the exact
determinant at 11 `sigma` from 1.5 to 16, with threshold 0.005; these are the
gold blobs. Run each route on the rendered sum, at the same `sigma`. The box
routes use $\sqrt{2}$ times these `sigma`, or the odd lobes nearest to those,
for the radius convention of §5.

The determinant routes are in the units of the gold standard, so they use the
same threshold. The box routes and the Laplacian are in other units. For them,
`strongest` keeps as many blobs as the gold standard has, the strongest
first. "FD, same count" does the same for FD, to separate the effect of its
low response at small `sigma` from the effect of its position errors. The
current box route is the `blob_doh` cube; the cell checks that `blob_doh`
gives the same blobs from it.

```{code-cell} ipython3
GOLD_SIGMAS = np.geomspace(1.5, 16, 11)
GOLD_THRESHOLD = 0.005
BOX_SIGMAS = np.sqrt(2) * GOLD_SIGMAS
GOLD_LOBES = np.unique([lobe_for(s) for s in BOX_SIGMAS])

rows, fd_missed, canvases = [], [], {}
for name in ("coins", "camera"):
    levels, shape, image = fitted_canvas(name)
    gold = detect(exact_planes(levels, shape, GOLD_SIGMAS), GOLD_SIGMAS, GOLD_THRESHOLD)
    canvases[name] = (image, gold)
    n_gold = len(gold)
    box_cube = np.stack([hessian_matrix_det(image, s) for s in BOX_SIGMAS], axis=-1)
    # box_cube is the cube of blob_doh with log_scale=True: same blobs.
    via_blob_doh = blob_doh(image, BOX_SIGMAS[0], BOX_SIGMAS[-1], len(BOX_SIGMAS),
                            0.01, log_scale=True)
    via_cube = detect(box_cube, BOX_SIGMAS, 0.01)
    assert np.allclose(via_blob_doh[np.lexsort(via_blob_doh.T)],
                       via_cube[np.lexsort(via_cube.T)])
    fd_cube = fd_planes(image, GOLD_SIGMAS)
    found = {
        "GD": detect(np.stack([gd_plane(image, s) for s in GOLD_SIGMAS], axis=-1),
                     GOLD_SIGMAS, GOLD_THRESHOLD),
        "FD": detect(fd_cube, GOLD_SIGMAS, GOLD_THRESHOLD),
        "FD cascade": detect(fd_planes(image, GOLD_SIGMAS, cascade=True),
                             GOLD_SIGMAS, GOLD_THRESHOLD),
        "hybrid": detect(hybrid_planes(image, GOLD_SIGMAS), GOLD_SIGMAS, GOLD_THRESHOLD),
        "hybrid cascade": detect(hybrid_cascade_planes(image, GOLD_SIGMAS),
                                 GOLD_SIGMAS, GOLD_THRESHOLD),
        "FD, same count": strongest(fd_cube, GOLD_SIGMAS, n_gold),
        "box, current": strongest(box_cube, BOX_SIGMAS, n_gold),
        "box, fixed": strongest(
            np.stack([fixed_box_plane(image, L) for L in GOLD_LOBES], axis=-1),
            GOLD_LOBES, n_gold),
        "Laplacian, same count": strongest(
            np.stack([log_plane(image, s) for s in GOLD_SIGMAS], axis=-1),
            GOLD_SIGMAS, n_gold),
    }
    for route, blobs in found.items():
        g, f = match(gold, blobs)
        offset = np.hypot(*(gold[g, :2] - blobs[f, :2]).T)
        ratio = blobs[f, 2] / gold[g, 2]
        rows.append({"image": name, "route": route, "gold": n_gold,
                     "found": len(blobs), "matched / gold": len(g) / n_gold,
                     "matched / found": len(g) / len(blobs),
                     "exact centre": np.mean(offset == 0),
                     "scale / gold sigma, median": np.median(ratio),
                     "scale / gold sigma, IQR": np.subtract(*np.percentile(ratio, [75, 25]))})
        if route == "FD":
            fd_missed.append(gold[np.setdiff1d(np.arange(n_gold), g), 2])
gold_table = pd.DataFrame(rows)
show_table(gold_table, floatfmt=".2f")
fd_missed = np.concatenate(fd_missed)
print(f"FD misses {len(fd_missed)} gold blobs; {np.mean(fd_missed < 4):.0%} "
      f"of them have gold sigma below 4")
```

`exact centre` is the fraction of matched blobs at exactly the gold
position. The scale ratio is the reported scale over the gold `sigma`, its
median and its interquartile range (IQR) over the matched blobs; for the box
routes the reported scale is a radius or a lobe width, so only the IQR, the
spread, compares across routes.

GD finds every gold blob, at the gold position and scale, and nothing else.
This checks the test itself: the rendered sum, the margin and the matching
agree with the closed form. The hybrid route finds 96% and 99% of the gold
blobs, at the gold scale, and the hybrid cascade finds the same. FD, at the same threshold, finds only about two
thirds; 94% of the gold blobs it misses have `sigma` below 4. Given the same
count, FD matches 81% and 83%. The cascade finds almost the same blobs as
FD.

The box routes, given the same count, match 25% to 60% of the gold blobs.
Of the blobs they match, 11% to 28% have the exact centre. Their scale ratio
has an IQR of 0.38 to 0.56, about a quarter of its median, so no one factor
converts their scale to the gold `sigma`. The fixed route matches 60% on
`coins`, where the current one matches 36%; on `camera` they match 27% and
25%. The Laplacian matches 21% and 50%: it is a different detector, and it
finds other blobs.

### 7.4 The switch point of the hybrid cascade

The hybrid cascade uses GD below `switch` and cascade FD from `switch` up.
A higher switch moves more scales to GD, which is exact and costs more. Scan
the switch, on the scales and gold blobs of §7.3. The time is for the whole
detector on the photographs, over `blob_doh` with the same scales; a switch
of 0 is the FD cascade, and 20, above every scale, is GD alone:

```{code-cell} ipython3
photos = {"coins": ski.util.img_as_float(ski.data.coins()), "camera": camera}
box_times = {name: best_time(lambda: blob_doh(photo, BOX_SIGMAS[0], BOX_SIGMAS[-1],
                                              len(BOX_SIGMAS), log_scale=True))
             for name, photo in photos.items()}
rows = []
for switch in (0, 2, 3, 4, 5, 8, 20):
    row = {"switch": str(switch),
           "scales by GD": f"{np.sum(GOLD_SIGMAS < switch)} of {len(GOLD_SIGMAS)}"}
    for name, (image, gold) in canvases.items():
        blobs = detect(hybrid_cascade_planes(image, GOLD_SIGMAS, switch),
                       GOLD_SIGMAS, GOLD_THRESHOLD)
        g, f = match(gold, blobs)
        row[f"matched / gold, {name}"] = len(g) / len(gold)
        row[f"matched / found, {name}"] = len(g) / len(blobs)
    for name, photo in photos.items():
        row[f"time / box, {name}"] = best_time(lambda: detect(
            hybrid_cascade_planes(photo, GOLD_SIGMAS, switch), GOLD_SIGMAS,
            GOLD_THRESHOLD), repeats=1) / box_times[name]
    rows.append(row)
switch_scan = pd.DataFrame(rows)
show_table(switch_scan, floatfmt=".2f")
```

The match rises steeply up to a switch of 3 or 4 and slowly after it: from
two thirds at switch 0 to 96% and 99% at 4, and to 99% on both images at 5.
GD alone gives 100%. The time rises with each scale moved to GD, from about
1.6 times `blob_doh` at switch 0, to about 2.4 at 4, 2.8 at 5, about 4 at
8, and 6.7 to 7.1 for GD alone. Each time here is from one run, so it is
less exact than the times of §6. Every switch keeps 94% or more of its blobs
on gold blobs. The best trade-off on this corpus is a switch of 4 or 5;
above 5 the time grows faster than the match.

These times are higher than those of §6 at the same switch, because these
11 scales are spaced by a constant factor, so more of them are small: at
switch 4, five of the 11 go to GD, against one of the 10 default scales of
§6.

## 8. Comparators

`blob_log` is the scikit-image detector with the same structure: one
Gaussian-smoothed filter per scale, then the same peak search and pruning
(§5, §6). The exact reference for the determinant is the closed form of
the Gaussian sum (§2.4, §7.3); GD agrees with it to rounding
(`on_hessian_matrix_det.md` §6, and §7.3 here). The box routes are the current code and the
fix of `on_hessian_det_approx.md`. SURF uses a pyramid of box filters
[@bay2008speededup_robust]; SIFT uses a pyramid of Gaussians that halves the
image at each octave [@lowe2004distinctive_image]. This notebook does not
implement a pyramid.

## 9. Ways forward

The measurements support a Gaussian route for `blob_doh`:

* **FD with the closed-form determinant and $\sigma^4$**, as `fd_plane`. It
  needs no new compiled code: `ndi.gaussian_filter` and `np.gradient`.
* **Cascade smoothing** cuts the cost of the planes (§6), at a small error
  that grows with the number of scales (§4.2). Its steps must not be narrow:
  with steps from 0.28 it applies only half the variance it should (§7.1).
* **GD below `sigma` 4** removes the FD error at small scales at little cost,
  because GD kernels are short there (§4.1, §7.1).
* **GD below the switch and cascade FD above it**, as
  `hybrid_cascade_planes`, gives the accuracy of the hybrid at about the
  cost of the cascade (§6, §7.3). §7.4 gives the cost and the accuracy of
  each switch point. Its first FD scale is smoothed directly, so the narrow
  steps of the cascade start only above the switch.
* A **pyramid**, which smooths and then halves the image at each octave, would
  make the cost of the large scales close to flat. It needs care with
  resampling and with coordinates, and is not tested here.

Two ways to offer it:

* a `method` argument, `'box'` or `'gaussian'`, so that the fast box route
  stays available;
* or a Gaussian default in Skimage2, with the box route kept for speed.

Either way, the box route needs the fixes of `on_hessian_det_approx.md`.

## 10. What each change costs

* **Time.** The FD detector costs a few times the current box detector, and
  the cascade less (§6). Both cost less than `blob_log` over the same scales.
* **Meaning of `sigma`.** In a Gaussian route `sigma` is a Gaussian scale,
  and the radius is about $\sqrt{2}\,\sigma$ (§5). The current docstring says
  radius ≈ `sigma`. Code that uses the third column of the output as a radius
  must change.
* **Threshold.** The response is $\sigma^4 \det H$ in the units of the image,
  not the box response. The peak responses of §7.1 give the conversion for
  disks: a threshold tuned for the box route does not carry over.
* **Results.** Every blob may change scale; centres move by the shift of
  `on_hessian_det_approx.md` §4.2.

## 11. What not to do

* **Do not use `hessian_matrix_det(..., approximate=False)` as it is.** It has
  no $\sigma^4$ factor, so it cannot compare scales, and it uses
  `np.linalg.det`, which dominates its cost (§4.1).
* **Do not assume the box route is needed for speed.** Its cost is flat, but
  FD at small `sigma` costs the same, and the whole FD detector is within a
  small factor (§6).
* **Do not call a Gaussian `blob_doh` "a slower `blob_log`".** The
  determinant does not respond to edges and lines; the Laplacian does (§7.2).
  On fitted photographs the two find mostly different blobs (§7.3).
* **Do not use FD alone at small `sigma`.** At the threshold of the exact
  detector it misses a third of the gold blobs, nearly all small (§7.3).

## 12. Summary

| route | largest centre offset (px) | radius error on 12 disks | spread of peak response, Gaussian blobs | edge and line response / disk | gold blobs matched, fitted photos | time, × current box |
| --- | --- | --- | --- | --- | --- | --- |
| box, current | 1.4 | +4% to +57% | 16% | 0.02 to 0.14; 0.04 for a constant | 25% to 36%, same count | 1 |
| box, fixed | 0 | +3% to +40% | 16% | 0 on axes, 0.08 diagonal | 27% to 60%, same count | 1.0 (NumPy, beside the current layout) |
| FD | 0 | −5% to +4% | 49% | 0 | 66% to 69% | 3.2 to 3.4 |
| FD cascade | 0 | −5% to +4% | 91% with steps from 0.28 | 0 | 66% to 69% | 2.1 |
| hybrid (GD below `sigma` 4) | 0 | −5% to +4% | 2.9% | 0 (GD and FD rows) | 96% to 99% | 3.2 to 3.4 |
| hybrid cascade (GD below `sigma` 4) | 0 | −5% to +4% | 2.9% | 0 (GD and FD rows) | 96% to 99% | 2.1 to 2.2 |
| `blob_log` | 0 | −5% to +4% | 4.2% | 0.23 to 0.34 | 21% to 50%, same count; another detector | 8.6 to 9.9 |

The response columns come from §7.1 and §7.2, the gold column from §7.3, the
times from §6. "Same count" means that the route keeps as many blobs as the
gold standard, because its response is in other units. GD, the reference,
matches 100% of the gold blobs.

A `blob_doh` built on Gaussian smoothing finds the disks' centres and radii
more closely than the box routes and keeps the determinant's rejection of
edges and lines. With GD below `sigma` 4, it finds 96% to 99% of the blobs
of an exact detector on fitted photographs, where the box routes find 25% to
60%. The best trade-off measured is the hybrid cascade: the accuracy of the
hybrid, at 2.1 to 2.2 times the current box detector with the default scales
of `blob_doh`, against 3.2 to 3.4 for the hybrid and 8 to 10 for
`blob_log`. A switch of 5 instead of 4 raises its match to 99% on both
photographs (§7.4). Its costs are a new meaning for `sigma` and for
`threshold`.

### Limits

* The disks are synthetic and blurred by `sigma` 1. The gold-standard test
  uses two photographs, `coins` and `camera`, replaced by their Gaussian-sum
  fits, at 11 `sigma` from 1.5 to 16 and one threshold. The matching
  tolerance, half the gold `sigma` and at least 2 pixels, is a choice; the
  notebook does not test others.
* The photographs themselves are used for time only, because the thresholds
  of the routes are in different units.
* All times come from one run on one machine, and absolute times change with
  the load on the machine. Compare times only within one table.
* A pyramid and a recursive Gaussian filter are not tested.
* The fixed box route is a NumPy transcription. It is timed only against a
  NumPy transcription of the current layout.
