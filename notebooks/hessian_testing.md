---
title: Testing the one-pass Hessian
date: 2026-09-17
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

`hessian_matrix` is to change. Today it builds each element of the Hessian with
**two** convolution passes; the replacement uses **one** pass per element, with
kernels whose discrete moments have been corrected. `hessian_review.md` measures
why, against a supersampled reference and an exact spectral one, and
`on_hessian.md` traces the fault to its cause.

This notebook asks the narrower question:

> **What should the test suite assert, such that every assertion is a sentence
> about a picture, with a tolerance taken from the mechanism rather than from
> taste?**

The answer is five tests. None of them needs a reference implementation or any
of the scale-space literature. One (the boundary rule) is bit-identical when it
passes; the others use floors argued in §7.5. A single interior Gaussian blob
is enough for the closed-form check — multi-Gaussian scenes belong in
`hessian_review.md`, where they validate references, not here.

```{code-cell} ipython3
import functools

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from itertools import combinations_with_replacement

from nbhelper import show_table
```

```{code-cell} ipython3
import skimage as ski
from skimage.feature import hessian_matrix
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette, validated all-pairs;
# same palette as `hessian_review.md` and `on_hessian.md`.
C_ONE, C_TWO, C_THREE = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#dedcd5"

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "figure.facecolor": "white"}
)


def bare(ax, title=None):
    """Strip an image axes down to the pixels."""
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def recede(ax, title=None):
    """Push a plot axes' furniture into the background."""
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(RULE)
    if title:
        ax.set_title(title)
    return ax
```

## 1. The two candidates

Both compute the same continuous quantity: the second derivatives of the image
after smoothing with a Gaussian of width σ. They differ in how many times each
axis is touched, and in what kernel does the touching.

```{code-cell} ipython3
def shipped(image, sigma, mode="nearest", cval=0):
    """The scheme in the library: two first-order passes at sigma/sqrt(2)."""
    image = np.asarray(image, dtype=float)
    ndim = image.ndim
    if np.isscalar(sigma):
        sigma = (sigma,) * ndim
    truncate = 8 if all(s > 1 for s in sigma) else 100
    kwargs = dict(sigma=tuple(s / np.sqrt(2) for s in sigma), mode=mode,
                  cval=cval, truncate=truncate)
    orders = [[0] * d + [1] + [0] * (ndim - d - 1) for d in range(ndim)]
    gradients = [ndi.gaussian_filter(image, order=orders[d], **kwargs)
                 for d in range(ndim)]
    return [ndi.gaussian_filter(gradients[a], order=orders[b], **kwargs)
            for a, b in combinations_with_replacement(range(ndim), 2)]
```

That transcription must be faithful for anything below to mean anything, so
check it against the library before going on.

```{code-cell} ipython3
PHOTO = ski.util.img_as_float(ski.data.camera())[::2, ::2]

show_table(pd.DataFrame(
    [{"sigma": str(sigma),
      "transcription vs skimage": f"{max(np.abs(a - b).max() for a, b in zip(
          hessian_matrix(PHOTO, sigma=sigma, mode='nearest',
                         use_gaussian_derivatives=True),
          shipped(PHOTO, sigma, mode='nearest'))):.1e}"}
     for sigma in (0.5, 1.0, 2.0, (2.0, 3.0))]), index="sigma")
```

Bit-identical at every setting, so `shipped` *is* the library for the rest of
this notebook.

The replacement asks for the whole derivative in one go. Each axis is convolved
exactly once — twice-differentiated, once-differentiated or merely smoothed —
so the boundary rule is applied once, and there is no first-or-second choice to
get wrong on the mixed element. That brings back the even second-derivative
kernel, which a plain sampled Gaussian gets wrong at small σ, so the kernel is
corrected first.

```{code-cell} ipython3
def taps(sigma, order, truncate=8):
    """1-D Gaussian derivative kernel, with its discrete moments restored."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()                                 # exact on f = 1
    if order == 0:
        return g
    if order == 1:
        k = g * (x / sigma**2)
        return k / (k * x).sum()                 # exact on f = x
    k = g * ((x**2 - sigma**2) / sigma**4)
    k = k - k.sum() * g                          # no response to a constant
    k[len(k) // 2] -= k.sum()                    # ... and again, exactly (§4.1)
    return k / ((k * x**2).sum() / 2)            # exact on f = x**2 / 2


def one_pass(image, sigma, mode="nearest", cval=0, truncate=8, kernel=taps):
    """One convolution pass per element, with the corrected kernels."""
    image = np.asarray(image, dtype=float)
    ndim = image.ndim
    if np.isscalar(sigma):
        sigma = (sigma,) * ndim
    kernels = {(axis, o): kernel(sigma[axis], o, truncate)
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


METHODS = (("shipped, two-pass", shipped), ("one-pass, corrected", one_pass))
```

The three comments in `taps` are the whole of the correction, and they are also
the first three tests: a filter that gets 1, $x$ and $x^2/2$ right gets every
quadratic right.

+++

## 2. Test 1 — the boundary rule must be applied once

`mode` is a promise about what lies outside the picture. `mode='nearest'` says
the outside is a copy of the edge pixel; `mode='reflect'` says it is a mirror.
The promise has an exact consequence that needs no theory at all:

> **If we actually build that outside and hand over the larger picture, the
> answer inside must not change.**

```{code-cell} ipython3
# numpy's spelling of each scipy boundary rule.
PAD_MODE = {"nearest": "edge", "reflect": "symmetric", "mirror": "reflect",
            "wrap": "wrap", "constant": "constant"}


def pad_once_gap(method, image, sigma, mode, pad):
    """Worst disagreement between the direct answer and the padded one."""
    direct = method(image, sigma, mode=mode)
    bigger = np.pad(image, pad, mode=PAD_MODE[mode])
    padded = [h[pad:-pad, pad:-pad] for h in method(bigger, sigma, mode=mode)]
    scale = max(np.abs(h).max() for h in padded)
    return max(np.abs(a - b).max() for a, b in zip(direct, padded)) / scale
```

```{code-cell} ipython3
SIGMA, PAD = 2.0, 40                   # pad clears the kernel reach, 8 * sigma

show_table(pd.DataFrame(
    [{"mode": mode,
      **{name: f"{pad_once_gap(f, PHOTO, SIGMA, mode, PAD):.2%}"
         for name, f in METHODS}}
     for mode in ("nearest", "reflect", "mirror", "constant", "wrap")]),
    index="mode")
```

The one-pass column is not "small": it is `0.00%` because the two arrays are
bit-identical. The shipped column is wrong by a large fraction of the peak
response — about a fifth for `nearest` on this picture, and more for some of
the other modes — except `wrap`, which is the control below.

The `wrap` row is the control that makes the rest of the table mean something.
A periodic extension is the one rule under which applying the extension twice is
the same as applying it once, so the shipped scheme passes there too. The test
is not one that any filter passes, and not one that the shipped filter always
fails: it separates exactly the cases it should.

```{code-cell} ipython3
bigger = np.pad(PHOTO, PAD, mode="edge")
gaps = {}
for name, method in METHODS:
    direct = method(PHOTO, SIGMA, mode="nearest")
    padded = [h[PAD:-PAD, PAD:-PAD] for h in method(bigger, SIGMA, mode="nearest")]
    peak = max(np.abs(h).max() for h in padded)
    # In units of the peak response, so the two panels share one scale.
    gaps[name] = np.maximum.reduce(
        [np.abs(a - b) for a, b in zip(direct, padded)]) / peak

# The fault is a thin rim, so a linear scale hides it: 410 of the 428 pixels
# above 1% of the peak lie within two pixels of an edge.
FLOOR = 1e-5
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
bare(axes[0], "the picture")
axes[0].imshow(PHOTO, cmap="gray")
for ax, (name, _) in zip(axes[1:], METHODS):
    bare(ax, f"{name}\nworst {gaps[name].max():.1%} of the peak response")
    image = ax.imshow(np.maximum(gaps[name], FLOOR), cmap="magma",
                      norm=LogNorm(vmin=FLOOR, vmax=max(g.max() for g in gaps.values())))
fig.colorbar(image, ax=axes[2], fraction=0.046)
fig.suptitle("what changes when the boundary rule is built rather than assumed "
             "(log scale)", y=1.04)
fig.tight_layout()
```

A rim, one or two pixels deep, all the way round, and worst half-way down the
left edge rather than at a corner. The one-pass panel is not faint: every pixel
in it is exactly zero, so the whole panel sits on the floor of the scale.

```{code-cell} ipython3
rim = gaps["shipped, two-pass"] > 0.01
within_two = np.zeros_like(rim)
within_two[:2] = within_two[-2:] = True
within_two[:, :2] = within_two[:, -2:] = True
show_table(pd.DataFrame([{
    "pixels above 1% of the peak": int(rim.sum()),
    "of those, within 2 px of an edge": int((rim & within_two).sum()),
    "worst pixel": str(tuple(int(i) for i in np.unravel_index(
        gaps["shipped, two-pass"].argmax(), rim.shape))),
}]))
```

## 3. Test 2 — the second derivative of a quadratic is a constant

The most-used property of a second derivative, and the one a caller is most
entitled to. Build an image that is exactly a quadratic,

$$
f(r, c) = a + b\,r + b'c + \tfrac{1}{2} d\,r^2 + e\,rc + \tfrac{1}{2} g\,c^2 ,
$$

and the Hessian is the constant matrix $\begin{pmatrix} d & e \\ e & g
\end{pmatrix}$, at every pixel and at every σ. Smoothing does not change it,
because smoothing a quadratic returns a quadratic with the same curvature.

```{code-cell} ipython3
QUAD = dict(d=0.7, e=-0.4, g=1.3)      # the curvatures the filter must report


def quadratic(size=161, ndim=2):
    """An image that is exactly a quadratic, and the Hessian it must give."""
    coords = [c - size // 2 for c in np.indices((size,) * ndim, dtype=float)]
    curvature = {(i, j): v for (i, j), v in zip(
        combinations_with_replacement(range(ndim), 2),
        [0.7, -0.4, 0.2, 1.3, 0.5, -0.9][:ndim * (ndim + 1) // 2])}
    image = 3.0 + sum(0.2 * (k + 1) * x for k, x in enumerate(coords))
    for (i, j), v in curvature.items():
        image = image + (v / 2 if i == j else v) * coords[i] * coords[j]
    return image, [curvature[k] for k in curvature]


def interior(shape, sigma, truncate=8):
    """The pixels no kernel tap reached over the border."""
    reach = int(truncate * np.max(sigma) + 0.5) + 1
    return (slice(reach, -reach),) * len(shape)
```

```{code-cell} ipython3
def quadratic_error(method, sigma, ndim=2, size=161):
    """Worst departure from the curvatures that are there, away from the border.

    Divided by the largest value in the picture: the filter is linear, so
    doubling the picture doubles the absolute error (§7.5).
    """
    image, truth = quadratic(size, ndim)
    inner = interior(image.shape, sigma)
    return max(np.abs(h[inner] - t).max()
               for h, t in zip(method(image, sigma), truth)) / np.abs(image).max()


show_table(pd.DataFrame(
    [{"sigma": str(sigma), "ndim": ndim,
      **{name: f"{quadratic_error(f, sigma, ndim, size):.1e}" for name, f in METHODS}}
     for sigma, ndim, size in ((0.4, 2, 121), (0.5, 2, 121), (1.0, 2, 121),
                               (2.0, 2, 121), ((1.0, 3.0), 2, 121),
                               (0.5, 3, 61), (1.0, 3, 61))]),
    index=False)
```

The one-pass column sits at the rounding floor everywhere — two and three
dimensions, isotropic and anisotropic σ — and the shipped column is eleven
orders above it at σ = 0.4. The figure below says what that means in the units a caller
cares about.

```{code-cell} ipython3
sweep = np.round(np.arange(0.3, 2.01, 0.1), 3)   # coarse, to keep this quick
image, truth = quadratic(121)
gains = {name: [method(image, s)[0][interior(image.shape, s)].mean() / truth[0]
                for s in sweep]
         for name, method in METHODS}

fig, ax = plt.subplots(figsize=(6.2, 3.1))
recede(ax, "curvature reported, as a fraction of the curvature that is there")
ax.axhline(1.0, color=RULE, lw=4, zorder=1, label="the right answer")
for (name, _), colour in zip(METHODS, (C_TWO, C_ONE)):
    ax.plot(sweep, gains[name], color=colour, lw=1.6, marker="o", ms=2.5,
            label=name, zorder=2)
ax.set_xlabel("σ"); ax.set_ylabel("reported / true")
ax.set_ylim(-0.05, 1.15)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
```

The shipped curve does not approach the answer until σ is about 1, which is
where its own docstring advises callers not to go below. The corrected curve is
on the line at every σ in the sweep.

### 3.1 This test cannot be the only one

`taps` normalises the kernel so that it is exact on $x^2/2$. This test asserts
that it is exact on $x^2/2$. The agreement in the table above is therefore not
independent evidence — it confirms that the correction was applied, not that the
corrected filter is a good Hessian. §5 supplies the check that does not share a
premise with the thing it checks.

+++

## 4. Test 3 — a constant image has no curvature

The quadratic test contains this one, but it is worth its own name, because it
is the property the shipped scheme was designed around and the only one where
the correction is the weaker of the two.

```{code-cell} ipython3
flat = np.full((64, 64), 0.4)
show_table(pd.DataFrame(
    [{"sigma": sigma,
      **{name: f"{max(np.abs(h).max() for h in method(flat, sigma)):.1e}"
         for name, method in METHODS}}
     for sigma in (0.4, 0.5, 1.0, 3.0)]), index="sigma")
```

The shipped scheme returns exactly zero, by symmetry: it never uses an even
kernel, and an odd kernel's taps cancel in pairs however coarsely they are
sampled. The corrected kernel returns about $10^{-16}$, because its first tap
condition is imposed by subtraction rather than granted by symmetry.

So the test is `atol=1e-12`, not `== 0`, and the tolerance is chosen from the
mechanism: one subtraction of numbers of order one, on an image of order one.
Asserting exact equality here would be asserting the old implementation's
symmetry, not the documented behaviour.

### 4.1 That $10^{-16}$ is not free

The exact zero was load-bearing. Two of the four ridge filters that consume
`hessian_matrix` divide by a quantity derived from the Hessian, and a divisor
that is rounding noise turns the noise into the output. `meijering` scales each
scale by its own maximum; `frangi` takes its contrast reference γ from
`s.max()`. Both guard the divisor against being *exactly* zero, and that guard
is sufficient only while the Hessian is exactly zero.

```{code-cell} ipython3
import importlib
from unittest import mock

from skimage.filters import meijering, frangi

# `frangi` and `meijering` import `hessian_matrix` inside the function body,
# from the implementation package, so that is the name to replace.
corner_module = importlib.import_module("_skimage2.feature.corner")


def as_hessian_matrix(rule):
    """Wrap a Hessian rule in the `hessian_matrix` signature, ready to patch."""
    def replacement(image, sigma=1, mode="reflect", cval=0, order="rc",
                    use_gaussian_derivatives=True):
        return rule(np.asarray(image, float), sigma, mode=mode, cval=cval)
    return replacement


rows = []
for constant in (0.4, 1.0, 1 / 3, 0.7):
    flat_image = np.full((48, 48), float(constant))
    row = {"constant image": round(constant, 4)}
    for label, method in (("shipped", None), ("one-pass", one_pass)):
        if method is None:
            row[f"meijering, {label}"] = meijering(flat_image, sigmas=[1, 3]).max()
            row[f"frangi, {label}"] = frangi(flat_image, sigmas=[1, 3]).max()
        else:
            with mock.patch.object(corner_module, "hessian_matrix",
                                   as_hessian_matrix(method)):
                row[f"meijering, {label}"] = meijering(flat_image,
                                                       sigmas=[1, 3]).max()
                row[f"frangi, {label}"] = frangi(flat_image, sigmas=[1, 3]).max()
    rows.append(row)
show_table(pd.DataFrame(rows).round(6), index="constant image")
```

A flat image comes out of `frangi` at 0.8647 — the largest score it can return
— and out of `meijering` at 1.0 everywhere. The `0.4` row is a coincidence
worth noticing: it is a constant whose rounding noise happens to leave
`meijering`'s selected eigenvalue non-positive, so the clip to zero catches it.
Nothing about `0.4` is special otherwise, and a test that used only that
constant would miss this.

The repair belongs in those two filters, not in the kernel, for a reason that
is measurable: the exact zero cannot be recovered. Repairing the kernel's sum
again after the correction converges at σ = 0.5 and stalls at about
$10^{-17}$ from σ = 1 upward, because the adjustment falls below the last bit
of the accumulated sum — and `correlate1d` sums in its own order in any case.

```{code-cell} ipython3
rows = []
for sigma in (0.5, 1.0, 2.0, 3.0):
    k = taps(sigma, 2).copy()
    centre = len(k) // 2
    repairs = 0
    while repairs < 6:
        residue = ndi.correlate1d(np.ones(64), k, mode="nearest")[32]
        if residue == 0.0:
            break
        k[centre] -= residue
        repairs += 1
    rows.append({"sigma": sigma, "extra repairs": repairs,
                 "response to a constant row":
                     f"{ndi.correlate1d(np.ones(64), k, mode='nearest')[32]:.1e}"})
show_table(pd.DataFrame(rows), index="sigma")
```

So each consuming filter compares its divisor against a floor scaled to the
image, `100 * eps * abs(image).max()`, rather than against zero. That is a
change to `meijering` and `frangi` caused by this one, and it belongs in the
same pull request — it is not scope creep, it is the cost.

+++

+++

## 5. Test 4 — a Gaussian blob, against its closed form

This is the independent check §3.1 asks for. A Gaussian blob smoothed by a
Gaussian is another Gaussian, whose second derivatives are known exactly, and
nothing in the kernel correction was aimed at getting them right.

```{code-cell} ipython3
BLOB_WIDTH, BLOB_SIZE = 6.0, 201


def blob_and_truth(sigma, width=BLOB_WIDTH, size=BLOB_SIZE):
    """A Gaussian blob, and the Hessian it has after smoothing at `sigma`."""
    r, c = [x - size // 2 for x in np.indices((size, size), dtype=float)]
    blob = np.exp(-(r**2 + c**2) / (2 * width**2))
    scale = np.sqrt(width**2 + sigma**2)                 # the smoothed width
    smoothed = (width**2 / scale**2) * np.exp(-(r**2 + c**2) / (2 * scale**2))
    return blob, [smoothed * (r**2 - scale**2) / scale**4,
                  smoothed * (r * c) / scale**4,
                  smoothed * (c**2 - scale**2) / scale**4]


def blob_error(method, sigma):
    """Worst departure from the closed form, relative to its own peak."""
    blob, truth = blob_and_truth(sigma)
    inner = interior(blob.shape, sigma)
    scale = max(np.abs(t).max() for t in truth)
    return max(np.abs(h[inner] - t[inner]).max()
               for h, t in zip(method(blob, sigma), truth)) / scale


show_table(pd.DataFrame(
    [{"sigma": sigma, **{name: f"{blob_error(f, sigma):.1e}" for name, f in METHODS}}
     for sigma in (0.5, 0.7, 1.0, 2.0, 4.0)]), index="sigma")
```

At σ = 0.5 the shipped scheme is 92% wrong and the corrected one 0.3% wrong. At
σ ≥ 2 both are at the rounding floor and the comparison says nothing — the
shipped scheme is even a little closer there, which is rounding, not accuracy.

That 0.3% is the honest limit of the whole change. Correcting the kernel fixes
the operator on polynomials, which is a statement about zero frequency; a
Gaussian narrow enough to alias on the pixel grid is not a polynomial, and no
repair of a sampled kernel recovers it. So the **suite** below starts at σ = 1
with a floor of `1e-5`; the σ = 0.5 row here is diagnostic, not a gate.

```{code-cell} ipython3
blob, truth = blob_and_truth(0.5)
mid = BLOB_SIZE // 2
window = slice(mid - 20, mid + 21)
offsets = np.arange(-20, 21)

fig, ax = plt.subplots(figsize=(6.2, 3.1))
recede(ax, "Hrr through the centre of a Gaussian blob, σ = 0.5")
ax.plot(offsets, truth[0][window, mid], color=RULE, lw=4, zorder=1,
        label="closed form")
for (name, method), colour in zip(METHODS, (C_TWO, C_ONE)):
    ax.plot(offsets, method(blob, 0.5)[0][window, mid], color=colour, lw=1.5,
            label=name, zorder=2)
ax.set_xlabel("rows from the centre"); ax.set_ylabel("$H_{rr}$")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
```

The corrected curve lies on the closed form; the shipped one is a flat line near
zero, which is the same gain collapse §3 measured, now on a structure nobody
would call a corner case.

### 5.1 What these two tests caught

The two tests above are not hypothetical. The kernel correction has to make the
second-derivative taps sum to zero, and the obvious way to do it — subtract
`k.sum()` spread over the Gaussian, as §1's `taps` does in its first correction
line — is unstable when σ is small. The centre tap is then a difference of two
numbers of order $\sigma^{-2}$, and what should survive is of order
$e^{-1/2\sigma^{2}}$: at σ = 0.1 the whole of it is lost.

[DIPlib](https://diplib.org)'s published recipe subtracts the *mean* instead,
which is stable. Taking what is left out of the centre tap, as the shipped
version does, is a third option.

```{code-cell} ipython3
def variant_taps(sigma, order, truncate=8, fix="centre"):
    """`taps`, with the sum(k) = 0 condition imposed three different ways."""
    if order != 2:
        return taps(sigma, order, truncate)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    k = g * ((x**2 - sigma**2) / sigma**4)
    if fix == "gaussian":          # subtract k.sum() shaped like the Gaussian
        k = k - k.sum() * g
    elif fix == "mean":            # DIPlib: subtract the mean over the support
        k = k - k.mean()
    else:                          # both: Gaussian-shaped, then exactly
        k = k - k.sum() * g
        k[len(k) // 2] -= k.sum()
    return k / ((k * x**2).sum() / 2)


def with_variant(fix):
    """A `one_pass` that builds its kernels with one of the three fixes."""
    return functools.partial(
        one_pass, kernel=functools.partial(variant_taps, fix=fix))


VARIANTS = [("subtract k.sum() * g", "gaussian"),
            ("subtract the mean", "mean"),
            ("both, as shipped", "centre")]
flat_small = np.full((64, 64), 0.4)

show_table(pd.DataFrame(
    [{"correction": label,
      "flat, σ = 0.1":
          f"{max(np.abs(h).max() for h in with_variant(fix)(flat_small, 0.1)):.0e}",
      **{f"blob, σ = {sigma}": f"{blob_error(with_variant(fix), sigma):.0e}"
         for sigma in (0.2, 0.4, 0.7)}}
     for label, fix in VARIANTS]), index="correction")
```

Read the first column: the Gaussian-shaped subtraction responds to a constant
image with **0.8** at σ = 0.1, which is a complete failure, and which only a
test that reaches a σ that small will see. Then read the rest: the mean
subtraction is stable everywhere but three to eight times worse on the blob
between σ = 0.2 and σ = 0.7, because it spreads a pedestal across a support
that the true kernel barely occupies.

Doing both — the Gaussian-shaped correction, then taking the residue out of the
centre tap — is stable at σ = 0.1 and as accurate as the best of them
everywhere else. That is the line `taps` carries, and the reason the constant
test is parametrised down to σ = 0.1 rather than stopping where the pictures
look sensible.

+++

## 6. Test 5 — transposing the picture transposes the answer

A Hessian is a tensor. Relabel the axes and its entries must follow the
relabelling, with no arithmetic anywhere. This is the contract the `order`
argument was there to express, and it is about to be expressed by the tensor
instead.

```{code-cell} ipython3
def transpose_gap(method, image, sigma):
    """How far H(image.T) is from the transpose-relabelled H(image)."""
    Hrr, Hrc, Hcc = method(image, sigma)
    Txx, Txy, Tyy = method(image.T, sigma)
    scale = max(np.abs(h).max() for h in (Hrr, Hrc, Hcc))
    return max(np.abs(Txx - Hcc.T).max(), np.abs(Txy - Hrc.T).max(),
               np.abs(Tyy - Hrr.T).max()) / scale


show_table(pd.DataFrame(
    [{"sigma": sigma,
      **{name: f"{transpose_gap(f, PHOTO, sigma):.2%}" for name, f in METHODS}}
     for sigma in (0.5, 1.0, 2.0)]), index="sigma")
```

```{code-cell} ipython3
# Only the mixed element can differ, so show only that one.
mixed_gaps = {}
for name, method in METHODS:
    mixed = method(PHOTO, 2.0)[1]
    turned_back = method(PHOTO.T, 2.0)[1].T
    mixed_gaps[name] = np.abs(mixed - turned_back) / np.abs(mixed).max()

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
limit = np.abs(shipped(PHOTO, 2.0)[1]).max()
bare(axes[0], "$H_{rc}$ of the picture")
axes[0].imshow(shipped(PHOTO, 2.0)[1], cmap="gray", vmin=-limit, vmax=limit)
for ax, (name, _) in zip(axes[1:], METHODS):
    bare(ax, f"{name}\nworst {mixed_gaps[name].max():.1%} of $|H_{{rc}}|$")
    image = ax.imshow(np.maximum(mixed_gaps[name], FLOOR), cmap="magma",
                      norm=LogNorm(vmin=FLOOR,
                                   vmax=max(g.max() for g in mixed_gaps.values())))
fig.colorbar(image, ax=axes[2], fraction=0.046)
fig.suptitle("$H_{rc}$ against $H_{rc}$ of the transposed picture (log scale)",
             y=1.04)
fig.tight_layout()
```

Only the mixed element can differ — it is the only one built by differentiating
along two *different* axes — and under the shipped scheme it differs along the
whole border, because the two axes are extended in different frames.

### 6.1 What this means for `order='xy'`

With the contract restored, `order='xy'` is the reversed list and nothing more.
A caller who needs the old numbers bit-for-bit needs the transposed shim
instead, because the old numbers include the border fault.

```{code-cell} ipython3
def as_xy(method, image, sigma, **kwargs):
    """Run the whole computation in the transposed frame, and turn it back."""
    if not np.isscalar(sigma):
        sigma = tuple(sigma)[::-1]     # a sequence sigma is one value per axis
    return [h.T for h in method(image.T, sigma, **kwargs)]


show_table(pd.DataFrame(
    [{"sigma": str(sigma),
      **{name: f"{max(np.abs(a - b).max() for a, b in zip(
             as_xy(f, PHOTO, sigma), f(PHOTO, sigma)[::-1]))
             / max(np.abs(h).max() for h in f(PHOTO, sigma)):.1e}"
         for name, f in METHODS}}
     for sigma in (0.5, 1.0, (2.0, 3.0))]), index="sigma")
```

For the corrected filter the two shims agree to rounding, so the cheap one is
correct. For the shipped filter they differ by up to 14%, which is why the
migration note has to name the transposed one.

+++

## 7. What the tests must avoid

Five ways to write these tests so that they pass without meaning anything.

### 7.1 Crop the margin on the polynomial tests

A quadratic is not constant outside the frame, so no boundary rule continues it
correctly. The border pixels of the quadratic test are *supposed* to be wrong,
and a test that includes them is testing `mode`, not curvature.

```{code-cell} ipython3
image, truth = quadratic()
show_table(pd.DataFrame(
    [{"pixels used": label,
      **{name: f"{max(np.abs(h[region] - t).max() for h, t in zip(method(image, 1.0), truth)):.1e}"
         for name, method in METHODS}}
     for label, region in (
         ("the whole image", (slice(None),) * 2),
         ("a 4-pixel margin", (slice(4, -4),) * 2),
         ("interior(), 8σ + 1", interior(image.shape, 1.0)))]),
    index="pixels used")
```

### 7.2 Spell the boundary rule the way each library spells it

The test compares a scipy `mode` against a numpy `pad` mode, and two of the five
names cross over: scipy's `reflect` is numpy's `symmetric`, and scipy's `mirror`
is numpy's `reflect`. Matching them by name builds a *different* outside, and
the test then fails on a filter that is right.

```{code-cell} ipython3
SAME_NAME = {"nearest": "edge", "reflect": "reflect", "mirror": "symmetric",
             "wrap": "wrap", "constant": "constant"}


def gap_with(pad_modes, mode, pad=20):
    """`pad_once_gap` for the one-pass filter, under a given name mapping."""
    direct = one_pass(PHOTO, 2.0, mode=mode)
    bigger = np.pad(PHOTO, pad, mode=pad_modes[mode])
    padded = [h[pad:-pad, pad:-pad] for h in one_pass(bigger, 2.0, mode=mode)]
    scale = max(np.abs(h).max() for h in padded)
    return max(np.abs(a - b).max() for a, b in zip(direct, padded)) / scale


show_table(pd.DataFrame(
    [{"scipy mode": mode,
      "numpy spelling": PAD_MODE[mode],
      "matched correctly": f"{gap_with(PAD_MODE, mode):.1e}",
      "matched by name": f"{gap_with(SAME_NAME, mode):.1e}"}
     for mode in ("nearest", "reflect", "mirror", "wrap", "constant")]),
    index="scipy mode")
```

Ten per cent of the peak, on the filter this notebook is arguing for, from a
two-word mistake in the test.

### 7.3 Pad past the kernel's reach, for the reflecting modes

For `nearest`, `wrap` and `constant` the extension is idempotent — extending an
already-extended picture gives the same infinite picture — so any pad will do.
Reflection is not: its period is set by the frame, so a reflected pad narrower
than the kernel shows the kernel a different outside.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"pad": pad, **{mode: f"{gap_with(PAD_MODE, mode, pad):.1e}"
                     for mode in ("nearest", "reflect", "mirror")}}
     for pad in (1, 4, 8, 17, 60)]), index="pad")
```

At σ = 2 the kernel reaches 16 pixels, and the two reflecting columns drop to
zero at the first pad that clears it. Choose the pad from the kernel, not from
habit.

### 7.4 Do not test only `wrap`

§2's control row is the trap seen from the other side. `wrap` is the one mode
the shipped scheme gets right, so a border test parametrised over `wrap` alone
passes on the unfixed filter.

### 7.5 Take the tolerance from the mechanism

Three different tolerances are right here, and each is argued rather than
chosen: exact equality for the border test, where the arrays really are
bit-identical; about $10^{-15}$ of the peak measured for the transpose
(correlating along axis 0 and along axis 1 add in different orders), gated in
the suite at `1e-12`; a small multiple of the picture's own magnitude for
constants and quadratics, because the correction imposes a cancellation instead
of inheriting a symmetry; and a floor no tighter than the aliasing residue for
the Gaussian at σ ≥ 1, because that residue is real and is not going away.

The middle one is the one that is easy to get wrong. A quadratic's values grow
as the square of the distance from its centre, so a wider frame carries larger
numbers and a larger absolute rounding error, for exactly the same filter.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"frame": f"{size}²",
      "largest value in it": f"{np.abs(quadratic(size)[0]).max():.0f}",
      "absolute error": f"{quadratic_error(one_pass, 0.4, 2, size) * np.abs(quadratic(size)[0]).max():.1e}",
      "scaled by that value": f"{quadratic_error(one_pass, 0.4, 2, size):.1e}"}
     for size in (41, 81, 161, 301)]), index="frame")
```

The absolute column moves by two orders across those frames; the scaled column
does not move at all. So the tolerance belongs on the scaled quantity, which is
what `quadratic_error` returns.

+++

## 8. The suite

```{code-cell} ipython3
# Each test takes the method under test, so the same five can be pointed at
# either candidate.


def test_boundary_rule_is_applied_once(method):
    """Building the promised outside must not change the answer inside."""
    for mode in ("nearest", "reflect", "mirror", "constant", "wrap"):
        assert pad_once_gap(method, PHOTO, 2.0, mode, 40) == 0.0


def test_exact_on_a_quadratic(method):
    """The second derivative of a quadratic is the constant that built it."""
    for sigma, ndim, size in ((0.4, 2, 121), (1.0, 2, 121), ((1.0, 3.0), 2, 121),
                              (0.5, 3, 61), (1.0, 3, 61)):
        assert quadratic_error(method, sigma, ndim, size) < 1e-14


def test_no_response_to_a_constant(method):
    """A flat image has no curvature at any scale."""
    flat = np.full((64, 64), 0.4)
    for sigma in (0.4, 1.0, 3.0):
        assert max(np.abs(h).max() for h in method(flat, sigma)) < 1e-12


def test_matches_a_gaussian_blob(method):
    """Against a closed form the kernel correction was not aimed at."""
    for sigma in (1.0, 2.0, 4.0):
        assert blob_error(method, sigma) < 1e-5


def test_transposing_the_image_transposes_the_answer(method):
    """A Hessian is a tensor; relabelling the axes is not arithmetic."""
    for sigma in (0.5, 1.0, 2.0):
        assert transpose_gap(method, PHOTO, sigma) < 1e-12
```

```{code-cell} ipython3
SUITE = [test_boundary_rule_is_applied_once, test_exact_on_a_quadratic,
         test_no_response_to_a_constant, test_matches_a_gaussian_blob,
         test_transposing_the_image_transposes_the_answer]


def outcome(test, method):
    """Run one test against one candidate, and report pass or FAIL."""
    try:
        test(method)
        return "pass"
    except AssertionError:
        return "FAIL"


show_table(pd.DataFrame(
    [{"test": test.__name__.removeprefix("test_").replace("_", " "),
      "asserts": test.__doc__.splitlines()[0],
      **{name: outcome(test, method) for name, method in METHODS}}
     for test in SUITE]), index="test")
```

Four of the five fail on the shipped scheme. The exception is the constant, and
that is not an accident: the shipped scheme exists because someone noticed that
a sampled second-derivative kernel responds to a flat image, and built a pair of
odd first-derivative passes that cannot. It solved the one problem it was aimed
at and created the other four.

+++

## 9. Summary

| test | the sentence it asserts | tolerance |
| --- | --- | --- |
| boundary rule applied once | building the promised outside changes nothing inside | exact (`== 0`) |
| exact on a quadratic | the curvature reported is the curvature that is there | `1e-14`, scaled |
| no response to a constant | a flat image has no curvature | `1e-12` |
| matches a Gaussian blob | agrees with a closed form the correction did not target | `1e-5`, σ ≥ 1 (σ < 1 diagnostic only) |
| transpose | relabelling the axes relabels the tensor | `1e-12` of the peak (~$10^{-15}$ measured) |

and the rules the pictures must obey: crop `8σ + 1` from the polynomial tests
(§7.1), spell each boundary rule the way numpy spells it (§7.2), pad past the
kernel reach for the reflecting modes (§7.3), and never parametrise the border
test over `wrap` alone (§7.4). Keep the closed-form check on a **single
interior blob**; multi-Gaussian scenes are for `hessian_review.md`.

```{code-cell} ipython3
import time

rows = []
for sigma in (0.5, 1.0, 2.0):
    timing = {}
    for name, method in METHODS:
        runs = []
        for _ in range(2):                      # best of two, not the mean
            started = time.perf_counter()
            method(PHOTO, sigma)
            runs.append(time.perf_counter() - started)
        timing[name] = min(runs)
    rows.append({"sigma": sigma,
                 **{n: f"{t * 1e3:.0f} ms" for n, t in timing.items()},
                 "ratio": f"{timing[METHODS[1][0]] / timing[METHODS[0][0]]:.2f}x"})
show_table(pd.DataFrame(rows), index="sigma")
```

Cost is not a test, but it is the one number that might argue against the
change, and it does not. At σ ≤ 1 the corrected filter is about an order of
magnitude cheaper, because it does not carry the `truncate = 100` guard that the
sampled even kernel was thought to require. At σ = 2 the guard is gone from both
and the gap closes, the corrected kernels being the wider ones. Timings move by
tens of per cent between runs on this machine, so read the ratio's order and not
its last digit.

**Limits.** One photograph (256², `camera` decimated), one synthetic quadratic
and one Gaussian blob, at σ from 0.1 to 4, in two and three dimensions. §4.1
measures two consuming filters on flat images only; what the guard does to them
on real images is `on_frangi.md` and `on_meijering.md`'s business, not this
notebook's. The
five tests above are pointed at the local candidates (`shipped` / `one_pass`);
when they land in the library suite they must call public `hessian_matrix`,
not a notebook transcription. `float32` input is not tested here, nor is the
behaviour at σ below 0.3, where §5's aliasing residue grows and no kernel
repair addresses it. The `interior()` rule assumes `truncate=8`; a caller
passing a larger `truncate` needs a larger crop. The timings are single runs on
one machine and are indicative only.

```{code-cell} ipython3
print(f"scikit-image {ski.__version__}")
print(f"scipy        {__import__('scipy').__version__}")
```
