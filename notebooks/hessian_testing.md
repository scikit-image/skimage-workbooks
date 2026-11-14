---
title: 'Testing the one-pass Hessian'
date: 2026-09-17
options:
  updated: 2026-11-14
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

:::{assisted-by}
claude-code:claude-opus-5
:::

`hessian_matrix` builds each element of the Hessian. Up to scikit-image 0.26 it
used **two** convolution passes. From 0.27, and in `skimage2`, it uses **one**
pass per element, with kernels whose discrete moments have been corrected.
`hessian_review.md` measures why, against a supersampled reference and an exact
spectral one, and `on_hessian.md` traces the fault to its cause.

This notebook asks the narrower question:

> **What should the test suite assert, such that every assertion is a sentence
> about a picture, with a tolerance taken from the mechanism rather than from
> taste?**

The answer is five tests. None of them needs a reference implementation or any
of the scale-space literature. One (the boundary rule) is bit-identical when it
passes; the others use floors argued in §7.5. A single interior Gaussian blob
is enough for the closed-form check — multi-Gaussian scenes belong in
`hessian_review.md`, where they validate references, not here.

§1 transcribes both schemes rather than importing either, so the tests do not
depend on which release is installed. §1.1 reports which scheme the installed
library implements.

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
# The subject under test, in both namespaces.
import skimage as ski
import _skimage2
from scipy import linalg as scipy_linalg
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from _skimage2.feature import hessian_matrix as hessian_matrix2
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

## 1. The two schemes

Both compute the same continuous quantity: the second derivatives of the image
after smoothing with a Gaussian of width σ. They differ in how many times each
axis is touched, and in what kernel does the touching.

Each is transcribed here. Neither is imported, so the name of a scheme means
one fixed arithmetic in this notebook, and not whatever the installed release
happens to do.

```{code-cell} ipython3
def two_pass(image, sigma, mode="nearest", cval=0):
    """scikit-image <= 0.26: two first-order passes at sigma/sqrt(2)."""
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

The one-pass scheme asks for the whole derivative in one go. Each axis is convolved
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
    """scikit-image >= 0.27: one pass per element, with the corrected kernels."""
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


METHODS = (("two-pass (<= 0.26)", two_pass), ("one-pass (>= 0.27)", one_pass))
```

The three comments in `taps` are the whole of the correction, and they are also
the first three tests: a filter that gets 1, $x$ and $x^2/2$ right gets every
quadratic right.

The one-pass function that ships carries one more thing, which no kernel can
supply: an exactly flat image returns exact zeros, by inspection rather than by
arithmetic. §4.1 to §4.3 are about why.

```{code-cell} ipython3
def one_pass_shortcut(image, sigma, mode="nearest", cval=0, **kwargs):
    """`one_pass`, plus the short circuit for a flat image that 0.27 ships."""
    image = np.asarray(image, dtype=float)
    flat = image.size and np.ptp(image) == 0
    if flat and (mode != "constant" or cval == image.flat[0]):
        n_elements = image.ndim * (image.ndim + 1) // 2
        return [np.zeros_like(image) for _ in range(n_elements)]
    return one_pass(image, sigma, mode=mode, cval=cval, **kwargs)


SCHEMES = METHODS + (("one-pass + short circuit", one_pass_shortcut),)
```

+++

### 1.1 Which scheme is installed

Both namespaces are measured against all three transcriptions. The nearest one
names the arithmetic the installed release uses.

```{code-cell} ipython3
PHOTO = ski.util.img_as_float(ski.data.camera())[::2, ::2]


def nearest_scheme(func, sigma):
    """Name the transcription an installed `hessian_matrix` agrees with."""
    try:                      # `use_gaussian_derivatives` went in 0.27
        got = func(PHOTO, sigma=sigma, mode="nearest",
                   use_gaussian_derivatives=True)
    except TypeError:
        got = func(PHOTO, sigma=sigma, mode="nearest")
    gaps = {name: max(np.abs(a - b).max() for a, b in zip(got, s(PHOTO, sigma)))
            for name, s in SCHEMES}
    best = min(gaps, key=gaps.get)
    return best, gaps[best]


INSTALLED = (("skimage.feature", hessian_matrix),
             ("_skimage2.feature", hessian_matrix2))

show_table(pd.DataFrame(
    [{"namespace": label, "sigma": str(sigma),
      "agrees with": nearest_scheme(func, sigma)[0],
      "gap": f"{nearest_scheme(func, sigma)[1]:.1e}"}
     for label, func in INSTALLED for sigma in (0.5, 2.0)]), index="namespace")
```

The gap is zero where a transcription is exact, so the table reads as an
identification and not as a tolerance.

§2 to §8 use the transcriptions alone, so the five tests say the same thing in
either environment. §4.1 to §4.3 are the exception: they call the installed
`meijering`, `frangi` and `sato`, with only the Hessian replaced, so their
numbers carry the installed filters' own arithmetic.

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
bit-identical. The two-pass column is wrong by a large fraction of the peak
response — about a fifth for `nearest` on this picture, and more for some of
the other modes — except `wrap`, which is the control below.

The `wrap` row is the control that makes the rest of the table mean something.
A periodic extension is the one rule under which applying the extension twice is
the same as applying it once, so the two-pass scheme passes there too. The test
is not one that any filter passes, and not one that the two-pass filter always
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
rim = gaps["two-pass (<= 0.26)"] > 0.01
within_two = np.zeros_like(rim)
within_two[:2] = within_two[-2:] = True
within_two[:, :2] = within_two[:, -2:] = True
show_table(pd.DataFrame([{
    "pixels above 1% of the peak": int(rim.sum()),
    "of those, within 2 px of an edge": int((rim & within_two).sum()),
    "worst pixel": str(tuple(int(i) for i in np.unravel_index(
        gaps["two-pass (<= 0.26)"].argmax(), rim.shape))),
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
dimensions, isotropic and anisotropic σ — and the two-pass column is eleven
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

The two-pass curve does not approach the answer until σ is about 1, which is
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
is the property the two-pass scheme was designed around and the only one where
the correction is the weaker of the two.

```{code-cell} ipython3
flat = np.full((64, 64), 0.4)
show_table(pd.DataFrame(
    [{"sigma": sigma,
      **{name: f"{max(np.abs(h).max() for h in scheme(flat, sigma)):.1e}"
         for name, scheme in SCHEMES}}
     for sigma in (0.4, 0.5, 1.0, 3.0)]), index="sigma")
```

The two-pass scheme returns exactly zero, by symmetry: it never uses an even
kernel, and an odd kernel's taps cancel in pairs however coarsely they are
sampled. The bare corrected kernel returns about $10^{-16}$, because its first
tap condition is imposed by subtraction rather than granted by symmetry. The
short circuit returns exact zeros again, by not running the kernel at all.

So the test is `atol=1e-12`, not `== 0`, and the tolerance is chosen from the
mechanism: one subtraction of numbers of order one, on an image of order one.
Asserting exact equality would assert the short circuit rather than the
documented behaviour, and would pass for a flat image only.

### 4.1 What the $10^{-16}$ costs the consuming filters

Two of the four ridge filters that consume `hessian_matrix` divide by a
quantity derived from the Hessian, and a divisor that is rounding noise turns
the noise into the output. `meijering` scales each scale by its own maximum;
`frangi` takes its contrast reference γ from `s.max()`. Both guard the divisor
against being *exactly* zero, and that guard holds only while the Hessian is
exactly zero.

The three schemes are measured through the installed filters. Patching the
name the filters import is what makes the comparison possible in one process,
and `**kwargs` in the adapter is what makes it work against either release.

```{code-cell} ipython3
from unittest import mock

from skimage.filters import meijering, frangi, sato

# `frangi`, `meijering` and `sato` import `hessian_matrix` inside the function
# body, from the implementation package, so that is the name to replace.
import _skimage2.feature.corner as corner_module


def as_hessian_matrix(rule):
    """Wrap a Hessian rule in the `hessian_matrix` signature, ready to patch.

    `**kwargs` absorbs the parameters that differ between releases: `order` and
    `use_gaussian_derivatives` up to 0.26, `truncate` from 0.27.
    """
    def replacement(image, sigma=1, mode="reflect", cval=0, **kwargs):
        return rule(np.asarray(image, float), sigma, mode=mode, cval=cval)
    return replacement


def filter_scores(image):
    """Each filter's peak score on `image`, under each of the three schemes."""
    row = {}
    for label, scheme in SCHEMES:
        with mock.patch.object(corner_module, "hessian_matrix",
                               as_hessian_matrix(scheme)):
            for name, func in (("meij", meijering), ("frangi", frangi),
                               ("sato", sato)):
                row[f"{name}, {label}"] = func(image, sigmas=[1, 3]).max()
    return row


CONSTANTS = (0.4, 1.0, 1 / 3, 0.7)

show_table(pd.DataFrame(
    [{"constant image": round(c, 4),
      **{k: v for k, v in filter_scores(np.full((48, 48), float(c))).items()
         if not k.startswith("sato")}}
     for c in CONSTANTS]).round(6), index="constant image")
```

Read the three pairs of columns as the three schemes. Under two-pass every
score is zero. Under the bare corrected kernel `meijering` returns 1.0 — its
largest score — for three of the four constants, and `frangi` returns 0.07 to
0.12. The short circuit returns the column to zero, which is what it is for.

`0.4` is the row to notice. Its rounding noise happens to leave `meijering`'s
selected eigenvalue non-positive, so the clip to zero catches it, and the bare
kernel looks sound. Nothing about `0.4` is special otherwise. A test that used
only that constant would report no fault.

+++

### 4.2 The short circuit moves the cliff rather than removing it

`np.ptp(image) == 0` is exact equality, so it fires on a flat image and on
nothing else. Move one pixel by one part in $10^{15}$ and the divisor is
rounding noise again, under every scheme.

```{code-cell} ipython3
def nudged(constant, delta=1e-15):
    """A flat image with one pixel moved off the constant."""
    image = np.full((48, 48), float(constant))
    image[0, 0] += delta
    return image


show_table(pd.DataFrame(
    [{"constant image": round(c, 4),
      **{k: v for k, v in filter_scores(nudged(c)).items()
         if not k.startswith("sato")}}
     for c in CONSTANTS]).round(6), index="constant image")
```

`meijering` returns 1.0 for every constant and every scheme, including
two-pass. So this fault is not a consequence of the kernel change: the two-pass
scheme amplifies the same noise, and §1.1 identifies that scheme as the one
0.26 installs. What the kernel change broke is the exactly flat image alone,
and the short circuit closed that.

`0.4` fails here too, which makes the nudged image the better test: it catches
all four constants and all three schemes, where the flat image catches three
constants and one scheme.

`sato` is clear in both tables. It takes a product of clipped eigenvalues and
divides by nothing, so noise in the Hessian stays noise in the output.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"image": label,
      **{k: f"{v:.2e}" for k, v in filter_scores(image).items()
         if k.startswith("sato")}}
     for label, image in (("flat 0.7", np.full((48, 48), 0.7)),
                          ("flat 0.7, one pixel +1e-15", nudged(0.7)))]),
    index="image")
```

+++

### 4.3 Why the repair went into the kernel's caller and not the kernel

Repairing the kernel's sum again, after the correction, converges at σ = 0.5
and stalls at about $10^{-17}$ from σ = 1 upward. The adjustment falls below
the last bit of the accumulated sum, and `correlate1d` sums in its own order in
any case.

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

An exact zero is therefore not available from the taps, and the short circuit
supplies it by inspection instead. That is the whole of the repair the release
carries, and §4.2 shows what it does not reach.

The remaining fault wants a floor in each consuming filter: compare the divisor
against `100 * eps * abs(image).max()` rather than against zero. Both filters
are self-normalising — `meijering` divides by its own per-scale maximum,
`frangi` takes γ from `s.max()` — so any input with a non-zero response
normalises to full scale, and rounding noise is the only place a principled
floor can sit.

```{code-cell} ipython3
def meijering_floored(image, sigmas, scheme, floor):
    """`meijering`'s normalisation, with the divisor floored or not."""
    ndim = image.ndim
    alpha = -1 / (ndim + 1)
    mtx = scipy_linalg.circulant([1, *[alpha] * (ndim - 1)])
    limit = 100 * np.finfo(float).eps * np.abs(image).max() if floor else 0.0
    out = np.zeros_like(image, dtype=float)
    for sigma in sigmas:
        eigvals = hessian_matrix_eigvals(scheme(-image, sigma, mode="reflect"))
        vals = np.tensordot(mtx, eigvals, 1)
        vals = np.take_along_axis(vals, abs(vals).argmax(0)[None], 0).squeeze(0)
        vals = np.maximum(vals, 0)
        peak = vals.max()
        vals = vals / peak if peak > limit else np.zeros_like(vals)
        out = np.maximum(out, vals)
    return out


RIDGE = ski.util.img_as_float(ski.data.camera())[100:160, 100:160]

show_table(pd.DataFrame(
    [{"image": label,
      **{name: f"{meijering_floored(image, [1, 3], one_pass_shortcut, floor).max():.4f}"
         for name, floor in (("divisor > 0", False), ("divisor > floor", True))}}
     for label, image in (("flat 0.7", np.full((48, 48), 0.7)),
                          ("flat 0.7, one pixel +1e-15", nudged(0.7)),
                          ("camera crop", RIDGE))]
    ), index="image")
```

The floor zeroes the nudged image and leaves the photograph alone. It is a fix
for a fault that every release has carried, so it is not a cost of the kernel
change and does not belong in the same pull request. `on_meijering.md` and
`on_frangi.md` are where it goes.

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

At σ = 0.5 the two-pass scheme is 92% wrong and the corrected one 0.3% wrong. At
σ ≥ 2 both are at the rounding floor and the comparison says nothing — the
two-pass scheme is even a little closer there, which is rounding, not accuracy.

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

The corrected curve lies on the closed form; the two-pass one is a flat line near
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
which is stable. Taking what is left out of the centre tap, as the one-pass
kernel does, is a third option.

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
            ("both, as the kernel ships", "centre")]
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
argument was there to express. `skimage2.feature.hessian_matrix` has no
`order`, and expresses it through the tensor instead.

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
limit = np.abs(two_pass(PHOTO, 2.0)[1]).max()
bare(axes[0], "$H_{rc}$ of the picture")
axes[0].imshow(two_pass(PHOTO, 2.0)[1], cmap="gray", vmin=-limit, vmax=limit)
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
along two *different* axes — and under the two-pass scheme it differs along the
whole border, because the two axes are extended in different frames.

### 6.1 What this means for `order='xy'`

With the contract restored, `order='xy'` is the reversed list and nothing more.
Reversing is not enough for the two-pass numbers, because those include the
border fault, and reproducing them needs the whole computation run in the
transposed frame.

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

For the corrected filter the two shims agree to rounding, so the reversed list
is correct. For the two-pass filter they differ by up to 14%.

That 14% is the whole of the question for a port. `skimage.feature.hessian_matrix`
answers with the one-pass scheme from 0.27, including for
`use_gaussian_derivatives=True`, so no supported call returns two-pass numbers
and the transposed shim has nothing left to reproduce. The migration note
names the reversed list.

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
the two-pass scheme gets right, so a border test parametrised over `wrap` alone
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


def outcome(test, scheme):
    """Run one test against one scheme, and report pass or FAIL."""
    try:
        test(scheme)
        return "pass"
    except AssertionError:
        return "FAIL"


show_table(pd.DataFrame(
    [{"test": test.__name__.removeprefix("test_").replace("_", " "),
      "asserts": test.__doc__.splitlines()[0],
      **{name: outcome(test, scheme) for name, scheme in SCHEMES}}
     for test in SUITE]), index="test")
```

Four of the five fail on the two-pass scheme. The exception is the constant, and
that is not an accident: the two-pass scheme exists because someone noticed that
a sampled second-derivative kernel responds to a flat image, and built a pair of
odd first-derivative passes that cannot. It solved the one problem it was aimed
at and created the other four.

Both one-pass schemes pass all five. The short circuit changes no outcome
here: the constant test has a floor of `1e-12`, which $10^{-16}$ already meets,
so the short circuit changes that test's margin and not its result. §4.2 is
where it matters and §4.3 is why it exists.

These five landed as
`test_hessian_matrix_applies_the_boundary_rule_once`,
`test_hessian_matrix_is_exact_on_a_quadratic`,
`test_hessian_matrix_has_no_response_to_a_constant`,
`test_hessian_matrix_matches_a_gaussian_blob` and
`test_hessian_matrix_transposes_with_the_image` in
`tests/skimage2/feature/test_corner.py`, calling public `hessian_matrix`. The
`skimage` suite carries the first four against
`use_gaussian_derivatives=True`.

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
to §4.3 measure the consuming filters on flat and near-flat images only; what
the floor does to them on real images is `on_frangi.md` and `on_meijering.md`'s
business, not this notebook's, and the one photograph in §4.3 is a check that
the floor does not fire, not a measurement of its effect. The five tests run
against the transcriptions, not against the installed `hessian_matrix`; §1.1
identifies the installed scheme but does not run the suite against it.
`float32` input is not tested here, nor is the behaviour at σ below 0.3, where
§5's aliasing residue grows and no kernel repair addresses it. The
`interior()` rule assumes `truncate=8`; a caller passing a larger `truncate`
needs a larger crop. The timings are single runs on one machine and are
indicative only.

```{code-cell} ipython3
print(f"scikit-image  {ski.__version__}")
print(f"_skimage2     {_skimage2.__version__}")
print(f"scipy         {__import__('scipy').__version__}")
for label, func in INSTALLED:
    print(f"{label:<13} implements {nearest_scheme(func, 1.0)[0]}")
```
