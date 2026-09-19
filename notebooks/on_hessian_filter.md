---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# On the `hessian` filter

`skimage.filters.hessian` is documented as the "Hybrid Hessian filter" (HHF)
and cites [Ng, Yap, Costen and Li
(2014)](https://doi.org/10.1007/978-3-319-16811-1_40), a wrinkle detector. Its
whole body is three statements:

```python
filtered = frangi(image, sigmas=sigmas, alpha=alpha, beta=beta, gamma=gamma, ...)
filtered[filtered <= 0] = 1
return filtered
```

with `gamma=15` in the signature. That second line looks like a bug: it takes
every pixel the filter rejected and gives it the largest value in range — a
value that turns out to sit *above* the best score the filter can award a real
ridge.

It is not invented. The paper is in `library/ng2014hybrid_hessian.pdf`, and
that line is the surviving half of its eq. (16). The defect is subtler and
worse than an invented clamp: `hessian` implements the middle of a five-step
pipeline, keeps a fragment of the step that ends it, and drops the two steps
that make the fragment mean anything. What comes back is neither the paper's
output nor a vesselness.

This notebook reads the pipeline out of the paper, marks off what `skimage`
implements, and measures what the difference does. `on_frangi.md` covers the
filter this one delegates to; nothing here depends on that notebook's repairs.

```{code-cell} ipython3
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

from nbhelper import show_table
```

```{code-cell} ipython3
import skimage as ski
from skimage.filters import frangi, hessian, meijering, sato
from skimage.morphology import remove_small_objects  # max_size=N drops <=N
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette, validated all-pairs;
# same palette as `on_frangi.md` and `on_meijering.md`.
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


# `camera()` is uint8. The ridge filters cast rather than rescale (`astype`,
# not `img_as_float`), so 0-255 reaches the filter intact -- see section 4.
CAMERA = ski.data.camera()
PHOTO = CAMERA[::2, ::2]                       # decimated, for speed
PHOTO_FLOAT = ski.util.img_as_float(PHOTO)     # the same picture in [0, 1]
RIDGE_SIGMAS = (1, 3, 5, 7, 9)


def dark_ridge(shape=(128, 128), width=4, blur=1.0):
    """One dark vertical bar on a flat white field, in 0-255."""
    image = np.ones(shape)
    centre = shape[1] // 2
    image[:, centre - width // 2:centre + width // 2] = 0.0
    return ndi.gaussian_filter(image, blur) * 255


RIDGE = dark_ridge()
BACKGROUND = (slice(5, 20), slice(5, 20))      # a corner, genuinely flat
SPINE = (slice(20, 108), 63)                   # down the middle of the bar
```

## 1. The problem

`hessian` and `sato` are neighbours in `skimage.filters`, documented alike and
listed in each other's "See also". Run both on `camera()` with their defaults.

```{code-cell} ipython3
their_sato = sato(CAMERA)
their_hessian = hessian(CAMERA)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
bare(axes[0], "camera()")
axes[0].imshow(CAMERA, cmap="gray")
bare(axes[1], "sato(camera())")
axes[1].imshow(their_sato, cmap="gray", vmin=0,
               vmax=np.percentile(their_sato, 99.5))
bare(axes[2], "hessian(camera())")
axes[2].imshow(their_hessian, cmap="gray", vmin=0, vmax=1)
fig.tight_layout()
```

`sato` returns a ridge map. `hessian` returns the ridge map with a white crust
laid over the grass, the sky and the coat — and the crust is not noise. It is
the filter's *maximum* value, written into the pixels it was most confident
were not ridges.

```{code-cell} ipython3
show_table(pd.DataFrame([
    {"quantity": "the filter's maximum value",
     "sato": f"{their_sato.max():.4f}", "hessian": f"{their_hessian.max():.4f}"},
    {"quantity": "share of the image pinned at exactly that value",
     "sato": f"{(their_sato == their_sato.max()).mean():.2%}",
     "hessian": f"{(their_hessian == 1).mean():.2%}"},
    {"quantity": "largest value below it",
     "sato": f"{their_sato[their_sato < their_sato.max()].max():.4f}",
     "hessian": f"{their_hessian[their_hessian < 1].max():.6f}"},
]), index="quantity")
```

The last two rows are the whole notebook in one place. Nearly a fifth of the
image is pinned at the maximum, and the strongest genuine ridge response
anywhere in the picture is **0.999790** — just below it. The sentinel that means "not a
ridge" outranks the best evidence of a ridge the filter can produce, so there
is no threshold that separates them and no sense in which the array can be read
as a filtered image.

Nothing here is a tuning problem. These are the documented defaults, on the
library's own sample image, in the dtype it ships in.

+++

## 2. What the paper specifies

The paper builds a wrinkle detector in five steps. Numbered as it numbers them:

1. **Directional gradient** (eq. 1). The forehead image `I` is reduced to one
   partial derivative, and *that* becomes the image the rest of the pipeline
   sees: "Let ∂I/∂y denoted as 𝓘, the Hessian matrix 𝓗 of 𝓘 at scale σ".
   ∂I/∂y "emphasizes the horizontal line", which is what forehead wrinkles are.
2. **Hessian and eigenvalues** of 𝓘 at each scale (eqs. 2 to 11).
3. **Frangi's vesselness** (eqs. 12 to 15): 𝓡 = (λ₁/λ₂)², 𝓢 = λ₁² + λ₂², the
   two-term exponential of eq. (14) with the zero branch on the sign of λ₂, and
   the maximum over σ ∈ {1, 3, 5, 7} of eq. (15). β₁ = 0.5, β₂ = 15.
4. **Binarise** (eq. 16): $\mathcal{L}(x,y) = 0$ where $\mathcal{L}(x,y) > 0$,
   and $1$ otherwise.
5. **Area threshold**: "each region of interest (8-connected pixels) is
   filtered by an area threshold where regions less than 250 pixels are
   removed", giving the estimated wrinkle mask.

Steps 4 and 5 are a pair. Eq. (16) does not score anything — it produces a
binary candidate mask, and the caption of Fig. 2(e) says which pixels it keeps:
"Image vectors less than zero was preserved as ridge-like pattern." Step 5 then
throws away every connected component too small to be a wrinkle. The paper's
output is a mask, and it is only meaningful after both.

`skimage.filters.hessian` implements step 3, on the wrong input, and the
`otherwise 1` half of step 4.

```{code-cell} ipython3
show_table(pd.DataFrame([
    {"step": "1. directional gradient, eq. (1)", "in `skimage.filters.hessian`": "absent"},
    {"step": "2. Hessian of that image", "in `skimage.filters.hessian`": "present, but of the raw image"},
    {"step": "3. vesselness, eqs. (12)-(15)", "in `skimage.filters.hessian`": "present, via `frangi`"},
    {"step": "4. binarise, eq. (16)", "in `skimage.filters.hessian`": "half — the `1` branch only"},
    {"step": "5. area threshold, 250 px", "in `skimage.filters.hessian`": "absent"},
]), index="step")
```

The paper's constants, by contrast, came across intact, which is what shows the
port was made from this paper and not from somewhere else: `beta=0.5` is its
β₁, `gamma=15` is its β₂, and `sigmas=range(1, 10, 2)` is its {1, 3, 5, 7} with
a 9 added.

+++

## 3. D1 — half of eq. (16)

Eq. (16) maps a vesselness to `{0, 1}`. `skimage` writes the `1` and omits the
`0`, so the positive vesselness values survive into the output alongside it.

```{code-cell} ipython3
def paper_eq16(vesselness):
    """The paper's eq. (16): 0 where the vesselness is positive, 1 elsewhere."""
    return (vesselness <= 0).astype(float)


def skimage_clamp(vesselness):
    """What `hessian` does: the `otherwise 1` branch, and nothing else."""
    out = vesselness.copy()
    out[out <= 0] = 1
    return out


vesselness = frangi(PHOTO, sigmas=RIDGE_SIGMAS, gamma=15, mode="reflect")
show_table(pd.DataFrame([
    {"output": "the paper, eq. (16)", "distinct values":
        f"{len(np.unique(paper_eq16(vesselness)))}", "share equal to 1":
        f"{(paper_eq16(vesselness) == 1).mean():.2%}"},
    {"output": "`skimage.filters.hessian`", "distinct values":
        f"{len(np.unique(skimage_clamp(vesselness))):,}", "share equal to 1":
        f"{(skimage_clamp(vesselness) == 1).mean():.2%}"},
]), index="output")
```

The two agree on exactly which pixels become 1 — `skimage` reproduces the
paper's mask faithfully — and disagree everywhere else, where `skimage` leaves
the vesselness in place.

```{code-cell} ipython3
mask, clamped = paper_eq16(vesselness), skimage_clamp(vesselness)
print("the 1-pixels are the same set:", np.array_equal(mask == 1, clamped == 1))
print("elsewhere `skimage` keeps the vesselness:",
      np.array_equal(clamped[mask == 0], vesselness[mask == 0]))
```

That makes the returned array two incompatible things at once. Where a pixel
reads 1 it means *the sign test rejected this pixel*; where it reads anything
else it means *this is how ridge-like the pixel is*. The docstring promises
"Filtered image (maximum of pixels across all scales)", which describes neither
half.

The two senses would still be separable if the sentinel sat clear of the
scores. §1 measured that it does not:

```{code-cell} ipython3
show_table(pd.DataFrame([
    {"in the returned array": "the sentinel, meaning *rejected*", "value": "1.0"},
    {"in the returned array": "the best genuine ridge score",
     "value": f"{clamped[clamped < 1].max():.6f}"},
    {"in the returned array": "the gap between them",
     "value": f"{1 - clamped[clamped < 1].max():.2e}"},
]), index="in the returned array")
```

The paper's eq. (16) has no such problem, because it maps the scores to 0 and
keeps only the sentinel — one scale, two values. `skimage` keeps both scales,
and they overlap.

Those 1-pixels are the polarity branch firing, not an underflow. Reconstructing
the branch from the eigenvalues directly — the pixels where the second
eigenvalue fails the sign test at *every* scale — reproduces the set exactly.

```{code-cell} ipython3
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def sign_accepted(image, sigma):
    """Pixels the polarity branch keeps at one scale: lambda_2 > 0."""
    elements = hessian_matrix(image, sigma, mode="reflect",
                              use_gaussian_derivatives=True)
    eigvals = hessian_matrix_eigvals(elements)
    eigvals = np.take_along_axis(eigvals, np.abs(eigvals).argsort(0), 0)
    return eigvals[1] > 0


accepted = np.logical_or.reduce(
    [sign_accepted(PHOTO.astype(float), s) for s in RIDGE_SIGMAS])
print("rejected at every scale == the mask:", np.array_equal(~accepted, mask == 1))
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
bare(axes[0], "the picture")
axes[0].imshow(PHOTO, cmap="gray")
bare(axes[1], "the paper's eq. (16) mask")
axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
bare(axes[2], "`hessian()`: the mask plus the scores")
axes[2].imshow(clamped, cmap="gray", vmin=0, vmax=1)
fig.suptitle("in the right panel, white means either \u201crejected\u201d or"
             " \u201cstrong ridge\u201d", y=1.04)
fig.tight_layout()
```

Panel 2 is what the paper computes at this point: a candidate mask on its way
to an area threshold, where white means *rejected* and nothing else. Panel 3 is
what `skimage` returns — the same white pixels, with the picture's ridge
response superimposed at the same brightness. The camera, the tripod and the
man's outline are bright there because they are ridges; the grass and sky are
bright because they were thrown away. Both readings render identically, and
the table above says why: the sentinel is 1.0 and the best score is 0.999795.

```{code-cell} ipython3
bright = clamped > 0.99
show_table(pd.DataFrame([
    {"pixels brighter than 0.99": "total", "share of the image": f"{bright.mean():.2%}"},
    {"pixels brighter than 0.99": "... that are the sentinel (rejected)",
     "share of the image": f"{(bright & (mask == 1)).mean():.2%}"},
    {"pixels brighter than 0.99": "... that are genuine ridge scores",
     "share of the image": f"{(bright & (mask == 0)).mean():.2%}"},
]), index="pixels brighter than 0.99")
```

A threshold at 0.99 — the natural way to ask this array for its strongest
ridges — returns a set that is 97% rejected pixels. Lowering the threshold
admits more genuine ridges, but it cannot exclude any of the rejected ones:
they sit at the very top of the range, so *every* threshold contains all
18.38% of them.

+++

## 4. D2 — `gamma=15` makes the answer depend on the input dtype

`gamma` is Frangi's `c`, the reference level for the Hessian norm `S`, and the
only quantity in the filter carrying absolute units — see `on_frangi.md` §1.
The paper is explicit about both the value and its dependence on range:

> "β₂ depends on the greyscale range of the ridge of interest and controls the
> sensitivity of the filter to the measure 𝓢 and the default value is 15."

So 15 is the paper's β₂, chosen for the 8-bit skin images it works on. And the
ridge filters do not rescale their input — `ridges.py` casts with `astype`, not
`img_as_float` — so a `uint8` image arrives as 0-255 and 15 is the right
constant for it. Everything in §1 was measured that way.

The problem is the same picture in floating point. `img_as_float` is a
conversion `skimage` invites users to apply anywhere, and it is the identity on
input that is already float, so both of these are ordinary things to pass:

```{code-cell} ipython3
from skimage.feature import hessian_matrix


def hessian_norm(image, sigma):
    """`S` of eq. (13): the Frobenius norm of the Hessian."""
    elements = hessian_matrix(image, sigma, mode="reflect",
                              use_gaussian_derivatives=True)
    doubled = [e**2 if k in (0, len(elements) - 1) else 2 * e**2
               for k, e in enumerate(elements)]
    return np.sqrt(sum(doubled))


show_table(pd.DataFrame(
    [{"the image the filter sees": label,
      "largest S": f"{hessian_norm(-image, 3.0).max():.4g}",
      "structuredness at gamma = 15":
          f"{1 - np.exp(-hessian_norm(-image, 3.0).max() ** 2 / (2 * 15.0**2)):.3e}"}
     for label, image in (("camera(), uint8 0-255", PHOTO.astype(float)),
                          ("img_as_float(camera()), [0, 1]", PHOTO_FLOAT))]),
    index="the image the filter sees")
```

The gate is open at the range the constant was chosen for, and shut five orders
down at the other. So `hessian` answers differently depending on the dtype it
is handed — and not merely by a scale factor, but by changing which pixels
outrank which:

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"filter": name,
      "same answer for uint8 and float?":
          str(np.allclose(f(PHOTO), f(PHOTO_FLOAT))),
      "correlation between the two answers":
          f"{np.corrcoef(f(PHOTO).ravel(), f(PHOTO_FLOAT).ravel())[0, 1]:+.3f}"}
     for name, f in (("frangi", frangi), ("meijering", meijering),
                     ("sato", sato), ("hessian", hessian))]),
    index="filter")   # on the decimated copy, for speed
```

`frangi` and `meijering` are identical either way. `sato` rescales with its
input but preserves the ranking, so it correlates at 1. `hessian` is the only
one of the four that returns a *different answer* for the same picture, and the
cure is visible in the sibling function: `frangi` defaults `gamma=None` and
derives the constant from the image, which is range-independent by
construction. `hessian` hard-codes the literal instead.

```{code-cell} ipython3
rows = []
for g in (15, 15 / 255, None):
    out = hessian(PHOTO_FLOAT, sigmas=RIDGE_SIGMAS, gamma=g, mode="reflect")
    rows.append({"gamma, on the float image": str(g),
                 "largest value that is not the sentinel": f"{out[out < 1].max():.3e}",
                 "share set to the sentinel": f"{(out == 1).mean():.2%}"})
show_table(pd.DataFrame(rows), index="gamma, on the float image", floatfmt=".3e")
```

Dividing `gamma` by 255 restores the float image to the behaviour of the
`uint8` one, which confirms the diagnosis. Note also that the sentinel's share
does not move at any `gamma`: §3 showed it is fixed by the eigenvalue signs.
D1 and D2 are independent, and D1 is the one that does not depend on dtype.

+++

## 5. D3 — the step the filter is named for is missing

The filter is *hybrid* because it combines a **directional gradient** with the
Hessian. Eq. (1) of the paper takes the gradient of `I`, and the sentence after
it fixes what the Hessian is then taken of:

> "∂I/∂x and ∂I/∂y are the directional gradient as shown in Fig. 2(c). ∂I/∂y
> emphasizes the horizontal line. Let ∂I/∂y denoted as 𝓘, the Hessian matrix 𝓗
> of 𝓘 at scale σ is defined as Eq. (2)."

Every later equation is written in 𝓘, not `I`. The Hessian is taken of a
first-derivative image. `skimage.filters.hessian` passes the image straight to
`frangi`.

This also explains the docstring's otherwise puzzling "uses alternative method
of smoothing", which corresponds to nothing in the code. It is the paper's own
account of why HHF beat plain Frangi in their experiment: "in HHF the
directional gradient has greatly smoothed the image and preserved the data of
interest". The docstring describes the step the implementation left out.

```{code-cell} ipython3
# `PHOTO` is uint8; cast first or the derivative's negative lobe is clipped.
gradient_y = ndi.gaussian_filter(PHOTO.astype(float), 1, order=(1, 0))  # eq. (1)

comparison = {
    "frangi on the image (what `hessian` does)":
        frangi(PHOTO, sigmas=RIDGE_SIGMAS, mode="reflect"),
    "frangi on d/dy (the paper's HHF input)":
        frangi(gradient_y, sigmas=RIDGE_SIGMAS, mode="reflect"),
}
show_table(pd.DataFrame(
    [{"input": name, "max": f"{out.max():.4f}", "mean": f"{out.mean():.4f}"}
     for name, out in comparison.items()]), index="input")
print("correlation between the two: "
      f"{np.corrcoef(*[o.ravel() for o in comparison.values()])[0, 1]:+.3f}")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
bare(axes[0], "d/dy of the picture")
limit = np.abs(gradient_y).max()
axes[0].imshow(gradient_y, cmap="gray", vmin=-limit, vmax=limit)
for ax, (name, out) in zip(axes[1:], comparison.items()):
    bare(ax, name.split("(")[0].strip())
    ax.imshow(out, cmap="gray", vmin=0, vmax=out.max())
fig.suptitle("the paper's filter runs on the left panel; `hessian` runs on the"
             " picture itself", y=1.04)
fig.tight_layout()
```

The two responses share a correlation of about 0.19 — they have little to do
with each other. Whatever `skimage.filters.hessian` is, it is not this paper's
filter with a different constant; it is a different filter.

+++

## 6. D4 — the area threshold is missing

Eq. (16) is not the end of the paper's pipeline. The next paragraph is:

> "Next, each region of interest (8-connected pixels) is filtered by an area
> threshold where regions less than 250 pixels are removed and the output is
> the estimated forehead wrinkle as shown in Fig. 2(f). Note that the area
> threshold is based on the initial image resolution."

Without it the mask is Fig. 2(e), which the paper shows as a field of speckle;
with it, Fig. 2(f), a clean wrinkle line. The step that turns one into the
other is a connected-component filter, and it is the reason a binary mask is
the right output of eq. (16): an area threshold needs components, which a
continuous vesselness does not have.

```{code-cell} ipython3
mask = paper_eq16(frangi(PHOTO, sigmas=RIDGE_SIGMAS, gamma=15, mode="reflect"))
kept = remove_small_objects(mask.astype(bool), max_size=249, connectivity=2)
show_table(pd.DataFrame([
    {"stage": "eq. (16) mask (Fig. 2e)", "share of the image set":
        f"{mask.mean():.2%}"},
    {"stage": "after the 250 px area threshold (Fig. 2f)",
     "share of the image set": f"{kept.mean():.2%}"},
]), index="stage")
```

The threshold is stated to be resolution-dependent — "based on the initial
image resolution" — so a port cannot hard-code 250 either; on the paper's
1600×1200 originals it means something different than on a 256² crop. The
number above is illustrative of the step, not a recommended default.

+++

## 7. Replicating the paper's own figure

Everything above reads the pipeline out of the paper. This section runs it.

The paper's test images are forehead crops from the Bosphorus face database
(its ref. [19]), which is not redistributable, and the three coders'
annotations that form its ground truth were never published — so the headline
result, a mean JSI of 75.67% over 100 images, cannot be reproduced by anyone
outside the group. But Fig. 2 prints one worked example at every stage, at
845x117 and 365 ppi, and those panels are embedded in the PDF as images. That
is enough to check the algorithm end to end: feed the paper its own panel (b),
and compare each stage against the panel it printed.

:::{attention} Provenance of the panels below
The five panels are extracted from Fig. 2 of Ng, Yap, Costen and Li,
"Automatic Wrinkle Detection using Hybrid Hessian Filter", ACCV 2014
([doi:10.1007/978-3-319-16811-1_40](https://doi.org/10.1007/978-3-319-16811-1_40)),
© Springer International Publishing. The authors' accepted manuscript is
distributed under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

They are reproduced here, unaltered apart from conversion to 8-bit greyscale,
for the non-commercial purpose of verifying the algorithm this notebook
assesses. They are **not** covered by this repository's own licence — see the
`Files:` stanza for `notebooks/hhf_fig2_fixtures/` in `LICENSE`, and
`notebooks/hhf_fig2_fixtures/README.md`. The underlying forehead photograph is
from the Bosphorus database and is not redistributed: only the printed figure
panels are, and only the five the comparison needs.
:::

The panels are committed under `notebooks/hhf_fig2_fixtures/` so that this
notebook runs anywhere. `make fixtures SET=hhf_fig2` regenerates them from the
paper, via `library/generate_fixtures.py` and poppler's `pdfimages` — a
developer step needing a local copy of the paper, which the build never does.

```{code-cell} ipython3
import imageio.v3 as iio

PANEL_FILES = {"b": "fig2b_greyscale", "c": "fig2c_gradient",
               "d": "fig2d_vesselness", "e": "fig2e_mask",
               "f": "fig2f_threshold"}


def _fixtures():
    """Locate hhf_fig2_fixtures, whatever the working directory is."""
    for base in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents):
        for candidate in (base, base / "notebooks"):
            if (candidate / "hhf_fig2_fixtures" / "fig2b_greyscale.png").is_file():
                return candidate / "hhf_fig2_fixtures"
    raise FileNotFoundError(
        "cannot find hhf_fig2_fixtures/; run this notebook from a checkout "
        "of the skimage-workbooks repository"
    )


# 8-bit greyscale, read as float in 0-255 -- the range the paper works in.
FIXTURES = _fixtures()
panel = {k: iio.imread(FIXTURES / f"{name}.png").astype(float)
         for k, name in PANEL_FILES.items()}
grey = panel["b"]                # (b), the greyscale forehead, in 0-255
```

Step 1 is eq. (1), and it settles D3 on the paper's own data: panel (c) is the
derivative down the rows, and is uncorrelated with the derivative across them.

```{code-cell} ipython3
def standardise(a):
    return (a - a.mean()) / a.std()


def agreement(a, b):
    return np.corrcoef(standardise(a).ravel(), standardise(b).ravel())[0, 1]


show_table(pd.DataFrame(
    [{"candidate for panel (c)": label, "correlation with the printed panel":
      f"{agreement(candidate, panel['c']):+.3f}"}
     for label, candidate in (
         ("d/dy, down the rows", ndi.gaussian_filter(grey, 1, order=(1, 0))),
         ("d/dx, across the columns", ndi.gaussian_filter(grey, 1, order=(0, 1))),
         ("the image itself, undifferentiated", grey))]),
    index="candidate for panel (c)")
```

The caption for panel (c) also says how: "Gaussian filter was used to derive
the directional gradient from greyscale image", so eq. (1) is a Gaussian
derivative rather than a finite difference. Steps 2 to 5 then run as specified:
σ ∈ {1, 3, 5, 7}, β₁ = 0.5, β₂ = 15, eq. (16) exactly as written, and the
250 px area threshold.

```{code-cell} ipython3
def jaccard(a, b):
    """The paper's JSI, eq. (17), between two boolean masks."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return (a & b).sum() / max((a | b).sum(), 1)


# Fig. 2(c): "Gaussian filter was used to derive the directional gradient".
gradient = ndi.gaussian_filter(grey, 1, order=(1, 0))                 # eq. (1)
ours_d = frangi(gradient, sigmas=(1, 3, 5, 7), beta=0.5, gamma=15,    # eqs. (2)-(15)
                black_ridges=True, mode="reflect")
ours_e = ours_d <= 0                                                  # eq. (16)
ours_f = remove_small_objects(ours_e, max_size=249, connectivity=2)   # 250 px

show_table(pd.DataFrame([
    {"stage": "(c) directional gradient", "agreement with the printed panel":
        f"correlation {agreement(gradient, panel['c']):+.3f}"},
    {"stage": "(d) vesselness", "agreement with the printed panel":
        f"correlation {agreement(ours_d, panel['d']):+.3f}"},
    {"stage": "(e) eq. (16) mask", "agreement with the printed panel":
        f"JSI {jaccard(ours_e, panel['e'] > 127):.3f}"},
    {"stage": "(f) after the area threshold", "agreement with the printed panel":
        f"JSI {jaccard(ours_f, panel['f'] > 127):.3f}"},
]), index="stage")
```

```{code-cell} ipython3
stages = [("(c) gradient", panel["c"], gradient),
          ("(d) vesselness", panel["d"], ours_d),
          ("(e) eq. (16) mask", panel["e"], ours_e.astype(float)),
          ("(f) area thresholded", panel["f"], ours_f.astype(float))]

fig, axes = plt.subplots(4, 2, figsize=(11, 5.4))
for (name, theirs, ours), (left, right) in zip(stages, axes):
    bare(left, f"paper, {name}")
    left.imshow(theirs, cmap="gray")
    bare(right, f"ours, {name}")
    right.imshow(ours, cmap="gray")
fig.suptitle("the paper's Fig. 2 (left) against the pipeline as this notebook"
             " reads it (right)", y=1.02)
fig.tight_layout()
```

The wrinkle is recovered at every stage. The differences are what a printed
figure explains: our panel (c) is smoother than theirs because theirs was
computed on the full-resolution crop before the figure was downsampled, so it
retains skin texture ours never sees, and our (e) accordingly carries less
speckle.

How good is JSI 0.57 on the last row? The honest ceiling is well short of 1:
panel (e) as printed is a JPEG-compressed, re-thresholded reproduction, and
connected components are fragile to that. The way to measure the ceiling is to
run the last step on *the paper's own* panel (e) and see how well that
reproduces *the paper's own* panel (f).

```{code-cell} ipython3
their_e = panel["e"] > 127
show_table(pd.DataFrame(
    [{"area threshold": f"{ms} px",
      "paper's (e) -> paper's (f)":
          f"{jaccard(remove_small_objects(their_e, max_size=ms - 1, connectivity=2), panel['f'] > 127):.3f}",
      "ours (e) -> paper's (f)":
          f"{jaccard(remove_small_objects(ours_e, max_size=ms - 1, connectivity=2), panel['f'] > 127):.3f}"}
     for ms in (100, 250, 400, 600)]), index="area threshold")
```

Two things fall out. The paper's own panels only reach JSI 0.557 through this
step — so that is the ceiling, and our end-to-end 0.574 matches it. Starting
from the greyscale photograph, this reading of the algorithm reproduces the
paper's published result about as closely as the paper's own printed
intermediate does. The reading is correct.

And the paper's ceiling peaks at its stated 250 px, independent confirmation of
both that number and the 8-connectivity — neither of which is in `skimage`.
Ours keeps improving past 250, which is the over-segmentation the paper itself
reports ("HHF ... increases the true positive rate, but it also generated false
wrinkle"), amplified by our smoother low-resolution input.

One more control, because it settles D2 on the paper's data rather than on
`camera`. The mask of eq. (16) is invariant to `gamma`:

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"gamma": str(g), "JSI against the paper's panel (e)":
      f"{jaccard(frangi(gradient, sigmas=(1, 3, 5, 7), beta=0.5, gamma=g, black_ridges=True, mode='reflect') <= 0, panel['e'] > 127):.4f}"}
     for g in (15, 15 / 255, 0.1, None)]), index="gamma")
```

Every value of `gamma` gives the same mask, because the mask is decided by the
sign of λ₂. That is why `gamma=15` does no harm *inside the paper's pipeline*,
where eq. (16) discards the magnitudes, and does great harm in `skimage`, which
returns them.

+++

## 8. Why nothing caught it

The filter has tests. They pass on the current behaviour and would pass on
almost any other.

The assertion that covers the photographic case is

```python
a_black = crop(camera(), ((200, 212), (100, 312)))
assert_allclose(hessian(a_black, black_ridges=True, mode='reflect'),
                np.ones((100, 100)), atol=1 - 1e-7)
```

Its tolerance is one minus a rounding constant, so it admits any output whose
every pixel is at least `1e-7`. Run on that exact input:

```{code-cell} ipython3
from skimage.util import crop, invert

a_black = crop(ski.data.camera(), ((200, 212), (100, 312)))
deviation = np.abs(hessian(a_black, black_ridges=True, mode="reflect") - 1.0)
show_table(pd.DataFrame([
    {"quantity": "tolerance `atol=1 - 1e-7`", "value": f"{1 - 1e-7:.7f}"},
    {"quantity": "worst deviation from 1 actually seen", "value": f"{deviation.max():.7f}"},
    {"quantity": "margin by which the assertion passes",
     "value": f"{(1 - 1e-7) - deviation.max():.2e}"},
]), index="quantity", floatfmt=".7g")
```

The assertion is satisfied, with a margin of about one part in a million,
because `gamma=15` pushes the un-clamped pixels close to zero and the clamp
pushes the rest to exactly 1 — an output that is *nearly* all-ones passes a
test asserting all-ones. The test therefore measures the two defects working
together and reads them as correct. Any repair that restores a real ridge
response will make this assertion fail, which is why it has to be rewritten
rather than re-toleranced.

The other assertion, `assert_equal(hessian(zeros), ones)`, runs on an image
with no structure at all, where every pixel is rejected and therefore clamped.
It asserts the inversion rather than catching it.

+++

## 9. Comparators

`hessian` sits in `skimage.filters` beside three other ridge filters, and is
documented like them. The smallest test of what that implies is one dark bar on
a flat white field: one ridge, and nothing else in the picture.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"filter": name,
      "on the empty background": f"{out[BACKGROUND].mean():.4f}",
      "along the bar": f"{out[SPINE].mean():.4f}",
      "stronger on the bar?": str(bool(out[SPINE].mean() > out[BACKGROUND].mean()))}
     for name, out in (
         ("frangi", frangi(RIDGE, sigmas=RIDGE_SIGMAS, mode="reflect")),
         ("sato", sato(RIDGE, sigmas=RIDGE_SIGMAS, mode="reflect")),
         ("meijering", meijering(RIDGE, sigmas=RIDGE_SIGMAS, mode="reflect")),
         ("hessian", hessian(RIDGE, sigmas=RIDGE_SIGMAS, mode="reflect")))]),
    index="filter", floatfmt=".4f")
```

Three of the four answer zero on the background and strongly on the bar.
`hessian` answers 1.0000 on both, which is the collision of §1 in its purest
form: at full precision the background is exactly 1.0 and the bar is
0.999999995, so the empty field outranks a perfect ridge by 5e-09. Stated as
the property that ought to hold — *a ridge filter answers more strongly on a
ridge than on nothing at all* — only `hessian` fails, and it fails by a margin
too small to see in the table above.

That failure is real, but it is a statement about `hessian` as a *ridge
filter*, which is the contract its docstring and its placement beside the other
three imply. It is not a defect in eq. (16): the paper's mask marks rejected
pixels on purpose, and the paper never asks it to rank ridges. The defect is
that `skimage` exposes the intermediate mask under a docstring promising a
filtered image, without the input transform or the area threshold that make it
a wrinkle detector.

The docstring's second reference, Kroon's MATLAB `FrangiFilter2D`, is the
source `frangi` itself was ported from, and `frangi` has no clamp; I have not
read that MATLAB source directly, so I take the clamp's provenance from eq.
(16), which it matches exactly, and claim nothing about Kroon's code.

+++

## 10. Ways forward

The defects are independent, and the paper decides most of the choices.

**`gamma` should default to `None`, as `frangi`'s does.** 15 is the paper's β₂
for 0-255 data and is right for a `uint8` image, but §4 showed it makes the
output depend on the input dtype, which no other filter in the module does.
`frangi` already solves this: `gamma=None` resolves to half the largest Hessian
norm, so it adapts to whatever range it is given. Hard-coding a literal is the
only thing that cannot be defended.

**Eq. (16) must be completed or dropped, not left in half.** Two coherent
outputs exist, and the present one is neither:

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"candidate": label,
      "on the empty background": f"{out[BACKGROUND].mean():.9f}",
      "along the bar": f"{out[SPINE].mean():.9f}"}
     for label, out in (
         ("as shipped", hessian(RIDGE, sigmas=RIDGE_SIGMAS, mode="reflect")),
         ("eq. (16) completed",
          paper_eq16(frangi(RIDGE, sigmas=RIDGE_SIGMAS, gamma=15,
                            mode="reflect"))),
         ("clamp dropped, gamma=None",
          frangi(RIDGE, sigmas=RIDGE_SIGMAS, mode="reflect")))]),
    index="candidate", floatfmt=".9f")
```

The first row is the sharpest statement of D1 in the notebook, and it needs no
adjustment to any parameter to produce: the bar scores 0.999999995, which is a
genuine vesselness with the filter working exactly as intended, and the empty
background scores 1.0 from the clamp. The sentinel is not merely on the same
scale as the scores — it sits *above the largest score the filter can produce*,
so the two can never be separated.

Completing eq. (16) gives the paper's mask, which marks the background on this
input because the input is an image rather than a gradient field — correct
inside the paper's pipeline, useless as a general ridge filter. Dropping the
clamp gives a vesselness that behaves.

**Which leaves the real question.** With eq. (16) dropped and `gamma` following
`frangi`, `hessian` *is* `frangi` — the same call with the same arguments, and
the row above shows it. The only thing that would make it a distinct filter is
the step the docstring already gestures at and the code never had: the
directional gradient. A faithful HHF is

```python
def hybrid_hessian(image, axis=0, sigmas=(1, 3, 5, 7), beta=0.5,
                   gamma=15, min_size=250):
    gradient = ndi.gaussian_filter(image.astype(float), 1, order=_order(axis))
    ridge = frangi(gradient, sigmas=sigmas, beta=beta, gamma=gamma)
    return remove_small_objects(ridge <= 0, max_size=min_size - 1, connectivity=2)
```

which returns a boolean mask, takes an orientation the present signature has no
room for, carries a `gamma` that is only meaningful for 0-255 input, and has a
resolution-dependent `min_size` with no safe default. That
is a new function, not a repair of this one. The choice is to write it, or to
deprecate `hessian` in favour of `frangi`.

+++

## 11. Summary

`skimage.filters.hessian` is a partial port. Its constants come from the paper
— β₁, β₂ and the scales, the last with an extra 9 — while the pipeline around
them does not.

| | defect | evidence | weight |
| --- | --- | --- | --- |
| D1 | eq. (16) is implemented by half: the `1` branch without the `0` | the 1-pixels match the paper's mask exactly, and the rest keeps the vesselness — one array carrying two opposite senses, whose sentinel (1.0) outranks the best genuine score (0.999790) | fatal |
| D2 | `gamma=15` is a hard-coded absolute constant, so the answer depends on the input dtype | `hessian(camera())` and `hessian(img_as_float(camera()))` correlate at only 0.73; the other three filters are unaffected | severe, and `frangi`'s `gamma=None` already fixes it |
| D3 | eq. (1), the directional gradient, is absent | every equation from (2) on is written in ∂I/∂y, not `I`; the responses correlate at 0.19, and §7 reproduces the paper's Fig. 2 only with the gradient in place | it is a different filter |
| D4 | the 250 px area threshold is absent | the paper's own figures are speckle before it and a wrinkle line after | the mask is unusable without it |
| D5 | the test asserts the broken output | `atol=1 - 1e-7` passes with a margin of 1e-06, because D1 and D2 together make the output nearly all-ones | why it survived |

The single sentence: **`hessian` computes step 3 of a five-step pipeline on the
wrong input, keeps half of step 4, drops step 5, and returns the result under a
docstring describing none of it.**

**Limits.** One photograph (`camera`, decimated to 256²), one synthetic bar, and
the default `sigmas`; no 3-D case, and no forehead imagery of the kind the
paper targets.

The algorithm is read from the 2014 paper directly
(`library/ng2014hybrid_hessian.pdf`), so D3 and D4 are quotations, and §6
reproduces its Fig. 2 end to end from the printed panel (b) — matching the
paper's own intermediate-to-final agreement (JSI 0.574 against a ceiling of
0.557). The reading of the pipeline is therefore confirmed, not assumed.

What §7 does **not** establish is the paper's headline claim. Its mean JSI of
75.67% is over 100 forehead crops from the Bosphorus face database, which is
not redistributable, against ground truth from three coders that was never
published; that result cannot be checked by anyone outside the group. Nor is
§6 a test of accuracy — panel (f) is the paper's *output*, not annotated truth,
so agreeing with it shows the implementation is faithful, not that the
algorithm is right. An independent evaluation that obtained the authors' own
code ([Osman et al. 2020](https://doi.org/10.3390/jimaging6040017)) reports a
mean JSI of 31.69% for HHF on FERET, against the 75.67% reported here — so the
headline number does not appear to be portable off its original dataset.

§7 also rests on one image, reproduced through print and JPEG at 845×117, where
the paper's σ and its resolution-dependent 250 px threshold were tuned for
1600×1200 originals.

One reading is worth stating against myself. Eq. (14) sets the vesselness to
zero when λ₂ < 0, while the surrounding text says "λ₂ < 0 highlights the data
of interest" — the paper contradicts itself on that sign, and the figures are
the only tiebreak. Nothing above depends on resolving it: D1 is about eq. (16)
being half-implemented whichever sign eq. (14) uses.

**Which build this measures.** The cell below names it: a working tree in which
`frangi` has had the missing `sigma**2` factor of issue #7711 restored (see
`on_frangi.md`). `hessian` delegates to `frangi`, so it inherits that. This
matters only for D2, and only in the direction that makes the released build
worse: `sigma**2 >= 1` for every default scale, so `S` in this build is at
least as large as on the released build at every pixel, and the structuredness
gate at `gamma = 15` is therefore at least as open here as it is in a release.
D1, D3, D4 and D5 do not depend on the build.

```{code-cell} ipython3
print(f"scikit-image {ski.__version__}")
print(f"from {ski.__file__}")
```
