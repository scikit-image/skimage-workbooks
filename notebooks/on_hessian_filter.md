---
title: On the `hessian` filter
date: 2026-09-19
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

`skimage.filters.hessian` calls `frangi`, then sets every non-positive pixel to
1. The docstring calls the result a filtered image and cites the Hybrid Hessian
Filter of [Ng, Yap, Costen and Li
(2014)](https://doi.org/10.1007/978-3-319-16811-1_40).

Two questions follow. Does the returned array rank ridges above non-ridges?
Is it the cited algorithm? The answer to both is no.

```{code-cell} ipython3
import pathlib

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import sympy as sp

from nbhelper import show_table
```

```{code-cell} ipython3
# The subject under test, and its three neighbours in the same module.
import skimage as ski
from skimage.filters import frangi, hessian, meijering, sato
from skimage.morphology import remove_small_objects
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

# Every figure here shows pixels in greyscale. The only colours are furniture.
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#dedcd5"
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


def paper_eq16(score):
    """Eq. (16) of the paper: 0 where the score is positive, 1 elsewhere."""
    return (score <= 0).astype(float)


def skimage_clamp(score):
    """What `hessian` adds to `frangi`: the `otherwise 1` branch, alone."""
    out = score.copy()
    out[out <= 0] = 1
    return out


def jaccard(a, b):
    """The Jaccard similarity index of two boolean masks, the paper's JSI."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return (a & b).sum() / max((a | b).sum(), 1)


def standardise(a):
    return (a - a.mean()) / a.std()


def agreement(a, b):
    """Correlation of two images, after removing mean and scale."""
    return np.corrcoef(standardise(a).ravel(), standardise(b).ravel())[0, 1]
```

```{code-cell} ipython3
# `camera()` is uint8. The ridge filters cast rather than rescale, so 0-255
# reaches the filter intact. Section 4 measures what that implies.
CAMERA = ski.data.camera()
PHOTO = CAMERA[::2, ::2]                       # decimated, for speed
PHOTO_FLOAT = ski.util.img_as_float(PHOTO)     # the same picture in [0, 1]
SIGMAS = (1, 3, 5, 7, 9)                       # the `hessian` default


def dark_bar(shape=(128, 128), width=4, blur=1.0):
    """One dark vertical bar on a flat white field, in 0-255."""
    image = np.ones(shape)
    centre = shape[1] // 2
    image[:, centre - width // 2:centre + width // 2] = 0.0
    return ndi.gaussian_filter(image, blur) * 255


BAR = dark_bar()
FIELD = (slice(5, 20), slice(5, 20))           # a corner, with no structure
SPINE = (slice(20, 108), 63)                   # down the middle of the bar
```

## 1. The problem

`hessian` and `sato` sit in the same module. Both documents say they detect
continuous ridges. Run both on `camera()` with their defaults.

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

`sato` gives a ridge map. `hessian` gives a ridge map with a white crust over
the grass, the sky and the coat. The crust is the filter's largest value.

```{code-cell} ipython3
show_table(pd.DataFrame([
    {"quantity": "the filter's largest value",
     "sato": f"{their_sato.max():.4f}", "hessian": f"{their_hessian.max():.4f}"},
    {"quantity": "share of the 512x512 image at that value",
     "sato": f"{(their_sato == their_sato.max()).mean():.2%}",
     "hessian": f"{(their_hessian == 1).mean():.2%}"},
    {"quantity": "largest value below it",
     "sato": f"{their_sato[their_sato < their_sato.max()].max():.4f}",
     "hessian": f"{their_hessian[their_hessian < 1].max():.6f}"},
]), index="quantity")
```

17.88% of the 262144 pixels read exactly 1.0. The largest value below 1.0 is
0.999790. Section 3 shows that the pixels at 1.0 are the ones the filter
rejected, and that 0.999790 is a genuine ridge score.

These are the documented defaults, on the library's own sample image, in the
dtype it ships in.

+++

## 2. What the code computes

The body of `hessian` is three statements.

```python
filtered = frangi(image, sigmas=sigmas, alpha=alpha, beta=beta, gamma=gamma, ...)
filtered[filtered <= 0] = 1
return filtered
```

The second statement is not an invention. It is half of eq. (16) of the paper.

The paper builds a wrinkle detector in five steps.

1. **Directional gradient**, eq. (1). The forehead image $I$ gives one partial
   derivative. That derivative, not $I$, is the image the rest of the pipeline
   reads. The paper writes: "Let $\partial I/\partial y$ denoted as
   $\mathcal{I}$, the Hessian matrix $\mathcal{H}$ of $\mathcal{I}$ at scale
   $\sigma$ is defined as Eq. (2)."
2. **Hessian and eigenvalues** of $\mathcal{I}$ at each scale, eqs. (2) to (11).
3. **Vesselness**, eqs. (12) to (15):
   $\mathcal{R} = (\lambda_1/\lambda_2)^2$,
   $\mathcal{S} = \lambda_1^2 + \lambda_2^2$, the two-term exponential of
   eq. (14) with a zero branch on the sign of $\lambda_2$, and the maximum over
   $\sigma \in \{1, 3, 5, 7\}$ of eq. (15). $\beta_1 = 0.5$, $\beta_2 = 15$.
4. **Binarise**, eq. (16): $\mathcal{L} = 0$ where $\mathcal{L} > 0$, and $1$
   otherwise.
5. **Area threshold**: remove every 8-connected region below 250 pixels.

Steps 4 and 5 work as a pair. Eq. (16) scores nothing. It makes a binary
candidate mask. Step 5 then drops every component too small to be a wrinkle.
The paper's output is a mask.

`skimage.filters.hessian` implements step 3, on the wrong input, and the
`otherwise 1` branch of step 4.

```{code-cell} ipython3
show_table(pd.DataFrame([
    {"step in the paper": "1. directional gradient, eq. (1)",
     "in `skimage.filters.hessian`": "absent"},
    {"step in the paper": "2. Hessian and eigenvalues",
     "in `skimage.filters.hessian`": "present, but of the raw image"},
    {"step in the paper": "3. vesselness, eqs. (12)-(15)",
     "in `skimage.filters.hessian`": "present, through `frangi`"},
    {"step in the paper": "4. binarise, eq. (16)",
     "in `skimage.filters.hessian`": "half: the `1` branch only"},
    {"step in the paper": "5. area threshold, 250 px",
     "in `skimage.filters.hessian`": "absent"},
]), index="step in the paper")
```

The paper's constants came across intact. `beta=0.5` is its $\beta_1$.
`gamma=15` is its $\beta_2$. `sigmas=range(1, 10, 2)` is its
$\{1, 3, 5, 7\}$ with a 9 added.

+++

## 3. D1 — eq. (16) is implemented by half

Eq. (16) maps a score to $\{0, 1\}$. `skimage` writes the 1 and omits the 0.
The positive scores stay in the output, beside the 1s.

```{code-cell} ipython3
score = frangi(PHOTO, sigmas=SIGMAS, gamma=15, mode="reflect")
mask, clamped = paper_eq16(score), skimage_clamp(score)

show_table(pd.DataFrame([
    {"output": "the paper, eq. (16)",
     "distinct values": f"{len(np.unique(mask))}",
     "share equal to 1": f"{(mask == 1).mean():.2%}"},
    {"output": "`skimage.filters.hessian`",
     "distinct values": f"{len(np.unique(clamped)):,}",
     "share equal to 1": f"{(clamped == 1).mean():.2%}"},
]), index="output")
```

Both put 1 in the same pixels. They differ everywhere else.

```{code-cell} ipython3
print("the 1-pixels are the same set:", np.array_equal(mask == 1, clamped == 1))
print("elsewhere `skimage` keeps the score:",
      np.array_equal(clamped[mask == 0], score[mask == 0]))
```

The 1-pixels are the zero branch of eq. (14). Rebuild that branch from the
eigenvalues. A pixel enters it when $\lambda_2$ fails the sign test at every
scale.

```{code-cell} ipython3
from skimage.feature import hessian_matrix, hessian_matrix_eigvals


def sign_accepted(image, sigma):
    """Pixels that pass the sign test at one scale: lambda_2 > 0."""
    elements = hessian_matrix(image, sigma, mode="reflect",
                              use_gaussian_derivatives=True)
    eigvals = hessian_matrix_eigvals(elements)
    eigvals = np.take_along_axis(eigvals, np.abs(eigvals).argsort(0), 0)
    return eigvals[1] > 0


accepted = np.logical_or.reduce(
    [sign_accepted(PHOTO.astype(float), s) for s in SIGMAS])
print("rejected at every scale == the 1-pixels:",
      np.array_equal(~accepted, mask == 1))
```

One array now holds two meanings. A pixel at 1 means *the sign test rejected
this pixel*. Any other value means *this is how ridge-like the pixel is*. The
docstring promises "Filtered image (maximum of pixels across all scales)",
which describes neither meaning.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3))
bare(axes[0], "the picture")
axes[0].imshow(PHOTO, cmap="gray")
bare(axes[1], "eq. (16): white means rejected")
axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
bare(axes[2], "`hessian()`: white means either")
axes[2].imshow(clamped, cmap="gray", vmin=0, vmax=1)
fig.tight_layout()
```

Panel 2 holds one meaning. White marks the pixels eq. (16) rejects. Panel 3
holds both. The camera, the tripod and the man's outline are bright there
because they are ridges. The grass and the sky are bright because the filter
threw them away.

### 3.1 The sentinel is above every score that the filter can give

Call the 1 written by the clamp the **sentinel**. The two meanings stay
separable while the sentinel is above or below every score. It is above. That
is a property of eq. (14), not of this image.

Eq. (14) multiplies a blobness factor by a structuredness factor:

$$
V = \exp\left(-\frac{\mathcal{R}}{2\beta_1^2}\right)
    \left(1 - \exp\left(-\frac{\mathcal{S}}{2\beta_2^2}\right)\right)
$$

```{code-cell} ipython3
S, c, R, beta = sp.symbols("S c R beta", positive=True)
blobness = sp.exp(-R / (2 * beta**2))
structuredness = 1 - sp.exp(-S / (2 * c**2))
V = blobness * structuredness

print("blobness <= 1      :", sp.simplify(blobness <= 1))
print("structuredness < 1 :", sp.simplify(structuredness < 1))
print("1 - V at R = 0     :", sp.simplify(1 - V.subs(R, 0)))
print("sup of V over S    :", sp.limit(V.subs(R, 0), S, sp.oo))
```

The structuredness factor stays below 1 for every finite $\mathcal{S}$. The
blobness factor stays at or below 1. So $V < 1$ always. The supremum of $V$ is
1, and $V$ never reaches it. The clamp writes exactly 1.

The sentinel is therefore above every score the filter can produce, for every
image and every parameter value.

That holds in exact arithmetic. In `float64` the gap $1 - V$ can fall below
$2^{-53}$ and round to zero, which makes a genuine score equal to the sentinel
bit for bit. Section 11 measures a case. Either way no threshold separates the
two meanings, and every threshold that keeps the strongest scores keeps all of
the rejected pixels first.

```{code-cell} ipython3
bright = clamped > 0.99
show_table(pd.DataFrame([
    {"pixels above 0.99": "every one of them",
     "share of the 256x256 image": f"{bright.mean():.2%}"},
    {"pixels above 0.99": "those at the sentinel, meaning rejected",
     "share of the 256x256 image": f"{(bright & (mask == 1)).mean():.2%}"},
    {"pixels above 0.99": "those that are genuine ridge scores",
     "share of the 256x256 image": f"{(bright & (mask == 0)).mean():.2%}"},
]), index="pixels above 0.99")
```

### 3.2 How often, and how big

The proof covers every image. The corpus below tests 15 sample images from
`skimage.data`, decimated by 2, colour images converted to grey, each filtered
with the `hessian` defaults.

```{code-cell} ipython3
CORPUS = ["camera", "coins", "moon", "text", "page", "brick", "grass",
          "gravel", "cell", "human_mitosis", "microaneurysms", "retina",
          "coffee", "astronaut", "horse"]


def as_grey_ubyte(image):
    """One 2-D uint8 image, whatever `skimage.data` returned."""
    if image.ndim == 3:
        image = ski.color.rgb2gray(image)
    return ski.util.img_as_ubyte(image)


rows = []
for name in CORPUS:
    image = as_grey_ubyte(getattr(ski.data, name)())[::2, ::2]
    out = hessian(image, mode="reflect")
    rows.append({"image": name,
                 "sentinel share": (out == 1).mean(),
                 "best score": out[out < 1].max()})
corpus = pd.DataFrame(rows).set_index("image")

# `gap` carries the claim: the best genuine score is below the sentinel by it.
# Shown instead of the score itself, which rounds to 1.000000 for `horse`.
show_table(pd.DataFrame({
    "share at the sentinel": corpus["sentinel share"].map("{:.2%}".format),
    "gap below the sentinel": 1 - corpus["best score"],
}), index=True, floatfmt=".2e")
```

```{code-cell} ipython3
# Compared on the raw floats: formatting to 6 dp rounds `horse` up to 1.000000.
below = corpus["best score"] < 1.0
print(f"images where the sentinel is above every genuine score: "
      f"{below.sum()} of {len(corpus)}")
print(f"smallest gap to the sentinel: {(1 - corpus['best score']).min():.1e} "
      f"({(1 - corpus['best score']).idxmin()})")
print(f"share at the sentinel ranges {corpus['sentinel share'].min():.2%} "
      f"to {corpus['sentinel share'].max():.2%}")
```

The rank order is wrong on all 15 images. The gap shrinks to 3.6e-08.

+++

## 4. D2 — `gamma` is absolute, so the answer follows the dtype

`gamma` is Frangi's $c$, written $\beta_2$ in the paper. It sets the reference
level for the Hessian norm $\mathcal{S}$. It is the only quantity in the filter
with absolute units. The paper states both the value and its dependence on
range:

> "$\beta_2$ depends on the greyscale range of the ridge of interest and
> controls the sensitivity of the filter to the measure $\mathcal{S}$ and the
> default value is 15."

The ridge filters do not rescale their input. `ridges.py` casts with `astype`,
not `img_as_float`. A `uint8` image arrives as 0-255, and 15 is the right
constant for it. Section 1 measured that case.

`img_as_float` is the identity on input that is already float. Both of the
images below are ordinary things to pass.

```{code-cell} ipython3
def hessian_norm(image, sigma):
    """The Frobenius norm of the Hessian: eq. (13) before the square."""
    elements = hessian_matrix(image, sigma, mode="reflect",
                              use_gaussian_derivatives=True)
    doubled = [e**2 if k in (0, len(elements) - 1) else 2 * e**2
               for k, e in enumerate(elements)]
    return np.sqrt(sum(doubled))


show_table(pd.DataFrame(
    [{"the image the filter sees": label,
      "largest S": f"{hessian_norm(-image, 3.0).max():.4g}",
      "structuredness at gamma = 15":
          f"{1 - np.exp(-hessian_norm(-image, 3.0).max()**2 / (2 * 15.0**2)):.3e}"}
     for label, image in (("camera(), uint8 0-255", PHOTO.astype(float)),
                          ("img_as_float(camera()), [0, 1]", PHOTO_FLOAT))]),
    index="the image the filter sees")
```

The gate is open at the range the constant was chosen for. It is shut five
orders down at the other range. `hessian` gives a different answer for each
dtype, and the difference changes the rank order.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"filter": name,
      "same answer for uint8 and float?":
          str(np.allclose(f(PHOTO), f(PHOTO_FLOAT))),
      "correlation between the two answers":
          f"{np.corrcoef(f(PHOTO).ravel(), f(PHOTO_FLOAT).ravel())[0, 1]:+.3f}"}
     for name, f in (("frangi", frangi), ("meijering", meijering),
                     ("sato", sato), ("hessian", hessian))]),
    index="filter")
```

`frangi` and `meijering` give the same answer for both dtypes. `sato` rescales
with its input and keeps the rank order, so it correlates at 1. `hessian` is
the only one of the four that changes the rank order.

`frangi` avoids this with `gamma=None`, which takes the constant from the
image. `hessian` hard-codes the literal.

```{code-cell} ipython3
rows = []
for g in (15, 15 / 255, None):
    out = hessian(PHOTO_FLOAT, sigmas=SIGMAS, gamma=g, mode="reflect")
    rows.append({"gamma, on the float image": str(g),
                 "largest value below the sentinel": f"{out[out < 1].max():.3e}",
                 "share at the sentinel": f"{(out == 1).mean():.2%}"})
show_table(pd.DataFrame(rows), index="gamma, on the float image",
           floatfmt=".3e")
```

Dividing `gamma` by 255 returns the float image to the behaviour of the `uint8`
one. The share at the sentinel does not move at any `gamma`: section 3 showed
that the eigenvalue signs fix it. D1 and D2 are independent. D1 does not depend
on the dtype.

+++

## 5. D3 — the directional gradient is missing

The filter is *hybrid* because it joins a directional gradient to the Hessian.
Eq. (1) takes the gradient of $I$. Every equation from (2) onward is written in
$\mathcal{I} = \partial I/\partial y$, not in $I$. The caption of the paper's
Fig. 2(c) says a Gaussian filter derives that gradient.

`skimage.filters.hessian` passes the image straight to `frangi`.

```{code-cell} ipython3
gradient_y = ndi.gaussian_filter(PHOTO.astype(float), 1, order=(1, 0))

on_image = frangi(PHOTO, sigmas=SIGMAS, mode="reflect")
on_gradient = frangi(gradient_y, sigmas=SIGMAS, mode="reflect")
print(f"correlation of the two responses: "
      f"{np.corrcoef(on_image.ravel(), on_gradient.ravel())[0, 1]:+.3f}")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
limit = np.abs(gradient_y).max()
bare(axes[0], "d/dy of the picture")
axes[0].imshow(gradient_y, cmap="gray", vmin=-limit, vmax=limit)
bare(axes[1], "frangi on the image, what `hessian` does")
axes[1].imshow(on_image, cmap="gray", vmin=0, vmax=on_image.max())
bare(axes[2], "frangi on d/dy, what the paper does")
axes[2].imshow(on_gradient, cmap="gray", vmin=0, vmax=on_gradient.max())
fig.tight_layout()
```

The two responses correlate at 0.19. No choice of constant turns one into the
other, because `gamma` cannot change which image is differentiated.

The docstring says `hessian` "uses alternative method of smoothing". Nothing in
the code does that. The paper's discussion names the gradient step: "in HHF the
directional gradient has greatly smoothed the image and preserved the data of
interest". The docstring describes the step the port left out.

+++

## 6. D4 — the area threshold is missing

Eq. (16) is not the end of the paper's pipeline. The paper then removes every
8-connected region below 250 pixels, and calls the result the estimated
wrinkle. The threshold depends on the image: the paper states that it is "based
on the initial image resolution".

```{code-cell} ipython3
kept = remove_small_objects(mask.astype(bool), max_size=249, connectivity=2)
show_table(pd.DataFrame([
    {"stage": "the eq. (16) mask",
     "share of the 256x256 image": f"{mask.mean():.2%}"},
    {"stage": "after the 250 px area threshold",
     "share of the 256x256 image": f"{kept.mean():.2%}"},
]), index="stage")
```

A component filter needs components. A continuous score has none. The area
threshold is the reason eq. (16) returns a mask.

+++

## 7. Agreement with the paper's worked example

The paper's images are forehead crops from the Bosphorus face database. That
database is not redistributable. The ground truth came from three coders and
was never published. The headline result, a mean JSI of 75.67% over 100 images,
is therefore closed to replication outside the group.

The paper prints one worked example at every stage, as its Fig. 2, at 845x117
and 365 ppi. Those panels are enough to check the pipeline end to end. Give the
paper its own panel (b), and compare each stage against the panel it printed.

:::{attention} Provenance
The panels under `notebooks/hhf_fig2_fixtures/` are extracted from Fig. 2 of
Ng, Yap, Costen and Li, "Automatic Wrinkle Detection using Hybrid Hessian
Filter", ACCV 2014
([doi:10.1007/978-3-319-16811-1_40](https://doi.org/10.1007/978-3-319-16811-1_40)),
© Springer International Publishing. The authors' accepted manuscript carries
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).

They are reproduced below, unaltered apart from conversion to 8-bit greyscale,
for the non-commercial purpose of verifying the algorithm this notebook
assesses, with attribution as above. That use is permitted by the licence; the
licence forbids distributing modified versions of them.

They are not covered by this repository's own licence. See the `Files:` stanza
for `notebooks/hhf_fig2_fixtures/` in `LICENSE`, and that directory's
`README.md`. The underlying forehead photograph comes from the Bosphorus face
database and is not redistributed: only the five printed panels the comparison
needs.
:::

```{code-cell} ipython3
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


import imageio.v3 as iio

# 8-bit greyscale, read as float in 0-255, the range the paper works in.
FIXTURES = _fixtures()
panel = {k: iio.imread(FIXTURES / f"{name}.png").astype(float)
         for k, name in PANEL_FILES.items()}
grey = panel["b"]
```

Step 1 is eq. (1). It settles D3 on the paper's own data. Panel (c) is the
derivative down the rows. It has no relation to the derivative across them.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"candidate for panel (c)": label,
      "correlation with the printed panel": f"{agreement(candidate, panel['c']):+.3f}"}
     for label, candidate in (
         ("d/dy, down the rows", ndi.gaussian_filter(grey, 1, order=(1, 0))),
         ("d/dx, across the columns", ndi.gaussian_filter(grey, 1, order=(0, 1))),
         ("the image itself, undifferentiated", grey))]),
    index="candidate for panel (c)")
```

Steps 2 to 5 then run as the paper specifies them:
$\sigma \in \{1, 3, 5, 7\}$, $\beta_1 = 0.5$, $\beta_2 = 15$, eq. (16) as
written, and the 250 px area threshold.

```{code-cell} ipython3
gradient = ndi.gaussian_filter(grey, 1, order=(1, 0))                  # eq. (1)
ours_d = frangi(gradient, sigmas=(1, 3, 5, 7), beta=0.5, gamma=15,     # (2)-(15)
                black_ridges=True, mode="reflect")
ours_e = ours_d <= 0                                                   # eq. (16)
ours_f = remove_small_objects(ours_e, max_size=249, connectivity=2)    # 250 px

show_table(pd.DataFrame([
    {"stage": "(c) directional gradient",
     "agreement with the printed panel":
         f"correlation {agreement(gradient, panel['c']):+.3f}"},
    {"stage": "(d) vesselness",
     "agreement with the printed panel":
         f"correlation {agreement(ours_d, panel['d']):+.3f}"},
    {"stage": "(e) eq. (16) mask",
     "agreement with the printed panel":
         f"JSI {jaccard(ours_e, panel['e'] > 127):.3f}"},
    {"stage": "(f) after the area threshold",
     "agreement with the printed panel":
         f"JSI {jaccard(ours_f, panel['f'] > 127):.3f}"},
]), index="stage")
```

```{code-cell} ipython3
stages = [("(c) gradient", panel["c"], gradient),
          ("(d) vesselness", panel["d"], ours_d),
          ("(e) eq. (16) mask", panel["e"], ours_e.astype(float)),
          ("(f) area thresholded", panel["f"], ours_f.astype(float))]

fig, axes = plt.subplots(4, 2, figsize=(11, 5.6))
for (name, theirs, ours), (left, right) in zip(stages, axes):
    bare(left, f"paper, {name}")
    left.imshow(theirs, cmap="gray")
    bare(right, f"ours, {name}")
    right.imshow(ours, cmap="gray")
fig.text(0.5, -0.02,
         "Left column: panels from Fig. 2 of Ng, Yap, Costen and Li, "
         "'Automatic Wrinkle Detection using Hybrid Hessian Filter', ACCV 2014,\n"
         "doi:10.1007/978-3-319-16811-1_40, \u00a9 Springer International "
         "Publishing, accepted manuscript under CC BY-NC-ND 4.0. "
         "Right column: computed here.",
         ha="center", va="top", fontsize=7, color=MUTED)
fig.tight_layout()
```

Every stage recovers the wrinkle. The paper's panel (c) carries more skin
texture than ours, and its panel (e) holds more speckle. A likely reason is
that the paper computed its panels at the resolution of the original crop,
before the figure was reduced for printing, so its input held detail that the
printed panel (b) no longer carries. This is a reading, not a result: nothing
here measures the original resolution.

The ceiling for the last row is below 1. Panel (e) is a printed, compressed
reproduction, and connected components are fragile to that. Measure the ceiling
by running the last step on the paper's own panel (e), against its own panel
(f).

```{code-cell} ipython3
their_e = panel["e"] > 127
show_table(pd.DataFrame(
    [{"area threshold": f"{ms} px",
      "paper's (e) to paper's (f)":
          f"{jaccard(remove_small_objects(their_e, max_size=ms - 1, connectivity=2), panel['f'] > 127):.3f}",
      "ours (e) to paper's (f)":
          f"{jaccard(remove_small_objects(ours_e, max_size=ms - 1, connectivity=2), panel['f'] > 127):.3f}"}
     for ms in (100, 250, 400, 600)]), index="area threshold")
```

The paper's own panels reach JSI 0.557 through this step. Our end-to-end result
is 0.574. This reading of the pipeline reproduces the paper's published result
as closely as the paper's own printed intermediate does.

The paper's column peaks at its stated 250 px. That confirms both the number
and the 8-connectivity. `skimage` has neither.

Our column keeps rising past 250 px. The paper reports the same tendency: HHF
"increases the true positive rate, but it also generated false wrinkle".

On the paper's data, as on `camera`, the eq. (16) mask does not move with
`gamma`.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"gamma": str(g), "JSI against the paper's panel (e)":
      f"{jaccard(frangi(gradient, sigmas=(1, 3, 5, 7), beta=0.5, gamma=g, black_ridges=True, mode='reflect') <= 0, panel['e'] > 127):.4f}"}
     for g in (15, 15 / 255, 0.1, None)]), index="gamma")
```

Every `gamma` gives the same mask, because the sign of $\lambda_2$ decides it.
`gamma=15` does no harm inside the paper's pipeline, where eq. (16) drops the
magnitudes. It does harm in `skimage`, which returns them.

+++

## 8. Why the tests pass

The assertion that covers the photographic case is:

```python
a_black = crop(camera(), ((200, 212), (100, 312)))
assert_allclose(hessian(a_black, black_ridges=True, mode='reflect'),
                np.ones((100, 100)), atol=1 - 1e-7)
```

The tolerance is one minus a rounding constant. It admits any output whose
every pixel is at least 1e-7. Run it on that input.

```{code-cell} ipython3
from skimage.util import crop

a_black = crop(ski.data.camera(), ((200, 212), (100, 312)))
deviation = np.abs(hessian(a_black, black_ridges=True, mode="reflect") - 1.0)
show_table(pd.DataFrame([
    {"quantity": "tolerance `atol=1 - 1e-7`", "value": f"{1 - 1e-7:.7f}"},
    {"quantity": "worst deviation from 1", "value": f"{deviation.max():.7f}"},
    {"quantity": "margin by which the assertion passes",
     "value": f"{(1 - 1e-7) - deviation.max():.2e}"},
]), index="quantity", floatfmt=".7g")
```

The assertion passes by about one part in a million. The clamp puts most pixels
at exactly 1, so an output that is nearly all ones satisfies a test that asserts
all ones. The test measures D1 and reads it as correct. Any repair that
restores a ridge response makes this assertion fail.

A second assertion, `assert_equal(hessian(zeros), ones)`, runs on an image with
no structure. Every pixel is rejected, so every pixel is at the sentinel. It
asserts D1.

+++

## 9. Properties that ought to hold

A ridge filter answers more strongly on a ridge than on nothing at all. Test
that on one dark bar on a flat field. The image holds one ridge and nothing
else.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"filter": name,
      "on the empty field": f"{out[FIELD].mean():.4f}",
      "along the bar": f"{out[SPINE].mean():.4f}",
      "stronger on the bar?": str(bool(out[SPINE].mean() > out[FIELD].mean()))}
     for name, out in (
         ("frangi", frangi(BAR, sigmas=SIGMAS, mode="reflect")),
         ("sato", sato(BAR, sigmas=SIGMAS, mode="reflect")),
         ("meijering", meijering(BAR, sigmas=SIGMAS, mode="reflect")),
         ("hessian", hessian(BAR, sigmas=SIGMAS, mode="reflect")))]),
    index="filter", floatfmt=".4f")
```

Three filters answer zero on the empty field and strongly on the bar. `hessian`
answers 1.0000 on both. At full precision the empty field is exactly 1.0 and
the bar is 0.999999995. The empty field outranks a perfect ridge by 5e-09,
which is section 3.1 again.

+++

## 10. Ways forward, and what each costs

The defects are independent. The paper decides most of the choices.

`gamma` should default to `None`, as `frangi`'s does. 15 is the paper's
$\beta_2$ for 0-255 data and is right for a `uint8` image. Section 4 showed
that it ties the answer to the dtype. `gamma=None` takes the constant from the
image instead.

Eq. (16) must be completed or dropped. Leaving it half-written is what produces
the sentinel.

```{code-cell} ipython3
shipped = hessian(PHOTO, sigmas=SIGMAS, mode="reflect")
candidates = {
    "drop the clamp, keep gamma=15":
        frangi(PHOTO, sigmas=SIGMAS, gamma=15, mode="reflect"),
    "drop the clamp, gamma=None":
        frangi(PHOTO, sigmas=SIGMAS, mode="reflect"),
    "complete eq. (16)":
        paper_eq16(frangi(PHOTO, sigmas=SIGMAS, gamma=15, mode="reflect")),
}
show_table(pd.DataFrame(
    [{"candidate": label,
      "pixels that change on camera": f"{(out != shipped).mean():.2%}",
      "on the empty field": f[FIELD].mean(),
      "along the bar": f[SPINE].mean()}
     for (label, out), f in zip(candidates.items(), (
         frangi(BAR, sigmas=SIGMAS, gamma=15, mode="reflect"),
         frangi(BAR, sigmas=SIGMAS, mode="reflect"),
         paper_eq16(frangi(BAR, sigmas=SIGMAS, gamma=15, mode="reflect"))))]),
    index="candidate", floatfmt=".9f")
```

Dropping the clamp changes 18.38% of the pixels of `camera` and restores the
rank order. Completing eq. (16) gives the paper's mask. That mask marks the
empty field on this input, because the input is an image and not a gradient
field. It is right inside the paper's pipeline and wrong as a general ridge
filter.

The clamp is one vectorised assignment. Its cost is below the run-to-run
spread of the filter it follows, so speed does not choose between the
candidates.

```{code-cell} ipython3
import timeit


def spread(call, repeats=5):
    """Best and worst of `repeats` timings, in seconds."""
    times = [timeit.timeit(call, number=1) for _ in range(repeats)]
    return min(times), max(times)


with_clamp = spread(lambda: hessian(PHOTO, sigmas=SIGMAS, mode="reflect"))
without = spread(
    lambda: frangi(PHOTO, sigmas=SIGMAS, gamma=15, mode="reflect"))
show_table(pd.DataFrame([
    {"call": "hessian, with the clamp",
     "best of 5": f"{with_clamp[0]:.3f} s", "worst of 5": f"{with_clamp[1]:.3f} s"},
    {"call": "frangi, the same work without it",
     "best of 5": f"{without[0]:.3f} s", "worst of 5": f"{without[1]:.3f} s"},
]), index="call")
```

+++

## 11. What not to do

Do not rescale `gamma` and stop there. On a `uint8` image `gamma=15` is already
the paper's constant. Lowering it raises the scores toward 1, which moves them
into the sentinel instead of away from it.

```{code-cell} ipython3
rows = []
for g in (15, 1.0, 15 / 255):
    out = hessian(BAR, sigmas=SIGMAS, gamma=g, mode="reflect")
    bar = out[SPINE].mean()
    rows.append({"setting": f"gamma = {g:g}",
                 "the empty field": f"{out[FIELD].mean():.1f}, the sentinel",
                 "gap between the bar and the sentinel": 1 - bar})
show_table(pd.DataFrame(rows), index="setting", floatfmt=".2e")
```

At `gamma=15` a perfect ridge sits 5e-09 below the sentinel. At `gamma=1` the
gap reaches 0: the ridge and the empty field hold the same `float64` value, and
nothing downstream can tell them apart.

Do not delete the clamp and call `hessian` repaired. With the clamp gone and
`gamma` following `frangi`, `hessian` runs the same call with the same
arguments as `frangi`.

```{code-cell} ipython3
print("clamp dropped, output identical to frangi(gamma=15):",
      np.array_equal(candidates["drop the clamp, keep gamma=15"],
                     frangi(PHOTO, sigmas=SIGMAS, gamma=15, mode="reflect")))
```

A repair of that shape turns a documented filter into an alias, without saying
so. The step that makes HHF a distinct filter is the directional gradient of
section 5. A faithful port is:

```python
def hybrid_hessian(image, axis=0, sigmas=(1, 3, 5, 7), beta=0.5, gamma=15,
                   min_size=250):
    order = tuple(1 if i == axis else 0 for i in range(image.ndim))
    gradient = ndi.gaussian_filter(image.astype(float), 1, order=order)
    ridge = frangi(gradient, sigmas=sigmas, beta=beta, gamma=gamma)
    return remove_small_objects(ridge <= 0, max_size=min_size - 1,
                                connectivity=2)
```

It returns a boolean mask. It takes an orientation that the present signature
has no room for. Its `gamma` suits 0-255 input only. Its `min_size` follows the
image resolution and has no safe default. It is a new function, not a repair of
this one.

+++

## 12. Summary

`skimage.filters.hessian` is a partial port. Its constants come from the paper.
The pipeline around them does not.

| | defect | evidence | weight |
| --- | --- | --- | --- |
| D1 | eq. (16) is implemented by half: the `1` branch without the `0` | $V < 1$ by §3.1, and the clamp writes 1, so the sentinel is above every score; 15 of 15 corpus images | fatal |
| D2 | `gamma=15` is absolute, so the answer follows the input dtype | uint8 and float answers correlate at 0.73; the other three filters are unaffected | severe, and `frangi`'s `gamma=None` already fixes it |
| D3 | eq. (1), the directional gradient, is absent | every equation from (2) on is written in $\partial I/\partial y$; the responses correlate at 0.19; §7 reproduces Fig. 2 only with the gradient in place | it is a different filter |
| D4 | the 250 px area threshold is absent | §7 finds the paper's own JSI peak at 250 px | the mask is unusable without it |
| D5 | the test asserts the broken output | `atol=1 - 1e-7` passes by 1.28e-06, because D1 puts most pixels at 1 | why it survived |

`hessian` computes step 3 of a five-step pipeline on the wrong input, keeps
half of step 4, drops step 5, and returns the result under a docstring that
describes none of it.

### Limits

The corpus is 15 images from `skimage.data`, decimated by 2, at the default
`sigmas`. There is no 3-D case. There is no forehead imagery of the kind the
paper targets.

§3.1 proves that the sentinel is above every score. That proof covers eq. (14)
as the paper writes it and as `frangi` implements it. It does not cover other
vesselness formulations.

§7 confirms this reading of the pipeline. It does not confirm the paper's
headline accuracy. Panel (f) is the paper's output, not annotated truth, so
agreement with it shows that the implementation is faithful, not that the
algorithm is accurate. An independent evaluation that obtained the authors' own
code ([Osman et al. 2020](https://doi.org/10.3390/jimaging6040017)) reports a
mean JSI of 31.69% for HHF on FERET, against the 75.67% reported in the paper.

§7 also rests on one image, reproduced through print and JPEG at 845x117, where
the paper's $\sigma$ and its resolution-dependent 250 px threshold were set for
1600x1200 originals.

Eq. (14) sets the score to zero when $\lambda_2 < 0$. The surrounding text says
that $\lambda_2 < 0$ marks the data of interest. The paper contradicts itself on
that sign, and its figures are the only tiebreak. Nothing above depends on it:
D1 concerns eq. (16), whichever sign eq. (14) uses.

### Which build this measures

`hessian` delegates to `frangi`, so it inherits whatever that function does.
This build restores the `sigma**2` factor of issue #7711; see `on_frangi.md`.
That affects D2 only, and in the direction that makes a release worse:
$\sigma^2 \ge 1$ at every default scale, so $\mathcal{S}$ here is at least as
large as in a release, and the gate at `gamma=15` is at least as open. D1, D3,
D4 and D5 do not depend on the build.

```{code-cell} ipython3
print(f"scikit-image {ski.__version__}")
print(f"from {ski.__file__}")
```
