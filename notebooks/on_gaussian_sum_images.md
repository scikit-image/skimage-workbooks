---
title: 'On Gaussian RBF image sums'
date: '2026-11-25'
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

:::{related}
https://arxiv.org/abs/1602.05610 | Mobahi's closed-form Weierstrass transform for Gaussian RBFs; identity in [§2](#sec-mobahi)
https://doi.org/10.1137/S0895479891222106 | Fausett & Fulton on Kronecker least squares; two-step fit in [§3](#sec-fit)
:::

:::{changelog}
2026-11-25 | cursor:composer | Expand §3 on basis separability (isotropic factoring + product grid) vs dense amplitudes
2026-11-25 | cursor:composer | Note that `test_hessian_schemes.py` asserts absolute Mobahi accuracy, not scheme comparisons
2026-11-25 | cursor:composer | Create workbook; Mobahi smoothing identity; two-step lstsq amplitude fit (Fausett & Fulton)
:::

A Hessian gold standard on photograph-like content needs a continuous image
whose Gaussian-smoothed second derivatives are known exactly. Mobahi
[@mobahi2016closed_form, §2.2] gives that for isotropic Gaussian radial-basis
sums: convolution with a normalised Gaussian is another sum of the same form.
This notebook writes that identity in his notation, fits the amplitudes with a
two-step separable least squares [@fausett1994large_least] to `camera`,
`coins` and greyscale `astronaut`, and evaluates the exact Hessian that
`hessian_testing.md` §5.2 and `test_hessian_schemes.py` assert against.

Coordinates are in array order: the first index runs down the picture.

## 1. Setup

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skimage as ski
from nbhelper import show_table

import gaussian_sum as gs
```

```{code-cell} ipython3
C_ONE, C_TWO = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd5"

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "figure.facecolor": "white"}
)


def bare(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)
    return ax
```

(sec-mobahi)=
## 2. Mobahi's Gaussian RBF identity

Mobahi writes an isotropic Gaussian radial-basis sum in $\mathbb{R}^n$

$$
f(\mathbf{x})
  = \sum_{j=1}^{m} a_j
    \exp\!\Bigl(-\frac{\|\mathbf{x}-\mathbf{x}_j\|^2}{2\delta_j^2}\Bigr)
$$

and convolves it with the normalised Gaussian density

$$
k_\sigma(\mathbf{x})
  = \bigl(\sqrt{2\pi}\,\sigma\bigr)^{-n}
    \exp\!\Bigl(-\frac{\|\mathbf{x}\|^2}{2\sigma^2}\Bigr).
$$

The Weierstrass transform stays in the same family
[@mobahi2016closed_form, §2.2]:

$$
[f \star k_\sigma](\mathbf{x})
  = \sum_{j=1}^{m} a_j
    \Bigl(\frac{\delta_j}{\sqrt{\delta_j^2+\sigma^2}}\Bigr)^{\!n}
    \exp\!\Bigl(-\frac{\|\mathbf{x}-\mathbf{x}_j\|^2}{2(\delta_j^2+\sigma^2)}\Bigr).
$$

Each centre $\mathbf{x}_j$ is unchanged. The width $\delta_j$ becomes
$\sqrt{\delta_j^2+\sigma^2}$. The amplitude $a_j$ is multiplied by
$(\delta_j / \sqrt{\delta_j^2+\sigma^2})^n$. For images $n=2$, so that factor
is $\delta_j^2 / (\delta_j^2+\sigma^2)$.

The Hessian of $[f \star k_\sigma]$ is the term-by-term second derivative of
that sum. No kernel is truncated and no interpolant is assumed.

A single term checks the helper against the formula on the nose:

```{code-cell} ipython3
delta, a, centre = 4.0, 1.7, (40.0, 55.0)
shape = (96, 96)
level = gs.Level(
    delta=delta,
    row_centres=np.array([centre[0]]),
    col_centres=np.array([centre[1]]),
    amplitudes=np.array([[a]]),
)
sigma = 1.5
got = gs.render([level], shape, sigma=sigma)

rr, cc = np.indices(shape, dtype=float)
r2 = (rr - centre[0]) ** 2 + (cc - centre[1]) ** 2
delta_s = np.sqrt(delta**2 + sigma**2)
truth = a * (delta / delta_s) ** 2 * np.exp(-r2 / (2 * delta_s**2))

print(f"single-term max |render − Mobahi| = {np.max(np.abs(got - truth)):.1e}")
```

(sec-fit)=
## 3. Fitting amplitudes on a grid of centres

The identity needs chosen centres $\mathbf{x}_j$, widths $\delta_j$ and
amplitudes $a_j$. Widths are shared on *levels*
$\delta \in \{16, 8, 4, 2, 1.5\}$ pixels. On each level the centres form a
regular grid of spacing $\delta$, reaching one spacing past each image edge.
Levels are fitted coarse to fine, each to the residual the coarser levels
leave.

### Why the basis is separable

An isotropic Gaussian factors along axes because the squared Euclidean
norm does:

$$
\exp\!\Bigl(-\frac{(r-r_j)^2+(c-c_k)^2}{2\delta^2}\Bigr)
  = \exp\!\Bigl(-\frac{(r-r_j)^2}{2\delta^2}\Bigr)\,
    \exp\!\Bigl(-\frac{(c-c_k)^2}{2\delta^2}\Bigr).
$$

The first factor depends only on the row coordinate and the row-centre index
$j$; the second only on the column coordinate and the column-centre index
$k$. Nothing in the isotropic kernel couples a row centre to a column centre
except through the amplitude that multiplies that pair.

That factoring alone is not enough for a cheap fit. If the centres
$\mathbf{x}_j$ were an unstructured cloud in the plane, each term would still
be a product of two 1-D Gaussians, but the set of products would not be a
*product set*: you could not write every term as one choice from a shared
row list times one choice from a shared column list. The design matrix would
then be a long list of independent 2-D bumps, with one column per bump and no
Kronecker structure.

Centres on a rectangular grid supply that missing structure. Every centre is
a pair $(r_j, c_k)$ from fixed 1-D lists $\{r_j\}$ and $\{c_k\}$, and every
pair appears once. The 2-D dictionary is therefore the outer product of two
1-D dictionaries: columns of $A_r$ are the row Gaussians
$\exp(-(r-r_j)^2/(2\delta^2))$ evaluated on the image rows; columns of $A_c$
are the column Gaussians on the image columns. One amplitude $a_{jk}$ per
grid point is Mobahi's $a_j$ for that centre. The model at one width is

$$
Y \approx A_r\,A\,A_c^\top
  = \sum_{j,k} a_{jk}\,
    \bigl(\text{row Gaussian}_j\bigr)\,
    \bigl(\text{column Gaussian}_k\bigr)^\top,
$$

with $Y$ the residual image and $A = (a_{jk})$ a full matrix — not forced to
rank one. Separability lives in the *basis*, not in the coefficients: the
picture can need a dense $A$; the solver only needs the dictionary to factor.

### Two-step least squares

Vectorising the model gives
$(A_c \otimes A_r)\,\mathrm{vec}(A) \approx \mathrm{vec}(Y)$, a
Kronecker-structured least-squares problem [@fausett1994large_least]. Rather
than form the Kronecker product, fit in two ordinary steps with
`np.linalg.lstsq`:

1. $A_r M \approx Y$ for the mid-factor $M$,
2. $A_c A^\top \approx M^\top$ for the amplitudes $A$.

That is the same $A$ the big Kronecker solve would return (when the truncated
SVDs agree), at the cost of two small dense problems instead of one huge one.
Singular values below `rcond` times the largest are dropped (NumPy's default
truncated-SVD regularisation).

```{code-cell} ipython3
PHOTOGRAPHS = {
    "camera": ski.util.img_as_float(ski.data.camera()),
    "coins": ski.util.img_as_float(ski.data.coins()),
    "astronaut": ski.util.img_as_float(ski.color.rgb2gray(ski.data.astronaut())),
}

rows = []
FITS = {}
for name, image in PHOTOGRAPHS.items():
    levels, residual = gs.fit(image)
    FITS[name] = (levels, image.shape, residual)
    rows.append({
        "image": name,
        "shape": image.shape,
        "centres": sum(level.amplitudes.size for level in levels),
        "residual RMS": f"{np.sqrt(np.mean(residual**2)):.3e}",
        "δ levels": ", ".join(f"{d:g}" for d in gs.DELTAS),
    })
show_table(pd.DataFrame(rows))
```

```{code-cell} ipython3
levels, shape, residual = FITS["camera"]
rendered = gs.render(levels, shape)

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), layout="constrained")
bare(axes[0], "camera").imshow(PHOTOGRAPHS["camera"], cmap="gray", vmin=0, vmax=1)
bare(axes[1], r"Mobahi RBF sum $f$").imshow(rendered, cmap="gray", vmin=0, vmax=1)
bare(axes[2], "residual").imshow(residual, cmap="gray")
```

## 4. Exact Hessian of the smoothed sum

`gs.hessian` differentiates Mobahi's smoothed field. At $\sigma = 0$ that is
the Hessian of $f$ itself; at $\sigma > 0$ each level first takes the
smoothed width and amplitude factor, then the second derivatives.

```{code-cell} ipython3
sigma = 1.0
H = gs.hessian(levels, shape, sigma)
labels = (r"$H_{rr}$", r"$H_{rc}$", r"$H_{cc}$")

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), layout="constrained")
limit = max(np.abs(h).max() for h in H)
for ax, h, label in zip(axes, H, labels):
    bare(ax, fr"{label} of $[f \star k_{{\sigma}}]$, $\sigma = {sigma:g}$")
    ax.imshow(h, cmap="coolwarm", vmin=-limit, vmax=limit)
```

The test suite never compares a filter to the photograph. It renders $f$,
runs the filter on that render, and compares the result to
`gs.hessian(levels, shape, sigma)`. The photograph only chooses the
amplitudes.

## 5. What this standard is for

Mobahi's identity is exact for the continuous RBF sum. It does not assume a
band-limited interpolant, so it is a different referee from the DCT continuous
Gaussian of `hessian_review.md` §3.6. Use it when the question is how close a
discrete Hessian filter stays to the heat-semigroup derivatives of a
photograph-like field that was built to have those derivatives in closed form.

`hessian_testing.md` §5.2 measures the schemes against these three fits.
`tests/skimage2/feature/test_hessian_schemes.py` asserts absolute accuracy of
the installed filter against Mobahi's smoothed sums — not that it beats an
older scheme.
