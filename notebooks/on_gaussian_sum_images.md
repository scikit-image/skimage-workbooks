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
2026-11-28 | claude-code:claude-opus-5-5 | Add an RMS test of the whole Hessian, split into gain and shape; describe the ridge fit; compare joint and level-by-level fits; match `gaussian_sum.py` to the test file
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
`hessian_testing.md` §5.2 and `test_hessian_metrics.py` assert against.

Coordinates are in array order: the first index runs down the picture.

## 1. Setup

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skimage as ski
from nbhelper import show_table
from matplotlib.colors import LinearSegmentedColormap

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
than form the Kronecker product, fit in two steps:

1. $A_r M \approx Y$ for the mid-factor $M$,
2. $A_c A^\top \approx M^\top$ for the amplitudes $A$.

Each step is a ridge ([Tikhonov](https://en.wikipedia.org/wiki/Ridge_regression))
least-squares solve. In Wikipedia's notation, with design matrix $X$ ($A_r$
or $A_c$), response $Y$ and coefficients $\beta$, it minimizes
$\lVert Y - X\beta \rVert^2 + \lambda \lVert \beta \rVert^2$, with solution
$\hat\beta_\lambda = (X^\mathsf{T} X + \lambda I)^{-1} X^\mathsf{T} Y$.
$\lambda$ is `RIDGE` times the largest squared singular value of $X$. The ridge
shrinks the amplitudes and the cancellation between them, at some cost in fit
accuracy. `gs.ridge_solve` solves it as ordinary least squares on $X$ stacked
over $\sqrt{\lambda} I$, with $Y$ stacked over zeros.

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

### Why the levels are fitted one at a time

The levels need not be fitted one at a time. A *joint* fit puts the row bases
of all levels side by side, $R = [A_r^{(16)} \cdots A_r^{(1.5)}]$, and the
column bases likewise in $C$, then does the same two ridge solves once for
$Y \approx R\,A\,C^\top$. Because $A$ is a full matrix, the joint model also
holds cross terms between widths, such as a row Gaussian of width 16 times a
column Gaussian of width 1.5. These terms are axis-aligned anisotropic
Gaussians, not Mobahi's isotropic ones. They still smooth in closed form, one
axis at a time, but `gs.render` would need a width for each axis.

A second option keeps the levels and repeats the coarse-to-fine pass
(*backfitting*): each level is refitted to the residual of all the others.

The cell below compares the three on `camera`. The penalty $\lambda$ scales
with the largest singular value of the design matrix, which is larger for the
joint basis, so the joint fit needs a smaller `RIDGE` to give the same
residual.

```{code-cell} ipython3
def joint_fit(image, ridge):
    """Fit all levels in one two-step ridge solve; return amplitudes, residual."""
    rows = np.arange(image.shape[0], dtype=float)
    cols = np.arange(image.shape[1], dtype=float)
    R = np.hstack([gs.basis(rows, gs.grid_centres(len(rows), d), d) for d in gs.DELTAS])
    C = np.hstack([gs.basis(cols, gs.grid_centres(len(cols), d), d) for d in gs.DELTAS])
    A = gs.ridge_solve(C, gs.ridge_solve(R, image, ridge).T, ridge).T
    return A, image - R @ A @ C.T


def backfit_sweep(image, levels, residual, ridge=gs.RIDGE):
    """Refit each level, coarse to fine, to the residual of all the others."""
    rows = np.arange(image.shape[0], dtype=float)
    cols = np.arange(image.shape[1], dtype=float)
    new = []
    for level in levels:
        Rb = gs.basis(rows, level.row_centres, level.delta)
        Cb = gs.basis(cols, level.col_centres, level.delta)
        part = residual + Rb @ level.amplitudes @ Cb.T
        A = gs.ridge_solve(Cb, gs.ridge_solve(Rb, part, ridge).T, ridge).T
        residual = part - Rb @ A @ Cb.T
        new.append(gs.Level(level.delta, level.row_centres, level.col_centres, A))
    return new, residual


image = PHOTOGRAPHS["camera"]
levels, residual = FITS["camera"][0], FITS["camera"][2]


def summary(method, ridge, amplitudes, residual):
    return {
        "fit": method,
        "RIDGE": f"{ridge:.0e}",
        "amplitudes": sum(a.size for a in amplitudes),
        "max |a|": f"{max(np.abs(a).max() for a in amplitudes):.2f}",
        "residual RMS / image RMS": f"{np.sqrt(np.mean(residual**2) / np.mean(image**2)):.2%}",
    }


order_rows = [summary("level by level", gs.RIDGE, [l.amplitudes for l in levels], residual)]
refit, refit_residual = backfit_sweep(image, levels, residual)
order_rows.append(summary("level by level, 2 passes", gs.RIDGE,
                          [l.amplitudes for l in refit], refit_residual))
lower, lower_residual = gs.fit(image, ridge=1e-4)
order_rows.append(summary("level by level", 1e-4, [l.amplitudes for l in lower], lower_residual))
for ridge in (1e-3, 1e-4, 1e-5):
    A, joint_residual = joint_fit(image, ridge)
    order_rows.append(summary("joint", ridge, [A], joint_residual))
show_table(pd.DataFrame(order_rows))
```

The second pass leaves the residual as it was. The joint fit with the same
`RIDGE` fits worse, because its penalty is larger. With `RIDGE` lowered to
1e-5 it gives about the same residual as the level-by-level fit with 1e-4,
with smaller amplitudes, but with more than three times as many of them. We
keep the level-by-level fit, because each solve is small and every term
stays isotropic.

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

(sec-rms)=
## 5. An RMS test of the whole Hessian

`test_hessian_metrics.py` measures the largest gap between one Hessian element
and its exact value, divided by the peak exact element. That number comes from
one element at one pixel. An RMS test uses the whole matrix at every pixel.

Let $\hat H(\mathbf{x})$ be the computed Hessian at pixel $\mathbf{x}$ and
$H(\mathbf{x})$ the exact one. Sum the
[Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product)
over the interior pixels $I$, which are the pixels that an $8\sigma$ kernel
can reach without passing the image edge:

$$
\langle A, B \rangle = \sum_{\mathbf{x} \in I}
  \bigl(A_{rr} B_{rr} + 2 A_{rc} B_{rc} + A_{cc} B_{cc}\bigr),
\qquad \lVert A \rVert^2 = \langle A, A \rangle,
\qquad E = \frac{\lVert \hat H - H \rVert}{\lVert H \rVert}.
$$

$H_{rc}$ counts twice, because it is in the matrix twice. Then the term at one
pixel is $\lambda_1^2 + \lambda_2^2$, the sum of the squared eigenvalues, which
does not change when the image axes rotate. Each pixel counts in proportion to
its curvature, so flat regions, where a relative error at one pixel would
divide by almost zero, count for little.

$E$ splits into two parts. Let $g = \langle \hat H, H \rangle / \lVert H
\rVert^2$, the one gain that best scales the exact Hessian onto the computed
one, and $S = \lVert \hat H - gH \rVert / \lVert H \rVert$. The residual
$\hat H - gH$ is orthogonal to $H$, and $\hat H - H = (g - 1) H + (\hat H - gH)$,
so

$$
E^2 = (g - 1)^2 + S^2.
$$

A gain error multiplies every eigenvalue by the same factor. It leaves the
eigenvectors and the ratio of the eigenvalues alone, so a filter that uses only
that ratio, such as the shape index, does not see it. $S$ is the error that no
one gain explains: errors in the eigenvalue ratios and the principal directions,
and a gain that changes over the image.

```{code-cell} ipython3
import warnings
from itertools import combinations_with_replacement

import scipy.ndimage as ndi
from skimage.feature import hessian_matrix

FROBENIUS_WEIGHTS = (1, 2, 1)          # H_rr, H_rc, H_cc: H_rc is in the matrix twice


def interior(sigma):
    """Pixels that an 8-sigma kernel reaches without passing the image edge."""
    reach = int(8 * sigma + 0.5) + 1
    return (slice(reach, -reach),) * 2


def frobenius(A, B, inner):
    """<A, B>: the Frobenius inner product of two Hessians, summed over `inner`."""
    return sum(w * np.sum(a[inner] * b[inner]) for w, a, b in zip(FROBENIUS_WEIGHTS, A, B))


def rms_error(got, truth, sigma):
    """Relative RMS error E of the whole Hessian, and its split into gain and shape."""
    inner = interior(sigma)
    norm2 = frobenius(truth, truth, inner)
    gain = frobenius(got, truth, inner) / norm2
    diff = [a - b for a, b in zip(got, truth)]
    rest = [a - gain * b for a, b in zip(got, truth)]
    return {"E": np.sqrt(frobenius(diff, diff, inner) / norm2),
            "g - 1": gain - 1,
            "S": np.sqrt(frobenius(rest, rest, inner) / norm2)}


def max_error(got, truth, sigma):
    """The measure of `test_hessian_metrics.py`: largest gap / peak exact element."""
    inner = interior(sigma)
    peak = max(np.abs(t[inner]).max() for t in truth)
    return max(np.abs(a[inner] - b[inner]).max() for a, b in zip(got, truth)) / peak
```

Three schemes. Two are the installed `hessian_matrix`, with and without
`use_gaussian_derivatives`. The third is the moment-corrected Gaussian
derivative of the `hessian-refactor` branch, which `test_hessian_metrics.py`
tests. `taps` is a transcription of `_gaussian_derivative_kernel` on that
branch. The two agree to the last bit for `sigma` from 0.1 to 4 and all three
orders; that check ran in the branch environment, outside this notebook.

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


def corrected(image, sigma):
    """Moment-corrected Gaussian derivatives: one separable pass per element."""
    elements = []
    for ax0, ax1 in combinations_with_replacement(range(2), 2):
        orders = [0, 0]
        orders[ax0] += 1
        orders[ax1] += 1
        result = image
        for axis, order in enumerate(orders):
            result = ndi.correlate1d(result, taps(sigma, order), axis=axis, mode="nearest")
        elements.append(result)
    return elements


def installed(use_gaussian_derivatives):
    """The installed `hessian_matrix`, with or without Gaussian derivatives."""
    def scheme(image, sigma):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return hessian_matrix(image, sigma=sigma, mode="nearest",
                                  use_gaussian_derivatives=use_gaussian_derivatives)
    return scheme


SCHEMES = {"finite differences (installed)": installed(False),
           "Gaussian derivatives (installed)": installed(True),
           "moment-corrected (branch)": corrected}
```

Checks of the test itself, on `camera` at `sigma` 1, with the moment-corrected
Hessian as $\hat H$:

```{code-cell} ipython3
def rotate_hessian(H, angle):
    """R H R^T at every pixel, for a rotation of the axes by `angle`."""
    co, si = np.cos(angle), np.sin(angle)
    hrr, hrc, hcc = H
    return [co * co * hrr - 2 * co * si * hrc + si * si * hcc,
            co * si * (hrr - hcc) + (co * co - si * si) * hrc,
            si * si * hrr + 2 * co * si * hrc + co * co * hcc]


levels, shape, _ = FITS["camera"]
camera_sum = gs.render(levels, shape)
truth = gs.hessian(levels, shape, 1.0)
got = corrected(camera_sum, 1.0)
turned = np.deg2rad(35)
checks = {
    "exact against itself": rms_error(truth, truth, 1.0),
    "exact times 1.01": rms_error([1.01 * h for h in truth], truth, 1.0),
    "moment-corrected": rms_error(got, truth, 1.0),
    "moment-corrected, axes rotated 35 degrees":
        rms_error(rotate_hessian(got, turned), rotate_hessian(truth, turned), 1.0),
}
show_table(pd.DataFrame(
    [{"Hessian": name, **{k: f"{v:+.6e}" for k, v in r.items()},
      "E**2 - (g-1)**2 - S**2": f"{r['E']**2 - r['g - 1']**2 - r['S']**2:+.1e}"}
     for name, r in checks.items()]), index="Hessian")
```

The exact Hessian scores 0. A gain of 1.01 gives $g - 1 = 0.01$ and $S = 0$.
Rotating both Hessians leaves $E$, $g$ and $S$ unchanged to rounding, and
$E^2 = (g - 1)^2 + S^2$ to rounding.

Each scheme on each fitted photograph. The last column is the measure of
`test_hessian_metrics.py`:

```{code-cell} ipython3
RMS_SIGMAS = (0.5, 1.0, 2.0, 4.0)
RMS = {}
rows = []
for name, (levels, shape, _) in FITS.items():
    rendered = gs.render(levels, shape)
    for sigma in RMS_SIGMAS:
        truth = gs.hessian(levels, shape, sigma)
        for scheme, method in SCHEMES.items():
            got = method(rendered, sigma)
            RMS[name, sigma, scheme] = r = rms_error(got, truth, sigma)
            rows.append({"image": name, "sigma": sigma, "scheme": scheme,
                         "E": f"{r['E']:.2e}", "g - 1": f"{r['g - 1']:+.2e}",
                         "S": f"{r['S']:.2e}",
                         "max gap / peak": f"{max_error(got, truth, sigma):.2e}"})
table = pd.DataFrame(rows)
show_table(table[table["image"] == "camera"].drop(columns="image"), index="sigma")
```

```{code-cell} ipython3
spread = pd.DataFrame(
    [{"sigma": sigma, "scheme": scheme,
      **{f"E, {name}": f"{RMS[name, sigma, scheme]['E']:.2e}" for name in FITS}}
     for sigma in RMS_SIGMAS for scheme in SCHEMES])
show_table(spread, index="sigma")
```

$E$ and the max measure rank the schemes in the same order at each `sigma`.
$E$ changes by less than 20% between the three photographs. The split says what kind of error each
scheme makes:

- The finite-difference Hessian loses gain: $g - 1$ is about $-0.34$ at
  `sigma` 0.5, $-0.26$ at 1, $-0.11$ at 2 and $-0.03$ at 4. Its shape error
  $S$ is 50% to 85% of $|g - 1|$.
- The installed Gaussian-derivative Hessian loses almost all its gain at
  `sigma` 0.5, $g - 1 = -0.94$, with $S$ only 0.012. A filter that uses only
  the eigenvalue ratio would hardly see that error; the max measure, at 0.94,
  cannot tell it from a shape error. At `sigma` 1 both parts are about 0.01.
- The moment-corrected Hessian has $g - 1$ about 0.015 and $S$ about 0.08 at
  `sigma` 0.5. At `sigma` 1 its error is almost all
  shape: $S$ about $2 \times 10^{-4}$, against $g - 1$ about
  $7 \times 10^{-5}$. From `sigma` 2 both are rounding.

Where the error lies on the image. $\lVert H \rVert_F$ at each pixel is the
exact curvature. $\lVert \hat H - H \rVert_F$ is the whole error, and
$\lVert \hat H - gH \rVert_F$ the part that no one gain explains. Each row
shares one scale for its two error maps:

```{code-cell} ipython3
def frobenius_map(H):
    """||H||_F at each pixel."""
    return np.sqrt(sum(w * h**2 for w, h in zip(FROBENIUS_WEIGHTS, H)))


SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", C_ONE, INK])
MAP_SIGMA, VIEW = 1.0, (slice(60, 220), slice(180, 340))
levels, shape, _ = FITS["camera"]
camera_sum = gs.render(levels, shape)
truth = gs.hessian(levels, shape, MAP_SIGMA)

fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0), layout="constrained")
for row, scheme in zip(axes, ("finite differences (installed)", "moment-corrected (branch)")):
    got = SCHEMES[scheme](camera_sum, MAP_SIGMA)
    gain = 1 + RMS["camera", MAP_SIGMA, scheme]["g - 1"]
    whole = frobenius_map([a - b for a, b in zip(got, truth)])[VIEW]
    rest = frobenius_map([a - gain * b for a, b in zip(got, truth)])[VIEW]
    bare(row[0], "camera sum").imshow(camera_sum[VIEW], cmap="gray")
    bare(row[1], r"$\|H\|_F$, exact").imshow(frobenius_map(truth)[VIEW], cmap=SEQ)
    top = whole.max()
    bare(row[2], rf"$\|\hat H - H\|_F$, {scheme}").imshow(whole, cmap=SEQ, vmin=0, vmax=top)
    shown = bare(row[3], rf"$\|\hat H - gH\|_F$, g = {gain:.5f}").imshow(
        rest, cmap=SEQ, vmin=0, vmax=top)
    fig.colorbar(shown, ax=row[3], fraction=0.046).outline.set_visible(False)
    for ax in row:
        ax.title.set_fontsize(7.5)
fig.suptitle(rf"camera at $\sigma$ = {MAP_SIGMA:g}", color=MUTED, fontsize=9)
```

For finite differences the whole error has the pattern of the exact curvature:
most of it is the lost gain, and removing the best gain leaves a smaller error
at fine detail. For the moment-corrected Hessian the two error maps are the
same, because $g$ is 1 to within $10^{-4}$.

`test_hessian_metrics.py` asserts the two parts separately, with the floors
that this table measures, for the moment-corrected Hessian.

## 6. What this standard is for

Mobahi's identity is exact for the continuous RBF sum. It does not assume a
band-limited interpolant, so it is a different referee from the DCT continuous
Gaussian of `hessian_review.md` §3.6. Use it when the question is how close a
discrete Hessian filter stays to the heat-semigroup derivatives of a
photograph-like field that was built to have those derivatives in closed form.

`hessian_testing.md` §5.2 measures the schemes against these three fits.
`tests/skimage2/feature/test_hessian_metrics.py` asserts absolute accuracy of
the installed filter against Mobahi's smoothed sums — not that it beats an
older scheme.
