---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Meijering neuriteness: paper, scales, and scikit-image

Does `skimage.filters.meijering` match Meijering *et al.* (2004)? The paper
gives a **single-scale** neuriteness $\rho$, with one global normaliser. The
library adds a multiscale loop and a different per-scale normaliser. This
notebook answers four questions: does the filter reproduce the paper's
algorithm, which is single-scale; what do the choices for scaling across
$\sigma$ do to **detection** and to **width**; how does global normalisation
compare with local; and how did the current choice arise, and how does it
compare with other implementations.

Coordinates are array order. Bright ridges on a dark background use
`black_ridges=False` in scikit-image. The $\alpha$ sign is settled in
`meijering_alpha.md`; here the paper value $\alpha=-1/3$ is used unless a cell
says otherwise.

Paper: [Meijering *et al.*, Cytometry A 58:167–176 (2004)](https://doi.org/10.1002/cyto.a.20022),
local copy `library/meijering2004filter.pdf`. Author code: NeuronJ
([ImageScience/NeuronJ](https://github.com/ImageScience/NeuronJ)).

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import linalg

import skimage as ski
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import meijering, sato

from nbhelper import show_table
```

```{code-cell} ipython3
# Slots 1–3 of the workbook categorical palette (validated all-pairs).
C_ONE, C_TWO, C_THREE = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd5"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f4", C_ONE])

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


def recede(ax, title=None):
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    if title:
        ax.set_title(title)
    return ax
```

## Helpers

```{code-cell} ipython3
def modified_eigvals(eigvals, alpha):
    """Paper λ'_i = λ_i + α Σ_{j≠i} λ_j via the circulant used in skimage."""
    ndim = eigvals.shape[0]
    mtx = linalg.circulant([1, *[alpha] * (ndim - 1)]).astype(eigvals.dtype)
    return np.tensordot(mtx, eigvals, 1)


def selected_lambda(eigvals, alpha):
    """Larger-magnitude modified eigenvalue at each pixel."""
    vals = modified_eigvals(eigvals, alpha)
    return np.take_along_axis(vals, np.abs(vals).argmax(0)[None], 0).squeeze(0)


def hessian_eigvals(image, sigma, mode="nearest"):
    H = hessian_matrix(
        image, sigma, mode=mode, use_gaussian_derivatives=True
    )
    return hessian_matrix_eigvals(H)


def paper_rho(image, sigma, alpha=-1 / 3, mode="nearest"):
    """Meijering 2004 neuriteness for bright ridges (single scale).

    ρ(x) = λ(x)/λ_min if λ(x) < 0, else 0,
    with λ the larger-magnitude modified eigenvalue and λ_min the global
    minimum of λ (most negative).
    """
    lam = selected_lambda(hessian_eigvals(image, sigma, mode), alpha)
    out = np.zeros_like(lam)
    neg = lam < 0
    if np.any(neg):
        out[neg] = lam[neg] / lam.min()
    return out


def ski_one_scale(image, sigma, alpha, black_ridges=False, mode="nearest"):
    """One iteration of the current skimage recipe (no multiscale max)."""
    img = image.astype(np.float64, copy=False)
    if not black_ridges:
        img = -img
    lam = selected_lambda(hessian_eigvals(img, sigma, mode), alpha)
    vals = np.maximum(lam, 0)
    peak = vals.max()
    return vals / peak if peak > 0 else vals


def neuronj_adjusted(Hxx, Hxy, Hyy, bright=True):
    """Closed form from NeuronJ Costs.run (α = −1/3 baked in)."""
    inv = 1.0 if bright else -1.0
    b1 = inv * (Hxx + Hyy)
    b2 = inv * (Hxx - Hyy)
    d = np.sqrt(4 * Hxy * Hxy + b2 * b2)
    L1 = (b1 + 2 * d) / 3.0
    L2 = (b1 - 2 * d) / 3.0
    return L1, L2


def neuronj_magnitude(image, sigma, bright=True, mode="nearest"):
    """NeuronJ selected |λ'| when λ' < 0, else 0 (before display scaling)."""
    Hrr, Hrc, Hcc = hessian_matrix(
        image, sigma, mode=mode, use_gaussian_derivatives=True
    )
    # NeuronJ uses (x, y) = (column, row): Hxx=Hcc, Hyy=Hrr, Hxy=Hrc.
    L1, L2 = neuronj_adjusted(Hcc, Hrc, Hrr, bright=bright)
    pick1 = np.abs(L1) > np.abs(L2)
    out = np.where(pick1, L1, L2)
    return np.where(out < 0, np.abs(out), 0.0)


def gaussian_ridge(width, n=161, weight=1.0):
    """Vertical bright ridge of Gaussian profile, width = σ of the profile."""
    mid = n // 2
    cols = np.arange(n, dtype=float)
    line = weight * np.exp(-((cols - mid) ** 2) / (2 * width**2))
    return np.broadcast_to(line, (n, n)).copy(), mid


def gaussian_blob(width, n=161, weight=1.0):
    mid = n // 2
    rows, cols = np.indices((n, n), dtype=float)
    return weight * np.exp(
        -((rows - mid) ** 2 + (cols - mid) ** 2) / (2 * width**2)
    )
```

```{code-cell} ipython3
def meijering_core(image, sigma, alpha=-1 / 3, power=0.0, mode="nearest"):
    """Paper-polarity response, with sigma**power on the Hessian.

    The response is -lambda where lambda < 0, and 0 elsewhere, matching
    `paper_rho` and the fusion in §4 for bright ridges.
    """
    H = hessian_matrix(image, sigma, mode=mode, use_gaussian_derivatives=True)
    if power:
        H = [sigma**power * e for e in H]
    lam = selected_lambda(hessian_matrix_eigvals(H), alpha)
    return np.where(lam < 0, -lam, 0.0)


def detection_and_width(image, sigmas, alpha=-1 / 3, power=0.0, mode="nearest"):
    """Detection map max_sigma and width map argmax_sigma of the core response."""
    stack = np.stack([meijering_core(image, s, alpha, power, mode) for s in sigmas])
    return stack.max(0), np.asarray(sigmas, float)[stack.argmax(0)]


def ridge_peak_sigma(width, gamma):
    """Winning scale of sigma**(2*gamma) * width * (width**2 + sigma**2)**-1.5."""
    return width * np.sqrt(2 * gamma / (3 - 2 * gamma))


def ridge_peak_value(width, gamma):
    """Response at that winning scale, in closed form."""
    p = 2 * gamma
    return (width ** (p - 2) * p ** (p / 2)
            * (3 - p) ** ((3 - p) / 2) / 3 ** 1.5)
```

## 1. What the paper specifies

The appendix of Meijering *et al.* (2004) builds a modified Hessian so that the
eigenvalues reshape a second-order Gaussian filter to be flat along a ridge.
In 2-D the free parameter is fixed by that flatness condition:

$$
\alpha = -\frac13.
$$

Modified eigenvalues:

$$
\lambda'_1 = \lambda_1 + \alpha\lambda_2,\qquad
\lambda'_2 = \lambda_2 + \alpha\lambda_1.
$$

Neuriteness (bright structures):

$$
\rho(x) =
\begin{cases}
\lambda(x)/\lambda_{\min} & \text{if }\lambda(x)<0,\\
0 & \text{if }\lambda(x)\ge 0,
\end{cases}
$$

where $\lambda$ is the **larger-magnitude** $\lambda'$, and $\lambda_{\min}$ is
the smallest $\lambda$ **over the whole image**. Dark line-like responses
($\lambda\ge 0$) are set to zero.

The Gaussian scale $\sigma$ is a **single** tuning knob. Table 1 of the paper
fixes $\sigma=2.0$ for the validation. The text says the detector “can be tuned
to neurites of specific width”; it does **not** define a multiscale maximum
over several $\sigma$.

NeuronJ, the authors’ ImageJ tool, matches that design: one “Hessian smoothing
scale”, $\alpha=-1/3$ in closed form, then a global display scale of the
selected magnitudes.

```{code-cell} ipython3
ridge, mid = gaussian_ridge(4.0)
sigma = 3.0
rho = paper_rho(ridge, sigma, alpha=-1 / 3)
nj = neuronj_magnitude(ridge, sigma, bright=True)
nj_as_rho = nj / nj.max() if nj.max() > 0 else nj
lam = selected_lambda(hessian_eigvals(ridge, sigma), -1 / 3)
paper_mag = np.where(lam < 0, -lam, 0.0)

show_table(pd.DataFrame([{
    "check": "paper ρ peak",
    "value": f"{rho.max():.6f}",
}, {
    "check": "paper ρ vs NeuronJ / max",
    "value": f"{np.max(np.abs(rho - nj_as_rho)):.2e}",
}, {
    "check": "NeuronJ |λ| vs |selected λ'| (α=−1/3)",
    "value": f"{np.max(np.abs(nj - paper_mag)):.2e}",
}]))
```

NeuronJ’s eigenvalue step is the paper’s $\alpha=-1/3$ rewrite of the 2-D
characteristic equation, not a separate algorithm.

+++

## 2. Single-scale scikit-image versus the paper

At one $\sigma$, with $\alpha=-1/3$ and `black_ridges=False`, the current
skimage loop (clip positive responses after polarity flip, divide by the image
maximum) matches $\rho$ on a bright ridge.

```{code-cell} ipython3
rho_ski = ski_one_scale(ridge, sigma, alpha=-1 / 3, black_ridges=False)
rho_lib = meijering(
    ridge, sigmas=[sigma], alpha=-1 / 3, black_ridges=False, mode="nearest"
)

show_table(pd.DataFrame([{
    "pair": "paper ρ vs ski_one_scale(α=−1/3)",
    "max |Δ|": f"{np.max(np.abs(rho - rho_ski)):.2e}",
}, {
    "pair": "paper ρ vs meijering(sigmas=[σ], α=−1/3)",
    "max |Δ|": f"{np.max(np.abs(rho - rho_lib)):.2e}",
}, {
    "pair": "default α=+1/3 vs α=−1/3 (same ridge, normalised)",
    "max |Δ|": f"{np.max(np.abs(ski_one_scale(ridge, sigma, 1/3, False) - rho_ski)):.2e}",
}]))
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
for ax, img, title in (
    (axes[0], ridge, "input ridge"),
    (axes[1], rho, r"paper $\rho$"),
    (axes[2], rho_lib, "meijering, one σ, α=−1/3"),
):
    bare(ax, title).imshow(img, cmap=SEQ)
fig.suptitle(f"single scale σ={sigma}, bright ridge", y=1.02)
fig.tight_layout()
```

On a **single** isolated ridge, global normalisation maps any positive multiple
of the same spatial pattern to the same $\rho$. So $\alpha=+1/3$ can look
identical to $\alpha=-1/3$ after `/max`. Discrimination needs structures that
are **not** scalar multiples of each other.

```{code-cell} ipython3
blob = gaussian_blob(4.0)
rows = []
for alpha in (-1 / 3, 1 / 3):
    # Raw selected response before /max (bright-ridge polarity as in the paper).
    lam_r = selected_lambda(hessian_eigvals(ridge, sigma), alpha)
    lam_b = selected_lambda(hessian_eigvals(blob, sigma), alpha)
    raw_r = (-lam_r[mid, mid]) if lam_r[mid, mid] < 0 else 0.0
    raw_b = (-lam_b[mid, mid]) if lam_b[mid, mid] < 0 else 0.0
    rows.append({
        "alpha": f"{alpha:+.4f}",
        "raw ridge centre": f"{raw_r:.4f}",
        "raw blob centre": f"{raw_b:.4f}",
        "ridge / blob": f"{raw_r / max(raw_b, 1e-12):.3f}",
    })
show_table(pd.DataFrame(rows), index="alpha")
```

Paper $\alpha=-1/3$ prefers the ridge over the blob. Default skimage
$\alpha=+1/(2+1)=+1/3$ prefers the blob. That is the $\alpha$ defect measured
in `meijering_alpha.md`; it is separate from scaling across $\sigma$.

+++

## 3. Scale: detection strength versus estimated width

Two different jobs share the same eigenvalues.

**Detection.** Is there a ridge-like structure here, and how strongly? A score
that is comparable across the image (and, if multiscale, across $\sigma$) is
enough. The paper’s $\rho\in[0,1]$ is built for a **cost** $C_\lambda=1-\rho$
in live-wire tracing at one fixed $\sigma$.

**Width.** Which $\sigma$ best matches the local thickness? That needs a
scale-space comparison in which the **winning** $\sigma$ tracks width.

### 3.1 The normalisation, from Lindeberg

A second derivative of a Gaussian-smoothed image carries a factor that falls
with scale. Lindeberg normalises by multiplying a derivative of order $n$ by
$\sigma^{n\gamma}$; for a second derivative that is $\sigma^{2\gamma}$. The
choice of $\gamma$ sets what the winning scale means.

For a cylindrical Gaussian ridge of width $w$, the cross-ridge second
derivative on the axis is $e(\sigma) = w\,(w^2+\sigma^2)^{-3/2}$. The normalised
response is $r_\gamma(\sigma) = \sigma^{2\gamma} e(\sigma)$. Setting
$r_\gamma'(\sigma)=0$ gives

$$
\sigma^{*}(\gamma) = w\sqrt{\frac{2\gamma}{3-2\gamma}}, \qquad 0 < \gamma < \tfrac32,
$$

and the value there is

$$
D_\gamma = w^{\,2\gamma-2}\,c(\gamma).
$$

| $\gamma$ | factor | $\sigma^{*}$ | detection value $D_\gamma$ |
| --- | --- | --- | --- |
| $1$ | $\sigma^2$ | $w\sqrt2$ | $0.3849$, independent of $w$ |
| $3/4$ | $\sigma^{1.5}$ | $w$ | $\propto w^{-1/2}$ |
| $0$ | none | $0$ | $\propto w^{-2}$ |

$\gamma=1$ gives the same detection value for every ridge width. $\gamma=3/4$
makes the winning scale equal the width, but weakens wide ridges by
$w^{-1/2}$. Meijering *et al.* never fuse scales, so they never face this
choice; the paper's single-$\sigma$ $\rho$ is a detection map at a fixed width.

```{code-cell} ipython3
WIDTHS_PLOT = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0])
GRID_G = np.linspace(0.02, 1.4, 400)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0))
axes[0].plot(GRID_G, np.sqrt(2 * GRID_G / (3 - 2 * GRID_G)), color=INK, lw=1.8)
axes[0].axhline(1.0, color=GRID, lw=1)
for gamma, colour, label in ((1.0, C_TWO, "1"), (0.75, C_ONE, "3/4")):
    axes[0].plot([gamma], [ridge_peak_sigma(1.0, gamma)], "o", color=colour,
                 ms=7, zorder=3)
    axes[0].annotate(f"gamma = {label}", (gamma, ridge_peak_sigma(1.0, gamma)),
                     textcoords="offset points", xytext=(6, -13), fontsize=8,
                     color=colour)
recede(axes[0], "winning scale against exponent")
axes[0].set_xlabel("gamma", fontsize=8, color=MUTED)
axes[0].set_ylabel("winning sigma / ridge width", fontsize=8, color=MUTED)

axes[1].loglog(WIDTHS_PLOT, WIDTHS_PLOT ** -2, "o-", color=MUTED, lw=1.6,
               label="raw  (gamma = 0)")
for gamma, colour, label in ((1.0, C_TWO, "sigma**2  (gamma = 1)"),
                             (0.75, C_ONE, "sigma**1.5  (gamma = 3/4)")):
    axes[1].loglog(WIDTHS_PLOT, [ridge_peak_value(w, gamma) for w in WIDTHS_PLOT],
                   "o-", color=colour, lw=1.8, label=label)
recede(axes[1], "detection value against ridge width")
axes[1].set_xlabel("ridge width w", fontsize=8, color=MUTED)
axes[1].set_ylabel("value at the winning scale", fontsize=8, color=MUTED)
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
```

### 3.2 Detection map and width map

One scale scan gives two different pictures:

- the **detection map** $D(x)=\max_\sigma r_\gamma(\sigma,x)$, the fused score;
- the **width map** $W(x)=\arg\max_\sigma r_\gamma(\sigma,x)$, the winning scale.

The shipped `meijering` returns a detection map. It never returns a width map.
$\gamma$ moves the two maps in opposite directions (§3.1), so it is not a free
magnitude scale.

```{code-cell} ipython3
HT, WT = 180, 220
tyy, txx = np.indices((HT, WT), dtype=float)
true_width = 1.5 + 6.5 * (tyy / (HT - 1))
tapered = np.exp(-((txx - WT / 2) ** 2) / (2 * true_width ** 2))
DENSE = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

MAPS = {}
for gamma in (0.75, 1.0):
    det, win = detection_and_width(tapered, DENSE, power=2 * gamma)
    MAPS[gamma] = (det, np.where(det < 0.15 * det.max(), np.nan, win))

fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.0))
bare(axes[0, 0], "input; true width grows downward")
axes[0, 0].imshow(tapered, cmap=SEQ)
for ax, gamma in zip(axes[0, 1:], (0.75, 1.0)):
    im = ax.imshow(MAPS[gamma][1], cmap="viridis", vmin=1, vmax=9)
    bare(ax, f"width map: gamma = {gamma:g}")
    fig.colorbar(im, ax=ax, fraction=0.046, label="winning sigma")
for ax, gamma in zip(axes[1, :2], (0.75, 1.0)):
    im = ax.imshow(MAPS[gamma][0], cmap=SEQ, vmin=0,
                   vmax=max(d.max() for d, _ in MAPS.values()))
    bare(ax, f"detection map: gamma = {gamma:g}")
    fig.colorbar(im, ax=ax, fraction=0.046)
axes[1, 2].plot(true_width[:, 0], np.arange(HT), color=INK, lw=1.8,
                label="true width")
for gamma, colour in ((0.75, C_ONE), (1.0, C_TWO)):
    axes[1, 2].plot(MAPS[gamma][1][:, WT // 2], np.arange(HT), color=colour,
                    lw=1.5, label=f"width map, gamma = {gamma:g}")
recede(axes[1, 2], "centreline: winning sigma against true width")
axes[1, 2].set_xlabel("sigma  (and true width)", fontsize=8, color=MUTED)
axes[1, 2].legend(frameon=False, fontsize=7)
fig.tight_layout()
```

At $\gamma=3/4$ the width map tracks the taper, and the detection map fades as
the ridge widens. At $\gamma=1$ the width map sits above the taper (peaks at
$w\sqrt2$), and the detection map stays bright. To read a width from the
$\gamma=1$ map, divide the winning scale by $\sqrt2$. The shipped filter
returns neither map directly: its per-scale `/max` is a third normalisation
(§4).

### 3.3 The same statement on the filter

```{code-cell} ipython3
def center_strength(image, sigma, alpha=-1 / 3, scale_norm=False):
    """−λ at the centre if λ<0, optionally with σ² on the Hessian."""
    H = hessian_matrix(
        image, sigma, mode="nearest", use_gaussian_derivatives=True
    )
    if scale_norm:
        H = [sigma**2 * e for e in H]
    lam = selected_lambda(hessian_matrix_eigvals(H), alpha)
    c = lam[mid, mid]
    return (-c) if c < 0 else 0.0


widths = (1.5, 3.0, 5.0, 8.0)
sigmas = (1.0, 2.0, 3.0, 5.0, 8.0)
rows = []
for w in widths:
    img, _ = gaussian_ridge(w)
    raw = [center_strength(img, s, scale_norm=False) for s in sigmas]
    s2 = [center_strength(img, s, scale_norm=True) for s in sigmas]
    # Per-scale paper ρ at the centre (each scale normalised on its own image).
    rho_c = [paper_rho(img, s)[mid, mid] for s in sigmas]
    rows.append({
        "width": w,
        "argmax raw": sigmas[int(np.argmax(raw))],
        "argmax σ²-raw": sigmas[int(np.argmax(s2))],
        "argmax per-σ ρ": sigmas[int(np.argmax(rho_c))],
        "raw peaks": np.round(raw, 3).tolist(),
        "σ² peaks": np.round(s2, 3).tolist(),
    })
show_table(pd.DataFrame(rows), index="width")
```

Without $\sigma^{2}$, raw strength always peaks at the **smallest** $\sigma$ in
this set. Per-scale $\rho$ (divide by that scale’s own $\lambda_{\min}$) also
fails as a width estimator here: every scale’s centre score is driven toward 1
on a clean ridge. With $\sigma^{2}$ on the Hessian, the winning $\sigma$ moves
with width.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
for ax, scale_norm, title in (
    (axes[0], False, "raw −λ (no σ²)"),
    (axes[1], True, r"σ²-normalised −λ"),
):
    for w, color in zip(widths, (C_ONE, C_TWO, C_THREE, INK)):
        img, _ = gaussian_ridge(w)
        ys = [center_strength(img, s, scale_norm=scale_norm) for s in sigmas]
        recede(ax).plot(sigmas, ys, "-o", ms=4, color=color, label=f"width={w}")
    ax.set_xlabel("filter σ")
    ax.set_ylabel("centre strength")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)
fig.suptitle("width selection needs scale normalisation; detection alone does not",
             y=1.02)
fig.tight_layout()
```

**Trade-off.** If the product goal is “highlight neurites for tracing at a
known thickness” (the paper), pick one $\sigma$ and use $\rho$. If the goal is
“estimate width” or “fuse unknown widths”, you need an explicit scale-normalised
score (as Sato/Frangi do), not only a multiscale `maximum`.

+++

## 4. Global versus local normalisation

The paper’s $\lambda_{\min}$ is a **global** statistic of the processed image
(or ROI). So is skimage’s `vals.max()` at each scale. Both make the value at a
pixel depend on pixels arbitrarily far away.

```{code-cell} ipython3
photo = ski.util.img_as_float(ski.data.camera())[::2, ::2]
far = (slice(100, 200), slice(100, 200))
edited = photo.copy()
edited[:, 10:14] = 1.0  # strong bright bar that can reset the global peak


def far_change(before, after):
    return float(
        np.abs(before[far] - after[far]).max()
        / max(np.abs(before[far]).max(), 1e-12)
    )


rows = [
    {
        "method": "paper ρ, σ=2",
        "far max|Δ|/peak": f"{far_change(paper_rho(photo, 2), paper_rho(edited, 2)):.1%}",
    },
    {
        "method": "meijering sigmas=[2], α=−1/3",
        "far max|Δ|/peak": f"{far_change(meijering(photo, sigmas=[2], alpha=-1/3, black_ridges=False, mode='nearest'), meijering(edited, sigmas=[2], alpha=-1/3, black_ridges=False, mode='nearest')):.1%}",
    },
    {
        "method": "meijering sigmas=(1,3,5), α=−1/3",
        "far max|Δ|/peak": f"{far_change(meijering(photo, sigmas=(1, 3, 5), alpha=-1/3, black_ridges=False, mode='nearest'), meijering(edited, sigmas=(1, 3, 5), alpha=-1/3, black_ridges=False, mode='nearest')):.1%}",
    },
    {
        "method": "sato sigmas=(1,3,5) (no /max)",
        "far max|Δ|/peak": f"{far_change(sato(photo, sigmas=(1, 3, 5), black_ridges=False, mode='nearest'), sato(edited, sigmas=(1, 3, 5), black_ridges=False, mode='nearest')):.1%}",
    },
]
show_table(pd.DataFrame(rows), index="method")
```

Cropping the image changes the interval that the maximum is taken over, so it
changes the shared pixels without editing any of them.

```{code-cell} ipython3
def crop_change(fn, image, size=140, margin=40):
    whole, part = fn(image)[:size, :size], fn(image[:size, :size])
    inner = (slice(margin, -margin),) * 2
    return float(np.abs(whole[inner] - part[inner]).max()
                 / max(np.abs(whole[inner]).max(), 1e-12))


crop_rows = [
    {"method": "paper ρ, σ=2",
     "crop max|Δ|/peak": f"{crop_change(lambda im: paper_rho(im, 2), photo):.1%}"},
    {"method": "meijering, α=−1/3",
     "crop max|Δ|/peak": f"{crop_change(lambda im: meijering(im, alpha=-1/3, black_ridges=False, mode='nearest'), photo):.1%}"},
    {"method": "sato",
     "crop max|Δ|/peak": f"{crop_change(lambda im: sato(im, black_ridges=False, mode='nearest'), photo):.1%}"},
]
show_table(pd.DataFrame(crop_rows), index="method")
```

`meijering` moves by 8% of the peak in the shared interior. The paper's
$\rho$ and `sato` do not move here, but for different reasons: the crop retained
the pixel that sets $\rho$'s global minimum, so that probe reads zero by luck;
`sato` has no global statistic at all. An earlier probe that edited a pixel
moved $\rho$ as well.

| policy | what it does | detection | width / strength |
| --- | --- | --- | --- |
| Paper $\lambda/\lambda_{\min}$ (global, one $\sigma$) | Maps the strongest negative response to 1 | Good for cost maps at fixed thickness | No multiscale width |
| skimage `/max` each $\sigma$, then `maximum` | Every scale’s peak becomes 1 before fusion | Favours “some scale responds” even if that scale is weak in absolute terms | Destroys absolute strength; poor width cue |
| No `/max`, raw or $\sigma^{2}$-scaled, then `maximum` | Strongest absolute (or scale-normalised) response wins | Comparable to Sato/Frangi fusion | Width needs $\sigma^{2}$ (or similar) |
| Local (patch / soft) normalisation | Score relative to a neighbourhood | More invariant to uneven illumination | Extra parameters; not in the paper |

Local normalisation is **not** what Meijering *et al.* describe. Global
normalisation **is**. The open product choice for skimage is whether a
**multiscale** API should keep a global `/max` **inside** each scale before
`maximum`.

```{code-cell} ipython3
# Multiscale fusion: per-scale /max vs raw max vs σ²-max on a two-width scene.
wide, _ = gaussian_ridge(8.0, n=201)
narrow = np.zeros_like(wide)
nmid = 201 // 2
cols = np.arange(201, dtype=float)
narrow += np.broadcast_to(
    np.exp(-((cols - 40) ** 2) / (2 * 2.0**2)), wide.shape
)
scene = np.clip(wide + narrow, 0, 1)
sigmas = (1.0, 2.0, 4.0, 8.0)


def fuse(image, sigmas, mode):
    acc = None
    for s in sigmas:
        H = hessian_matrix(
            image, s, mode="nearest", use_gaussian_derivatives=True
        )
        if mode == "sigma2":
            H = [s**2 * e for e in H]
        lam = selected_lambda(hessian_matrix_eigvals(H), -1 / 3)
        score = np.where(lam < 0, -lam, 0.0)
        if mode == "per_scale_max":
            peak = score.max()
            score = score / peak if peak > 0 else score
        acc = score if acc is None else np.maximum(acc, score)
    return acc


fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.8))
panels = [
    (scene, "scene (widths 2 and 8)"),
    (fuse(scene, sigmas, "raw"), "max of raw −λ"),
    (fuse(scene, sigmas, "per_scale_max"), "max of per-σ /max"),
    (fuse(scene, sigmas, "sigma2"), "max of σ² −λ"),
]
for ax, (img, title) in zip(axes, panels):
    bare(ax, title).imshow(img, cmap=SEQ)
fig.suptitle("same α=−1/3; only the cross-scale policy changes", y=1.03)
fig.tight_layout()
```

## 5. History of the scikit-image choice

| stage | what it says or does about scale |
| --- | --- |
| PR [#3515](https://github.com/scikit-image/scikit-image/pull/3515) (2018) | First `meijering`. Multiscale by default (`sigmas=range(1,10,2)`). Default $\alpha=+1/\mathrm{ndim}$. Hessian eigenvalues scaled by $\sigma^{2}$ in a shared helper. Per-scale `abs(aux)/abs(max(aux))`, then `max` over scales. |
| [#5561](https://github.com/scikit-image/scikit-image/issues/5561) | The $\sigma^{2}$ on the Hessian eigenvalues is not in the paper; with several $\sigma$ it "unduly" prefers large $\sigma$. Also flags the $\alpha$ sign. |
| [#5571](https://github.com/scikit-image/scikit-image/pull/5571) / [#6149](https://github.com/scikit-image/scikit-image/pull/6149) | Meijering switches to `use_gaussian_derivatives=True`, which drops the $\sigma^{2}$ on that path. #6149 sets the signature default to $\alpha=-1/3$, the paper's value. |
| [#6436](https://github.com/scikit-image/scikit-image/issues/6436) (anntzer) | The paper's formula is a single-$\sigma$ $\rho=\lambda'/\max(-\lambda')$. It "does not discuss whether maximizing over values obtained for various $\sigma$ makes sense, or whether they should be normalized before comparison." |
| [#6446](https://github.com/scikit-image/scikit-image/pull/6446) (2022) | Rewrite. Circulant $\lambda'$. Replaces #6149's $\alpha=-1/3$ with $\alpha=\mathrm{None}\to+1/(\mathrm{ndim}+1)$, while the docstring keeps $-1/(\mathrm{ndim}+1)$. Drops the $\sigma^{2}$ for Meijering. Keeps per-scale `/max` then `maximum`. Also changes `frangi`'s default $\gamma$ to `s.max()/2`. |
| [#7403](https://github.com/scikit-image/scikit-image/issues/7403) / [#7711](https://github.com/scikit-image/scikit-image/issues/7711) | Frangi output and scale-selection regressions: the same "smallest $\sigma$ wins" failure when a scale correction is missing. |
| today | The same multiscale `/max` structure. Sato still has $\sigma^{2}$; Meijering does not. |

So the library **extended** a single-scale, globally normalised detector into a
Frangi-like multiscale wrapper, and kept a global `/max` that the paper uses
for a different reason (building $\rho\in[0,1]$ at one $\sigma$). The $\alpha$
sign error is older than the rewrite; see `meijering_alpha.md`.

**Pros of the current multiscale `/max`.** Easy gallery demos; every scale can
contribute even when absolute curvature is small; output lies in $[0,1]$.

**Cons.** Non-local (§4); weak as a width estimator (§3); not what NeuronJ or
the paper's Table 1 pipeline do; disagrees with Sato's scale-normalised fusion
in the same module.

The same question is now answered in `frangi` by the `frangi-fixes` branch: it
restores $\sigma^{2}$ on the Hessian norm and hoists the contrast threshold out
of the scale loop. `on_frangi.md` measures that work. `meijering` divides each
scale by its own maximum, which cancels any per-scale positive factor, so a
$\sigma$ power alone would be inert there while that `/max` stays
(`frangi_refactor_plan.md` §6).

+++

## 6. Other implementations

```{code-cell} ipython3
# NeuronJ formula vs paper selected |λ| on the synthetic ridge.
lam = selected_lambda(hessian_eigvals(ridge, sigma), -1 / 3)
paper_mag = np.where(lam < 0, -lam, 0.0)
nj_mag = neuronj_magnitude(ridge, sigma, bright=True)

show_table(pd.DataFrame([{
    "comparator": "NeuronJ Costs.run (α=−1/3 closed form)",
    "metric": "vs paper |λ| where λ<0",
    "value": f"{np.max(np.abs(nj_mag - paper_mag)):.2e}",
    "notes": "single scale; then min–max to 8-bit cost",
}, {
    "comparator": "skimage meijering(sigmas=[σ], α=−1/3)",
    "metric": "vs paper ρ",
    "value": f"{np.max(np.abs(meijering(ridge, sigmas=[sigma], alpha=-1/3, black_ridges=False, mode='nearest') - paper_rho(ridge, sigma))):.2e}",
    "notes": "matches ρ at one scale",
}, {
    "comparator": "skimage meijering default α, one σ",
    "metric": "vs paper ρ",
    "value": f"{np.max(np.abs(meijering(ridge, sigmas=[sigma], black_ridges=False, mode='nearest') - paper_rho(ridge, sigma))):.2e}",
    "notes": "same on this lone ridge after /max; see blob ratio above",
}]))
```

| software | Meijering neuriteness? | scales | notes |
| --- | --- | --- | --- |
| **NeuronJ** (authors) | Yes | Single $\sigma$ | Closed-form $\alpha=-1/3$; global display normalisation; reference implementation of the paper |
| **FeatureJ** | Derivatives / Hessian, not a named neuriteness | — | Same author; building blocks, not $\rho$ |
| **MATLAB `meij_hessianeigs`** ([amatov](https://github.com/amatov/NeurodegenerationMitochondriaLysosomes)) | Yes, the modified Hessian only | Single $\sigma$ in the snippet | $\alpha=-1/3$ on the modified Hessian; Gaussian-derivative kernels times $\sigma^{2}$; the caller fuses |
| **Fiji Tubeness / SNT** | No (Sato) | Single or multi | Different eigenvalue formula; $\sigma^{2}$-style tubeness |
| **ITK / SimpleITK** | No named Meijering | Hessian + Frangi objectness | No drop-in $\rho$ |
| **DIPlib** | No (Frangi vesselness) | Single scale | $\sigma^{2}$ plus a supremum is a caller recipe |
| **OpenCV** | No | — | — |
| **scikit-image** | Yes | Multi by default | Sections 2–5 |

There is no second widely used open-source “Meijering filter” that already chose
a multiscale policy. The fair comparator for **algorithm fidelity** is NeuronJ
at one $\sigma$, not Tubeness or Frangi.

### 6.1 Sibling filters and their scale policy

No sibling filter has Meijering neuriteness, but several fuse a Hessian response
over scale. Their normalisation is the comparable choice.

| filter | per-scale image statistic | local $\sigma$ power | width map |
| --- | --- | --- | --- |
| ITK objectness (multiscale) | no | $\sigma^{2}$ (`NormalizeAcrossScale`) | yes (`GenerateScalesOutput`) |
| DIPlib `FrangiVesselness` | no | $\sigma^{2}$ (documented caller recipe) | no (single scale) |
| Jerman (reference MATLAB) | `tau * max(lambda3)` | $\sigma^{2}$ on the Hessian | no |
| skimage `sato` | no | $\sigma^{2}$ | no |
| skimage `frangi` (`frangi-fixes`) | no | $\sigma^{2}$ on the Hessian norm | no |
| skimage `meijering` (today) | divide by each scale's max | none | no |

The rows are read from source or documentation, except `sato`, which is run in
§4; `on_frangi.md` §9 runs ITK and DIPlib directly. The pattern: every sibling
uses a local $\sigma$ power, and only `meijering` and Jerman divide by a
per-scale image statistic. ITK is the one that returns a width map, and it
returns it as a separate output rather than by retuning the detection exponent.

```{code-cell} ipython3
# Default skimage multiscale vs paper single-σ=2 on camera crop.
crop = photo[40:140, 40:140]
fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
for ax, img, title in (
    (axes[0], crop, "image"),
    (axes[1], paper_rho(crop, 2.0), r"paper $\rho$, $\sigma=2$"),
    (axes[2], meijering(crop, black_ridges=False, mode="nearest"),
     "meijering default sigmas"),
):
    bare(ax, title).imshow(img, cmap=SEQ)
fig.suptitle("same module name; different scale policy", y=1.02)
fig.tight_layout()
```

## 7. Summary

| question | answer |
| --- | --- |
| Does skimage match the paper? | **At one $\sigma$ with $\alpha=-1/3$ and bright-ridge polarity, yes** ($\rho$ agrees to numerical noise, §2). Defaults do not: wrong $\alpha$ sign, multiscale `/max` fusion. |
| Was the paper single-scale? | **Yes.** Table 1 uses $\sigma=2.0$ and the text says the detector is "tuned to neurites of specific width"; NeuronJ exposes one scale (§1). |
| Scaling across $\sigma$ | The paper defines none. A local $\sigma$ power calibrates the winning scale: $\gamma=1$ ($\sigma^2$) gives a flat detection value at $w\sqrt2$; $\gamma=3/4$ gives the true width but weakens wide ridges by $w^{-1/2}$ (§3.1). |
| Detection vs width | One scale scan gives two maps: the detection map $\max_\sigma$ and the width map $\arg\max_\sigma$. `meijering` returns a detection map only. Detecting wide structures favours $\gamma=1$; reading width favours $\gamma=3/4$ (§3.2). |
| Global vs local | Paper and skimage both use **global** normalisers, and both are non-local (§4). Local is a different product. Multiscale **per-$\sigma$ `/max`** is the controversial extension; it also cancels any $\sigma$ power. |
| History | The library extended a single-$\sigma$ detector into a Frangi-like multiscale wrapper. The per-scale `/max` is not the paper's, and #6446 dropped the $\sigma^2$ (§5). |
| Best external checks | **NeuronJ** eigenvalue step ≡ paper $\alpha=-1/3$. The sibling filters use a local $\sigma^2$, and none but Jerman uses a per-scale image statistic (§6). |

**Ways forward.** For a detection map, replace the per-scale `/max` with a local
$\sigma^2$ and take the maximum, matching `sato`, the repaired `frangi`, ITK and
DIPlib. For a width map, return the winning scale as a second output (ITK does),
calibrated at $\gamma=3/4$, or from the $\gamma=1$ map divided by $\sqrt2$. A
per-scale image statistic and a flat detection map cannot both hold.
`on_frangi.md` §12 and `frangi_refactor_plan.md` carry the sibling decision.

**Limits.** Synthetic Gaussian ridges and one camera crop; Hessian is the
current skimage Gaussian-derivative path (`use_gaussian_derivatives=True`).
No bit-exact match to NeuronJ's `Differentiator` kernels. ITK, DIPlib and
OpenCV have no Meijering $\rho$ to compare, so §6 compares their scale policy
rather than their output. $\alpha$ mechanics live in `meijering_alpha.md`; the
$\gamma$ calibration follows Lindeberg (1998) §5.6.1.
