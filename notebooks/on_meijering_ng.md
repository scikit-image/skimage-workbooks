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
notebook measures those differences, and what they mean for **detection**
versus **width**.

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
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import linalg

import skimage as ski
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import meijering, sato, frangi

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
scale-space comparison in which the **winning** $\sigma$ tracks width. Lindeberg
normalisation of derivatives (for a second-order operator, a factor
$\sigma^{2}$ on the Hessian) is the usual tool. Sato keeps an explicit
$\sigma^{2}$ in scikit-image; Frangi’s paper does too. Meijering *et al.* never
fuse scales, so they never face that choice.

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

| stage | behaviour |
| --- | --- |
| PR [#3515](https://github.com/scikit-image/scikit-image/pull/3515) (2018) | First `meijering`. Multiscale by default (`sigmas=range(1,10,2)`). Default $\alpha=+1/\mathrm{ndim}$. Hessian eigenvalues scaled by $\sigma^{2}$ in a shared helper. Per-scale `abs(aux)/abs(max(aux))`, then `max` over scales. |
| PR [#6446](https://github.com/scikit-image/scikit-image/pull/6446) (2022) | Rewrite. Circulant $\lambda'$. Docstring says optimal $\alpha=-1/(\mathrm{ndim}+1)$; code sets $+1/(\mathrm{ndim}+1)$. Drops the shared $\sigma^{2}$ on the Hessian for Meijering. Keeps per-scale `/max` then `maximum`. |
| Today | Same structure. Sato still has $\sigma^{2}$; Meijering does not. |

So the library **extended** a single-scale, globally normalised detector into a
Frangi-like multiscale wrapper, and kept a global `/max` that the paper uses
for a different reason (building $\rho\in[0,1]$ at one $\sigma$). The $\alpha$
sign error is older than the rewrite; see `meijering_alpha.md`.

**Pros of the current multiscale `/max`.** Easy gallery demos; every scale can
contribute even when absolute curvature is small; output lies in $[0,1]$.

**Cons.** Non-local (Section 4); weak as a width estimator (Section 3); not
what NeuronJ or the paper’s Table 1 pipeline do; disagrees with Sato’s
scale-normalised fusion in the same module.

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
| **Fiji Tubeness / SNT** | No (Sato) | Single or multi | Different eigenvalue formula; $\sigma^{2}$-style tubeness |
| **ITK / SimpleITK** | No named Meijering | Hessian + Frangi objectness | No drop-in $\rho$ |
| **OpenCV** | No | — | — |
| **scikit-image** | Yes | Multi by default | Sections 2–5 |

There is no second widely used open-source “Meijering filter” that already chose
a multiscale policy. The fair comparator for **algorithm fidelity** is NeuronJ
at one $\sigma$, not Tubeness or Frangi.

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
| Does skimage match the paper? | **At one $\sigma$ with $\alpha=-1/3$ and bright-ridge polarity, yes** ($\rho$ agrees to numerical noise). Defaults do not: wrong $\alpha$ sign, multiscale `/max` fusion. |
| Was the paper single-scale? | **Yes.** Table 1 uses $\sigma=2.0$; NeuronJ exposes one scale. |
| Detection vs width | Paper $\rho$ is for detection/cost at fixed thickness. Width needs scale-normalised scores ($\sigma^{2}$), which Meijering never defined for fusion. |
| Global vs local | Paper and skimage both use **global** normalisers. Local is a different product. Multiscale **per-σ `/max`** is the controversial extension. |
| Best external check | **NeuronJ** eigenvalue step ≡ paper $\alpha=-1/3$. |

**Limits.** Synthetic Gaussian ridges and one camera crop; Hessian is the
current skimage Gaussian-derivative path (`use_gaussian_derivatives=True`).
No bit-exact match to NeuronJ’s `Differentiator` kernels. ITK/OpenCV have no
Meijering $\rho$ to compare. $\alpha$ mechanics live in `meijering_alpha.md`.
