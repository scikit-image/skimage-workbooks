---
title: 'Shape Index Under Gaussian Derivatives'
date: 2026-11-14
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

:::{changelog}
2026-11-25 | claude-code:claude-opus-5-5 | Add the algorithm in 1-D and its extension to 2-D; add an RMS test of the whole Hessian; split the blob by shape, not by storage; add figures of the algorithm; audit claims, fix cell headings, add tutorial links
2026-11-21 | cursor:composer | Replace assisted-by with changelog
2026-11-17 | claude-code:claude-opus-5-5 | Restrict the blob comparison to the blob; add umbilic fix, far-field proof and published-formula checks; check literature against full texts
2026-11-17 | claude-code:claude-fable-5-1 | Add literature search section
2026-11-16 | claude-code:claude-opus-5 | Create workbook
2026-11-14 | claude-code:claude-opus-5 | Create workbook
:::

[`skimage.feature.shape_index`](https://scikit-image.org/docs/stable/api/skimage.feature.html)
smooths the image and then takes
[finite differences](https://en.wikipedia.org/wiki/Finite_difference) to make
its [Hessian](https://en.wikipedia.org/wiki/Hessian_matrix). The
`change-hessian-default` branch gives `hessian_matrix` moment-corrected
Gaussian derivative kernels, but keeps `shape_index` on finite differences,
because the shape index moves. This
notebook asks which of the two shape indices is closer to the truth, and what a
change would cost.

Inside a Gaussian blob, the Gaussian-derivative shape index is closer to the
closed form at every `sigma` tested. Finite differences are closer only in the
far field of the blob, where its Hessian is close to rank one. Both routes share
a defect in the formula: where the two eigenvalues are exactly equal, the shape
index goes to the wrong end of the scale.

Subscripts $i$ and $j$ name the image axes in skimage's `rc` order: axis $i$ is
the first array axis, down the rows, and axis $j$ the second, across the
columns.

+++

## Setup

```{code-cell} ipython3
from itertools import combinations_with_replacement
from math import atan, pi
import time

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.stats import spearmanr
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from nbhelper import show_table
```

```{code-cell} ipython3
# The subject under test.
import skimage as ski
from skimage.feature import shape_index, hessian_matrix_eigvals
```

```{code-cell} ipython3
# Comparators.
import cv2
import PIL.ImageFilter
```

```{code-cell} ipython3
# Slots 1 and 2 of the reference categorical palette; same palette as
# `hessian_review.md`.  Validated all-pairs: CVD dE 24.7 (protan), 32.7
# (tritan), normal-vision 33.6, all checks pass on the light surface.
C_ONE, C_TWO = "#2a78d6", "#eb6834"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#dedcd5"

# Diverging: two hues with a neutral midpoint.  The shape index is signed and
# its zero means something (a saddle), so the midpoint must be neutral.
DIV = LinearSegmentedColormap.from_list("div", [C_ONE, "#f2f1ec", C_TWO])

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


def interior(sigma, ndim=2, truncate=8):
    """The pixels no kernel tap reached over the border."""
    reach = int(truncate * sigma + 0.5) + 1
    return (slice(reach, -reach),) * ndim
```

A Gaussian blob of width $w$, smoothed by a Gaussian of scale $\sigma$, is a
Gaussian of variance $w^2 + \sigma^2$. Its Hessian, and so its shape index, has
a closed form. That gives a standard that neither route was built against.

```{code-cell} ipython3
def blob_truth(size, width, sigma):
    """A Gaussian blob, and the exact Hessian of its smoothing."""
    r, c = [x - size // 2 for x in np.indices((size, size), dtype=float)]
    image = np.exp(-(r**2 + c**2) / (2 * width**2))
    spread = width**2 + sigma**2
    amp = (width**2 / spread) * np.exp(-(r**2 + c**2) / (2 * spread))
    H = [amp * (r**2 - spread) / spread**2,
         amp * (r * c) / spread**2,
         amp * (c**2 - spread) / spread**2]
    return image, H


BLOB_SIZE, BLOB_WIDTH = 201, 6.0
```

`zones` splits the interior in two at a level `FLOOR` of the smoothed peak: the
**blob** above it, and the **far field** below it. §7.1 explains the split.

```{code-cell} ipython3
FLOOR = 1e-3


def zones(size, width, sigma, floor=FLOOR):
    """Blob and far-field masks over the interior, split at `floor` of the peak."""
    r2 = sum((x - size // 2) ** 2 for x in np.indices((size, size), dtype=float))
    level = np.exp(-r2 / (2 * (width**2 + sigma**2)))   # smoothed blob / peak
    inside = np.zeros((size, size), bool)
    inside[interior(sigma)] = True
    return inside & (level >= floor), inside & (level < floor)


def radius(size):
    """Distance of each pixel from the blob centre."""
    return np.hypot(*[x - size // 2 for x in np.indices((size, size), dtype=float)])
```

## 1. The algorithm in one dimension

A 1-D signal $g(x)$ is the special case of an image that does not change down
its rows, $f(r, c) = g(c)$. Its Hessian has one element that is not zero,
$H_{jj} = g''$. So its two eigenvalues are $g''$ and $0$, and the shape index of
the pair is

$$
s = \begin{cases}
+1/2 \ \text{(ridge)} & g'' < 0, \\
-1/2 \ \text{(rut)} & g'' > 0, \\
\text{NaN (planar)} & g'' = 0.
\end{cases}
$$

In 1-D the shape index keeps only the sign of the second derivative. The
algorithm has the same three steps as in 2-D: smooth the signal at scale
$\sigma$, take the second derivative, and map it to $s$.

A sum of Gaussian bumps gives each step in closed form. A bump of width $w$,
smoothed at scale $\sigma$, is a Gaussian of variance $v = w^2 + \sigma^2$, with
its height scaled by $w / \sqrt{v}$. About its centre, its second derivative is
$g\,(x^2 - v)/v^2$, which is zero at $x = \pm\sqrt{v}$.

`s_pair` maps an eigenvalue pair to $s$. It uses the form of the formula that §5
shows to be correct where the two eigenvalues tie.

```{code-cell} ipython3
def s_pair(l1, l2):
    """The shape index of an eigenvalue pair l1 >= l2, in the form of §5."""
    s = -(2 / np.pi) * np.arctan2(l1 + l2, l1 - l2)
    return np.where((l1 == 0) & (l2 == 0), np.nan, s)


def bumps_1d(x, bumps, sigma):
    """A sum of Gaussian bumps, smoothed at sigma, and its exact second derivative."""
    g, g2 = np.zeros_like(x), np.zeros_like(x)
    for centre, width, height in bumps:
        v = width**2 + sigma**2
        bump = height * width / np.sqrt(v) * np.exp(-(x - centre)**2 / (2 * v))
        g += bump
        g2 += bump * ((x - centre)**2 - v) / v**2
    return g, g2


def s_1d(g2):
    """Shape index of the image f(r, c) = g(c), whose eigenvalues are g'' and 0."""
    return s_pair(np.maximum(g2, 0), np.minimum(g2, 0))


# (centre, width, height) of each bump: one wide, two narrow and close, one dark.
BUMPS = ((-30, 6, 1.0), (4, 2, 0.8), (12, 2, 0.8), (40, 4, -0.7))
X_FINE = np.linspace(-60, 60, 2401)
SIGMA_1D = 1.0
```

The 2-D eigenvalue routine, on the Hessian $[0, 0, g'']$ of the image
$f(r, c) = g(c)$, gives the same $s$:

```{code-cell} ipython3
_, g2_fine = bumps_1d(X_FINE, BUMPS, SIGMA_1D)
zero = np.zeros((1, X_FINE.size))
s_2d_route = s_pair(*hessian_matrix_eigvals([zero, zero, g2_fine[None]]))[0]
show_table(pd.DataFrame([{
    "values of s": ", ".join(f"{v:+.2f}" for v in np.unique(s_1d(g2_fine))),
    "max |s from 2-D eigenvalues - s_1d|":
        f"{np.max(np.abs(s_2d_route - s_1d(g2_fine))):.1e}"}]),
    index="values of s")
```

The three steps on the four bumps. The bottom strip is the shape index, with the
colours of §3 and §4: orange for a ridge, blue for a rut. The vertical lines are
the zero crossings of $g''$:

```{code-cell} ipython3
def crossings(x, values):
    """Where `values` changes sign, by linear interpolation along x."""
    k = np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))
    return x[k] - values[k] * (x[k + 1] - x[k]) / (values[k + 1] - values[k])


raw, _ = bumps_1d(X_FINE, BUMPS, 0)
smooth, g2_fine = bumps_1d(X_FINE, BUMPS, SIGMA_1D)
s_fine = s_1d(g2_fine)
cuts = crossings(X_FINE, g2_fine)

fig, axes = plt.subplots(3, 1, figsize=(8.4, 4.6), sharex=True,
                         gridspec_kw=dict(height_ratios=[2, 2, 0.6]))
axes[0].plot(X_FINE, raw, color=RULE, linewidth=3, label="signal $g$")
axes[0].plot(X_FINE, smooth, color=INK, linewidth=1.2,
             label=rf"smoothed at $\sigma$ = {SIGMA_1D}")
axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")
recede(axes[0], "1. smooth")
axes[1].fill_between(X_FINE, g2_fine, 0, where=g2_fine > 0, color=C_ONE, alpha=0.5,
                     linewidth=0, label="$g'' > 0$: rut")
axes[1].fill_between(X_FINE, g2_fine, 0, where=g2_fine < 0, color=C_TWO, alpha=0.5,
                     linewidth=0, label="$g'' < 0$: ridge")
axes[1].plot(X_FINE, g2_fine, color=INK, linewidth=1)
axes[1].axhline(0, color=RULE, linewidth=0.8)
axes[1].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="lower right")
recede(axes[1], "2. second derivative $g''$")
axes[2].imshow(s_fine[None], cmap=DIV, norm=TwoSlopeNorm(0, -1, 1), aspect="auto",
               extent=(X_FINE[0], X_FINE[-1], 0, 1))
axes[2].set_yticks([])
recede(axes[2], "3. shape index: ridge (+1/2) orange, rut (-1/2) blue")
for ax in axes:
    for cut in cuts:
        ax.axvline(cut, color=MUTED, linewidth=0.6, linestyle="dotted")
axes[2].set_xlabel("x (pixels)", color=MUTED)
fig.tight_layout()
```

Each bright bump is a ridge between two zero crossings, with a rut on each side.
The dark bump is a rut, with a ridge on each side. The zero crossings of the
wide bump are at $-30 \pm \sqrt{w^2 + \sigma^2}$, the crossings of the bump
alone, to four decimals. The other bumps are too far away to move them:

```{code-cell} ipython3
wide_centre, wide_width, _ = BUMPS[0]
near = cuts[np.abs(cuts - wide_centre) < 15]
alone = wide_centre + np.array([-1, 1]) * np.sqrt(wide_width**2 + SIGMA_1D**2)
show_table(pd.DataFrame(
    [{"crossing": side, "measured": f"{m:.4f}", "bump alone": f"{a:.4f}"}
     for side, m, a in zip(("left", "right"), near, alone)]), index="crossing")
```

As $\sigma$ grows, each ridge widens, because its zero crossings are at
$\pm\sqrt{w^2 + \sigma^2}$. Two bumps of width $w$ at a distance $d$ apart each
have the second derivative $g\,((d/2)^2 - v)/v^2$ at the midpoint. So the rut
between them closes when $v = (d/2)^2$. The two narrow bumps have $w = 2$ and
$d = 8$, which gives $\sigma = \sqrt{12} \approx 3.46$. The scale-space picture
shows $s$ at every $\sigma$:

```{code-cell} ipython3
narrow = [b for b in BUMPS if b[1] == 2]
gap_centre = (narrow[0][0] + narrow[1][0]) / 2


def midpoint_g2(sigma):
    """g'' at the midpoint of the two narrow bumps."""
    return bumps_1d(np.array([gap_centre]), BUMPS, sigma)[1][0]


lo, hi = 1.0, 6.0                       # rut at lo, ridge at hi: bisect
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if midpoint_g2(mid) > 0 else (lo, mid)
SIGMA_MERGE = lo
d = narrow[1][0] - narrow[0][0]
alone = np.sqrt((d / 2)**2 - 2**2)
print(f"rut closes at sigma = {SIGMA_MERGE:.6f}; two bumps alone: "
      f"sqrt((d/2)**2 - w**2) = {alone:.6f}; difference {SIGMA_MERGE - alone:+.1e}")
```

```{code-cell} ipython3
sigmas_ss = np.geomspace(0.3, 12, 240)
scale_space = np.array([s_1d(bumps_1d(X_FINE, BUMPS, s)[1]) for s in sigmas_ss])

fig, ax = plt.subplots(figsize=(8.4, 3.0))
ax.pcolormesh(X_FINE, sigmas_ss, scale_space, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1),
              shading="auto", rasterized=True)
ax.set_yscale("log")
for level, label in ((SIGMA_1D, rf"$\sigma$ = {SIGMA_1D}, the figure above"),
                     (SIGMA_MERGE, f"rut between the narrow bumps closes, "
                                   rf"$\sigma$ = {SIGMA_MERGE:.2f}")):
    ax.axhline(level, color=INK, linewidth=0.8, linestyle="dashed")
    ax.annotate(label, (-58, level * 1.07), color=INK, fontsize=7.5)
recede(ax, r"shape index over $x$ and $\sigma$: ridge orange, rut blue")
ax.set_xlabel("x (pixels)", color=MUTED)
ax.set_ylabel(r"$\sigma$", color=MUTED)
fig.tight_layout()
```

The other bumps change the closing scale by about $10^{-5}$. Above it, the two
narrow bumps are one ridge. In 1-D there is one eigenvalue, so $s$ has two
values, and $\sigma$ decides which bumps are separate ridges.

+++

## 2. From one dimension to two

In 2-D each pixel has three second derivatives: $H_{ii}$ down the rows,
$H_{jj}$ across the columns, and the mixed derivative $H_{ij}$. They are the
elements of the symmetric $2 \times 2$ Hessian. Its eigenvalues
$\lambda_1 \geq \lambda_2$ are the second derivatives along the two
[principal directions](https://en.wikipedia.org/wiki/Principal_curvature), the
directions of largest and smallest curvature. Its eigenvectors are those
directions. The 1-D signal of §1 gives the pairs $(g'', 0)$ and $(0, g'')$,
which lie on two rays of the $(\lambda_1, \lambda_2)$ plane, at $s = \mp 1/2$.
A 2-D image can put the pair anywhere in the half-plane $\lambda_1 \geq
\lambda_2$, and $s$ is its angle (§4).

A Gaussian of variance $v$ in 1-D and an isotropic Gaussian blob of the same
variance in 2-D show the step. Let $\rho$ be the distance from the centre in
units of $\sqrt{v}$. In 1-D, $g'' = (g/v)(\rho^2 - 1)$. In 2-D, the second
derivative along the radius has the same form, $(f/v)(\rho^2 - 1)$, and the
blob adds a second one across the radius, $-f/v$ (§7.1 derives both). The cell
scales both pairs so that the value at the centre is 1:

```{code-cell} ipython3
def radial_pairs(rho):
    """Eigenvalue pairs of the 1-D Gaussian and the 2-D blob, scaled by v / peak."""
    level = np.exp(-rho**2 / 2)               # g / peak, and f / peak
    along = level * (rho**2 - 1)              # second derivative along the radius
    across = -level                           # 2-D only: across the radius
    return (along, across,
            (np.maximum(along, 0), np.minimum(along, 0)),
            (np.maximum(along, across), np.minimum(along, across)))


rho_line = np.linspace(0, 4, 801)
along, across, pair_1d, pair_2d = radial_pairs(rho_line)
s_line_1d, s_line_2d = s_pair(*pair_1d), s_pair(*pair_2d)
marks = {"0": 0.0, "1": 1.0, r"$\sqrt{2}$": np.sqrt(2), "3": 3.0}

fig = plt.figure(figsize=(8.4, 3.2))
grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])
ax_l, ax_s, ax_p = (fig.add_subplot(grid[0, k]) for k in range(3))

ax_l.plot(rho_line, along, color=INK, linewidth=1.5, label="along the radius (1-D, 2-D)")
ax_l.plot(rho_line, across, color=C_ONE, linewidth=1.5, linestyle="dashed",
          label="across the radius (2-D)")
ax_l.plot(rho_line, 0 * rho_line, color=RULE, linewidth=3, zorder=0,
          label="second eigenvalue, 1-D")
ax_l.legend(frameon=False, fontsize=7, labelcolor=MUTED, loc="lower right")
recede(ax_l, "eigenvalues, scaled by v / peak")
ax_l.set_xlabel(r"$\rho$", color=MUTED)

ax_s.plot(rho_line, s_line_1d, color=INK, linewidth=2, label="1-D")
ax_s.plot(rho_line, s_line_2d, color=C_ONE, linewidth=2, label="2-D")
ax_s.axhline(0, color=RULE, linewidth=0.8, zorder=0)
ax_s.set_ylim(-1.05, 1.05)
ax_s.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper right")
recede(ax_s, "shape index $s$")
ax_s.set_xlabel(r"$\rho$", color=MUTED)

# The paths of the two pairs, over the half-plane of §4.
l1g, l2g = np.meshgrid(np.linspace(-1.1, 1.1, 301), np.linspace(-1.1, 1.1, 301))
plane_s = s_pair(l1g, l2g)
plane_s[l2g > l1g] = np.nan
ax_p.imshow(plane_s, extent=(-1.1, 1.1, -1.1, 1.1), origin="lower", cmap=DIV,
            norm=TwoSlopeNorm(0, -1, 1), alpha=0.45)
ax_p.plot([-1.1, 1.1], [-1.1, 1.1], color=MUTED, linewidth=0.8)
ax_p.plot(*pair_1d, color=INK, linewidth=2, label="1-D")
ax_p.plot(*pair_2d, color=C_ONE, linewidth=2, label="2-D")
for name, rho_k in marks.items():
    k = np.argmin(np.abs(rho_line - rho_k))
    for (p1, p2), colour in ((pair_1d, INK), (pair_2d, C_ONE)):
        ax_p.plot(p1[k], p2[k], "o", color=colour, markersize=4)
    ax_p.annotate(rf"$\rho$ = {name}", (pair_2d[0][k], pair_2d[1][k]),
                  {0: (5, 4), 3: (2, -14)}.get(rho_k, (6, -6)),
                  textcoords="offset points",
                  color=INK, fontsize=7)
ax_p.set_xlim(-1.1, 1.1); ax_p.set_ylim(-1.1, 1.1); ax_p.set_aspect("equal")
ax_p.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="upper left")
recede(ax_p, r"the pair in the $(\lambda_1, \lambda_2)$ plane")
ax_p.set_xlabel(r"$\lambda_1$", color=MUTED); ax_p.set_ylabel(r"$\lambda_2$", color=MUTED)
fig.tight_layout()
```

In 1-D the pair starts at $(0, -1)$ and runs up the $\lambda_2$ axis to the
origin at $\rho = 1$, where $g''$ changes sign. Then it goes out along the
$\lambda_1$ axis and back towards the origin. So $s$ jumps from $+1/2$ to $-1/2$. In 2-D the second
derivative across the radius keeps the pair off the origin while $f$ is large.
The pair starts on the diagonal, as an umbilic, and turns through the
half-plane. So $s$ moves without a jump, from $+1$ at the centre, through $+1/2$
at $\rho = 1$ and $0$ at $\rho = \sqrt{2}$, towards $-1/2$:

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"rho": name,
      "s, 1-D": f"{s_pair(*radial_pairs(r_)[2]):+.3f}",
      "s, 2-D": f"{s_pair(*radial_pairs(r_)[3]):+.3f}"}
     for name, r_ in (("0", 0.0), ("1", 1.0), ("sqrt(2)", np.sqrt(2)), ("4", 4.0))]),
    index="rho")
```

At $\rho = 1$ the 1-D pair is $(0, 0)$, the planar point, so $s$ is NaN.

In 1-D the second derivative is along the one axis. In 2-D the principal
directions need not lie along the image axes. Then the mixed element $H_{ij}$
carries part of the curvature, and the eigenvalues collect it again. The 1-D
signal of §1, laid down at an angle $\theta$, is the image $f = g(u)$, with
$u = \mathbf{n} \cdot \mathbf{x}$ and $\mathbf{n} = (\sin\theta, \cos\theta)$ in
$(r, c)$. Its Hessian is $g''(u)\,\mathbf{n}\mathbf{n}^T$, so $H_{ii} = g''
\sin^2\theta$, $H_{ij} = g'' \sin\theta\cos\theta$ and $H_{jj} = g''
\cos^2\theta$. At $\theta = 0$ this is the image $f(r, c) = g(c)$ of §1:

```{code-cell} ipython3
def stripes_truth(points, angle, sigma):
    """The 1-D bumps laid down at `angle`, smoothed at sigma, with exact Hessian.

    `points` has shape (2, ...), in (r, c).  Returns the image and the Hessian
    elements [H_ii, H_ij, H_jj] at the points.
    """
    n = np.array([np.sin(angle), np.cos(angle)])
    g, g2 = bumps_1d(np.tensordot(n, points, axes=1), BUMPS, sigma)
    return g, [g2 * n[0] * n[0], g2 * n[0] * n[1], g2 * n[1] * n[1]]


GRID = np.indices((121, 121), dtype=float) - 60

fig, axes = plt.subplots(2, 5, figsize=(8.4, 3.8))
for row, degrees in zip(axes, (0, 35)):
    f, H = stripes_truth(GRID, np.deg2rad(degrees), SIGMA_1D)
    bare(row[0], f"image, {degrees} degrees").imshow(f, cmap="gray")
    limit = max(np.abs(h).max() for h in H)
    for ax, h, name in zip(row[1:4], H, ("$H_{ii}$", "$H_{ij}$", "$H_{jj}$")):
        bare(ax, name).imshow(h, cmap=DIV, norm=TwoSlopeNorm(0, -limit, limit))
    bare(row[4], "shape index $s$").imshow(
        s_pair(*hessian_matrix_eigvals(H)), cmap=DIV, norm=TwoSlopeNorm(0, -1, 1))
fig.tight_layout()
```

At 0 degrees only $H_{jj}$ is not zero, and it is $g''$. At 35 degrees each
element carries a fraction of $g''$: $\sin^2\theta$ in $H_{ii}$,
$\sin\theta\cos\theta$ in $H_{ij}$, and $\cos^2\theta$ in $H_{jj}$. The
eigenvalues of $g''\,\mathbf{n}\mathbf{n}^T$ are $g''$ and $0$ at every angle,
so the shape index is the same. Rotate the points with the stripes, and compare
$s$ at matching points:

```{code-cell} ipython3
def rotation_row(degrees):
    """Share of curvature in H_ij, and the change in s, for stripes at `degrees`."""
    angle = np.deg2rad(degrees)
    turn = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    _, H0 = stripes_truth(GRID, 0, SIGMA_1D)
    _, Ht = stripes_truth(np.tensordot(turn, GRID, axes=1), angle, SIGMA_1D)
    peak = np.abs(H0[2]).max()                            # max |g''|
    return {"angle (degrees)": degrees,
            **{f"max |{name}| / max |g''|": f"{np.abs(h).max() / peak:.3f}"
               for name, h in zip(("H_ii", "H_ij", "H_jj"), Ht)},
            "max |s(rotated) - s(original)|":
                f"{np.nanmax(np.abs(s_pair(*hessian_matrix_eigvals(Ht)) - s_pair(*hessian_matrix_eigvals(H0)))):.1e}"}


show_table(pd.DataFrame([rotation_row(d) for d in (0, 20, 35, 45, 90)]),
           index="angle (degrees)")
```

$H_{ij}$ peaks at 45 degrees, at half of $g''$, and falls to zero again at 90
degrees, where the curvature is all in $H_{ii}$. The shape index does not move,
to rounding.

So the 2-D algorithm needs all three elements, and the eigenvalues, to give a
shape index that follows the shape and not the image axes. §3 asks how each
discrete route computes the three elements.

+++

## 3. The two routes

Both smooth at scale `sigma` and take second derivatives. They differ in what
does the differentiating. The Gaussian-derivative route is the standard
derivative of [scale-space theory](https://en.wikipedia.org/wiki/Scale_space);
the SciPy [`ndimage` tutorial](https://docs.scipy.org/doc/scipy/tutorial/ndimage.html)
describes the 1-D correlation that both routes use.

```{code-cell} ipython3
def fd_hessian(image, sigma, mode="constant", cval=0):
    """Smooth at sigma, then two central differences: the shipped route."""
    smoothed = ski.filters.gaussian(image, sigma=sigma, mode=mode, cval=cval)
    gradients = np.gradient(smoothed)
    return [np.gradient(gradients[a], axis=b)
            for a, b in combinations_with_replacement(range(image.ndim), 2)]
```

The Gaussian-derivative route convolves once per element with a kernel whose
discrete moments have been restored. Sampling the derivative and using the taps
as they fall leaves a kernel that responds to a constant image and under-reports
curvature. Imposing $\sum_x k(x) = 0$ and $\tfrac12 \sum_x k(x)\, x^2 = 1$ repairs
both.
The `taps` function is a transcription of `_gaussian_derivative_kernel` on the
`change-hessian-default` branch, at commit `d43497d71`. The two agree to the
last bit for `sigma` from 0.1 to 2 and all three orders. That check ran outside
this notebook, because the installed scikit-image does not have the branch.

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
    # Below sigma of about 0.2 the line above cancels catastrophically, so the
    # residue comes out of the centre tap, before and after the x**2 scaling.
    center = len(kernel) // 2
    kernel[center] = -(kernel[:center].sum() + kernel[center + 1:].sum())
    kernel /= (kernel * x**2).sum() / 2
    kernel[center] = -(kernel[:center].sum() + kernel[center + 1:].sum())
    return kernel


def gd_hessian(image, sigma, mode="constant", cval=0, truncate=8):
    """One separable pass per element, with the corrected kernels."""
    image = np.asarray(image, dtype=float)
    ndim = image.ndim
    kernels = {(axis, o): taps(sigma, o, truncate)
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


ROUTES = (("finite differences", fd_hessian), ("Gaussian derivatives", gd_hessian))
SHORT = {"finite differences": "FD", "Gaussian derivatives": "GD"}
```

Along each axis, each route is a 1-D filter. Its impulse response is the kernel
that it applies. The finite-difference route smooths with a sampled Gaussian and
then applies the central difference $(\tfrac12, 0, -\tfrac12)$ once for a first
derivative and twice for a second. The Gaussian-derivative route applies one
corrected kernel. The grey line is the continuous Gaussian derivative that both
approximate:

```{code-cell} ipython3
def impulse_response(route_1d, sigma, order, size=41):
    """1-D impulse response of one route's derivative of `order`."""
    impulse = np.zeros(size)
    impulse[size // 2] = 1
    return route_1d(impulse, sigma, order)


def fd_1d(signal, sigma, order):
    """The finite-difference route in 1-D: smooth, then `order` central differences."""
    out = ski.filters.gaussian(signal, sigma=sigma, mode="constant")
    for _ in range(order):
        out = np.gradient(out)
    return out


def gd_1d(signal, sigma, order):
    """The Gaussian-derivative route in 1-D: one pass of the corrected kernel."""
    return ndi.correlate1d(signal, taps(sigma, order), mode="constant")


def gauss_derivative(x, sigma, order):
    """The continuous Gaussian derivative that both routes approximate."""
    g = np.exp(-x**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
    return {1: -x / sigma**2 * g, 2: (x**2 - sigma**2) / sigma**4 * g}[order]


fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.2))
x_fine = np.linspace(-6, 6, 400)
for row, order in enumerate((1, 2)):
    for ax, sigma in zip(axes[row], (0.5, 1.0, 2.0)):
        n = np.arange(41) - 20
        ax.plot(x_fine, gauss_derivative(x_fine, sigma, order), color=RULE,
                linewidth=2, label="continuous", zorder=0)
        for route_1d, colour, label, marker, shift in (
                (fd_1d, C_TWO, "finite differences", "o", -0.12),
                (gd_1d, C_ONE, "Gaussian derivatives", "s", 0.12)):
            h = impulse_response(route_1d, sigma, order)
            ax.vlines(n + shift, 0, h, color=colour, linewidth=1.5)
            ax.plot(n + shift, h, marker, color=colour, markersize=3.5, label=label)
        ax.axhline(0, color=RULE, linewidth=0.8, zorder=0)
        ax.set_xlim(-6.5, 6.5)
        recede(ax, f"order {order}, sigma = {sigma}")
axes[1, 1].set_xlabel("tap offset (pixels)", color=MUTED)
axes[0, 0].legend(frameon=False, fontsize=7, labelcolor=MUTED, loc="upper left")
fig.tight_layout()
```

At `sigma` 2 the two routes are close to each other and to the continuous
derivative. At `sigma` 0.5 they are not. The Gaussian-derivative second
derivative approaches $(1, -2, 1)$. The finite-difference one approaches the
doubled central difference, $(\tfrac14, 0, -\tfrac12, 0, \tfrac14)$, which is
twice as wide.

The shape index is the same arithmetic on either Hessian. With the eigenvalues
ordered $\lambda_1 \geq \lambda_2$, the library computes

$$
s = \frac{2}{\pi}\arctan\frac{\lambda_2 + \lambda_1}{\lambda_2 - \lambda_1}.
$$

```{code-cell} ipython3
def s_library(H):
    """The shape index exactly as `skimage.feature.shape_index` computes it."""
    l1, l2 = hessian_matrix_eigvals(H)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (2.0 / np.pi) * np.arctan((l2 + l1) / (l2 - l1))


PHOTO = ski.util.img_as_float(ski.data.camera())[::2, ::2]
```

The library docstring sorts $s$ into nine named classes, from spherical cup at
$-1$ to spherical cap at $+1$:

```{code-cell} ipython3
BINS = np.array([-7/8, -5/8, -3/8, -1/8, 1/8, 3/8, 5/8, 7/8])
CLASS_NAMES = ["spherical cup", "trough", "rut", "saddle rut", "saddle",
               "saddle ridge", "ridge", "dome", "spherical cap"]
```

The whole algorithm, on a crop of the photograph: three Hessian elements per
pixel, their two eigenvalues, the shape index, and its class. Across a bright
line, $\lambda_2$ is strongly negative and $\lambda_1$ is near 0, so $s$ is near
$+1/2$, a ridge:

```{code-cell} ipython3
# The algorithm, step by step, on a crop of the photograph.
CROP = (slice(20, 100), slice(85, 165))
PIPE_SIGMA = 2.0
H_crop = gd_hessian(PHOTO, PIPE_SIGMA)
l1_crop, l2_crop = hessian_matrix_eigvals(H_crop)
s_crop = s_library(H_crop)          # no exact ties on the photograph (§5)


def signed(ax, data, title):
    """Show signed data on the diverging map, symmetric about zero."""
    limit = np.abs(data).max()
    return bare(ax, title).imshow(data, cmap=DIV, norm=TwoSlopeNorm(0, -limit, limit))


fig, axes = plt.subplots(2, 4, figsize=(8.4, 4.6))
bare(axes[0, 0], "image").imshow(PHOTO[CROP], cmap="gray")
for ax, h, name in zip(axes[0, 1:], H_crop, ("$H_{ii}$", "$H_{ij}$", "$H_{jj}$")):
    signed(ax, h[CROP], name)
limit = max(np.abs(l1_crop[CROP]).max(), np.abs(l2_crop[CROP]).max())
for ax, lam, name in ((axes[1, 0], l1_crop, r"$\lambda_1$ (larger)"),
                      (axes[1, 1], l2_crop, r"$\lambda_2$ (smaller)")):
    bare(ax, name).imshow(lam[CROP], cmap=DIV, norm=TwoSlopeNorm(0, -limit, limit))
bare(axes[1, 2], "shape index $s$").imshow(s_crop[CROP], cmap=DIV,
                                         norm=TwoSlopeNorm(0, -1, 1))
classes = bare(axes[1, 3], "class").imshow(
    np.digitize(s_crop[CROP], BINS), cmap=DIV.resampled(9), vmin=-0.5, vmax=8.5)
bar = fig.colorbar(classes, ax=axes[1, 3], ticks=range(9), fraction=0.046)
bar.ax.set_yticklabels(CLASS_NAMES, fontsize=7, color=MUTED)
bar.outline.set_visible(False)
fig.suptitle(f"Gaussian-derivative route at sigma = {PIPE_SIGMA}: "
             "negative blue, positive orange, zero white", color=MUTED, fontsize=9)
fig.tight_layout()
```

The shape index uses the ratio of the two eigenvalues at each pixel, not their
size (§4). So faint texture, which is near white in the eigenvalue panels, gets
a class that is as clear as that of a strong edge.

The finite-difference transcription must be the library's:

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"sigma": sigma,
      "transcription vs skimage.feature.shape_index":
          "bit-identical" if np.array_equal(
              np.nan_to_num(shape_index(PHOTO, sigma=sigma), nan=-9),
              np.nan_to_num(s_library(fd_hessian(PHOTO, sigma)), nan=-9))
          else "DIFFERS"}
     for sigma in (0.5, 1.0, 2.0, 4.0)]), index="sigma")
```

The two moment conditions pin the Gaussian-derivative second-derivative
kernel:

```{code-cell} ipython3
def moments(sigma):
    """The two conditions that pin the second-derivative kernel."""
    k = taps(sigma, 2)
    radius = len(k) // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    return {"sigma": sigma, "taps": len(k),
            "sum(k), want 0": f"{k.sum():.1e}",
            "sum(k * x**2) / 2, want 1": f"{(k * x**2).sum() / 2:.12f}"}


show_table(pd.DataFrame([moments(s) for s in (0.3, 0.5, 1.0, 2.0)]), index="sigma")
```

## 4. The shape index only sees a ratio

The shape index is a function of the quotient of the two eigenvalues. So
multiplying the whole Hessian by any positive constant leaves it alone.

```{code-cell} ipython3
rng = np.random.default_rng(0)
random_H = [rng.normal(size=(64, 64)) for _ in range(3)]
baseline = s_library(random_H)

show_table(pd.DataFrame(
    [{"H scaled by": f"{c:g}",
      "max |change in s|":
          f"{np.nanmax(np.abs(s_library([c * h for h in random_H]) - baseline)):.1e}"}
     for c in (0.5, 2.0, 1e3, 1e-3)]), index="H scaled by")
```

That is the design. Koenderink and van Doorn build the shape index from a polar
coordinate system in the plane of the
[principal curvatures](https://en.wikipedia.org/wiki/Principal_curvature)
$(\kappa_1, \kappa_2)$, in which "the direction
encodes the shape, whereas the distance from the origin encodes the size"
[@koenderink1992surface_shape]. Their definition, with $\kappa_1 \geq
\kappa_2$, is the formula above. They also state that $s$ is "directly
proportional to the angle with the $\kappa_1 + \kappa_2 = 0$ axis". So an
error in that angle is an error in $s$, scaled by a constant:

```{code-cell} ipython3
rough = [a + 0.01 * rng.normal(size=a.shape) for a in random_H]
pair_a, pair_b = hessian_matrix_eigvals(random_H), hessian_matrix_eigvals(rough)
turn = np.arctan2(pair_b[1], pair_b[0]) - np.arctan2(pair_a[1], pair_a[0])
turn = np.abs(np.arctan2(np.sin(turn), np.cos(turn)))
moved = np.abs(s_library(rough) - s_library(random_H))
usable = np.isfinite(moved) & np.isfinite(turn) & (moved < 1.9)

show_table(pd.DataFrame([{
    "pixels compared": int(usable.sum()),
    "max |change in s - (2/pi) * change in angle|":
        f"{np.nanmax(np.abs(moved[usable] - (2 / np.pi) * turn[usable])):.1e}"}]),
    index="pixels compared")
```

The angle of the eigenvalue pair is the shape index, to a factor of $2/\pi$.
§8 uses the angle to find which Hessian element carries the error.

The companion measure in the same paper is the curvedness,
$c = \sqrt{(\kappa_1^2 + \kappa_2^2)/2}$, which is the distance from the
origin. The paper says that the shape index alone fixes the local shape "up to
a scaling factor, or similarity". The moment correction sets the gain of each
Hessian element on a quadratic to 1. A gain error common to all three elements
changes $c$ and leaves $s$ alone.

The figure shows $s$ over the half-plane $\lambda_1 \geq \lambda_2$. Each ray
from the origin has one value of $s$; the white rays are the class edges. The
diagonal $\lambda_1 = \lambda_2$ is the umbilic, with its two halves at the two
ends of the scale. On the right is one local intensity patch,
$(\lambda_1 u^2 + \lambda_2 v^2)/2$, at the centre of each class:

```{code-cell} ipython3
def pair_for(s):
    """Unit eigenvalue pair (l1 >= l2) with shape index `s`."""
    beta = -np.pi * np.asarray(s) / 2
    return (np.cos(beta) + np.sin(beta)) / 2, (np.sin(beta) - np.cos(beta)) / 2


fig = plt.figure(figsize=(8.4, 3.9))
grid = fig.add_gridspec(3, 6, width_ratios=[3.2, 1, 1, 1, 0.05, 0.05])
plane = fig.add_subplot(grid[:, 0])

# The half-plane l1 >= l2, coloured by the shape index.
l1g, l2g = np.meshgrid(np.linspace(-1, 1, 401), np.linspace(-1, 1, 401))
s_grid = -(2 / np.pi) * np.arctan2(l1g + l2g, l1g - l2g)   # the form of §5
s_grid[l2g > l1g] = np.nan
plane.imshow(s_grid, extent=(-1, 1, -1, 1), origin="lower", cmap=DIV,
             norm=TwoSlopeNorm(0, -1, 1))
for edge in BINS:
    a, b = pair_for(edge)
    plane.plot([0, 1.9 * a], [0, 1.9 * b], color="white", linewidth=0.8)
plane.plot([-1, 1], [-1, 1], color=INK, linewidth=1.2)
plane.plot(0, 0, "o", color=INK, markersize=4)
plane.annotate("planar point:\nNaN", (0, 0), (0.05, 0.55), color=INK, fontsize=7.5,
               arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
plane.annotate("umbilic, s = +1\n(spherical cap)", (-0.7, -0.7), (-0.95, -0.25),
               color=INK, fontsize=7.5,
               arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
plane.annotate("umbilic, s = -1\n(spherical cup)", (0.7, 0.7), (0.2, 0.86),
               color=INK, fontsize=7.5,
               arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
plane.set_xlim(-1, 1); plane.set_ylim(-1, 1); plane.set_aspect("equal")
recede(plane, r"$s$ over the $(\lambda_1, \lambda_2)$ plane, $\lambda_1 \geq \lambda_2$")
plane.set_xlabel(r"$\lambda_1$", color=MUTED); plane.set_ylabel(r"$\lambda_2$", color=MUTED)

# One local intensity patch per class, at the class centre.
u = np.linspace(-1, 1, 41)
U, W = np.meshgrid(u, u)
for k, (name, centre) in enumerate(zip(CLASS_NAMES, np.linspace(-1, 1, 9))):
    a, b = pair_for(centre)
    ax = fig.add_subplot(grid[k // 3, 1 + k % 3])
    patch = (a * U**2 + b * W**2) / 2
    bare(ax, f"{name}\ns = {centre:+.2f}").imshow(patch, cmap="gray")
    ax.contour(patch, levels=np.linspace(patch.min(), patch.max(), 9)[1:-1],
               colors=[MUTED], linewidths=0.5)
    ax.title.set_fontsize(7)
fig.tight_layout()
```

A bright spot is a spherical cap, and a dark spot is a spherical cup. A bright
line is a ridge, and a dark line is a rut.

+++

## 5. The umbilic

Koenderink and van Doorn separate two degenerate cases
[@koenderink1992surface_shape]. At a **planar** point, $\kappa_1 = \kappa_2 =
0$, the shape "is really indeterminate", and NaN is the right answer. At an
[**umbilic**](https://en.wikipedia.org/wiki/Umbilical_point), $\kappa_1 = \kappa_2 \neq 0$, the surface looks locally like the
outside ($s = +1$) or the inside ($s = -1$) of a sphere, and these are "true
endpoints of the shape scale". Their own example is a Gaussian hill: its top is
an umbilic, and it is convex, so $s = +1$.

The formula cannot reach that value reliably. `hessian_matrix_eigvals` returns
$\lambda_1 \geq \lambda_2$, so $\lambda_2 - \lambda_1 \leq 0$. Where the
difference is negative, the formula is correct. Where the two eigenvalues are
exactly equal, IEEE 754 subtraction gives $+0$, never
[$-0$](https://en.wikipedia.org/wiki/Signed_zero). The quotient then
takes the sign of $\lambda_1 + \lambda_2$, which is the wrong sign. A cap, with
$\lambda_1 = \lambda_2 < 0$, gives $-\infty$ and $s = -1$, a cup. A cup gives
$+1$, a cap.

The example in the library docstring is such a case. The centre of a single
bright pixel is a cap:

```{code-cell} ipython3
square = np.zeros((5, 5))
square[2, 2] = 4
with np.errstate(divide="ignore", invalid="ignore"):
    print(shape_index(square, sigma=0.1))
```

The centre value is $-1$, a spherical cup. The centre of the Gaussian blob is
also a cap. Whether the formula gets it right depends on whether rounding
leaves a residue:

```{code-cell} ipython3
def centre_rows(sigma, s_func):
    """The shape index at the centre of the bright blob, for each Hessian."""
    image, H_true = blob_truth(BLOB_SIZE, BLOB_WIDTH, sigma)
    middle = BLOB_SIZE // 2
    rows = []
    for name, H in (("exact", H_true),
                    *((n, route(image, sigma)) for n, route in ROUTES)):
        l1, l2 = hessian_matrix_eigvals(H)
        rows.append({"sigma": sigma, "Hessian": name,
                     "l2 - l1": f"{l2[middle, middle] - l1[middle, middle]:+.1e}",
                     "s": f"{s_func(H)[middle, middle]:+.4f}"})
    return rows


show_table(pd.DataFrame([r for s in (0.5, 1.0, 2.0, 4.0)
                         for r in centre_rows(s, s_library)]), index="sigma")
```

The exact Hessian ties at every `sigma`, and calls the top of the hill a cup.
The finite-difference Hessian leaves a residue of $-5.6 \times 10^{-17}$ and
gets the right answer. The Gaussian-derivative Hessian ties at `sigma` 2 and 4,
and gets the wrong one.

The [two-argument arctangent](https://en.wikipedia.org/wiki/Atan2) resolves the
tie. For $\lambda_1 > \lambda_2$,

$$
\arctan\frac{\lambda_2 + \lambda_1}{\lambda_2 - \lambda_1}
= -\operatorname{atan2}(\lambda_1 + \lambda_2,\ \lambda_1 - \lambda_2),
$$

and at $\lambda_1 - \lambda_2 = +0$ the right side is $\mp\pi/2$, with the
correct sign. Only the planar point needs a separate NaN.

```{code-cell} ipython3
def s_from(H):
    """The shape index, with the umbilic at -1 or +1 and the planar point NaN."""
    l1, l2 = hessian_matrix_eigvals(H)
    s = -(2.0 / np.pi) * np.arctan2(l1 + l2, l1 - l2)
    return np.where((l1 == 0) & (l2 == 0), np.nan, s)


show_table(pd.DataFrame([r for s in (0.5, 1.0, 2.0, 4.0)
                         for r in centre_rows(s, s_from)]), index="sigma")
```

Every centre value is now $+1$. Away from ties the two formulas differ only by
rounding:

```{code-cell} ipython3
def fix_cost(route, sigma):
    """How far the corrected formula moves the shape index on the photograph."""
    H = route(PHOTO, sigma)
    a, b = s_library(H), s_from(H)
    ok = np.isfinite(a) & np.isfinite(b)
    l1, l2 = hessian_matrix_eigvals(H)
    return {"sigma": sigma, "route": SHORT[[n for n, r in ROUTES if r is route][0]],
            "exact ties": int(np.sum((l1 == l2) & (l1 != 0))),
            "max |change in s|": f"{np.abs(a - b)[ok].max():.1e}",
            "NaN, library": int(np.isnan(a).sum()), "NaN, corrected": int(np.isnan(b).sum())}


show_table(pd.DataFrame([fix_cost(route, s) for s in (0.5, 1.0, 2.0, 4.0)
                         for _, route in ROUTES]), index="sigma")
square_fixed = s_from(fd_hessian(square, 0.1))
print(f"docstring example, centre: library {shape_index(square, 0.1)[2, 2]:+.0f}, "
      f"corrected {square_fixed[2, 2]:+.0f}")
```

The photograph has no exact ties, so the correction changes nothing there.
Ties need exact symmetry, such as a synthetic image or a single bright pixel.
All comparisons below use `s_from`, so that this defect does not mix with the
Hessian error.

+++

## 6. How much the pictures move

The fraction of pixels that change between the nine classes of §3 is the
change a caller sees.

```{code-cell} ipython3
CLASS_CHANGE = {}


def change_row(image, sigma):
    """How far the two routes disagree on one image at one scale."""
    inner = interior(sigma)
    a = s_from(fd_hessian(image, sigma))[inner]
    b = s_from(gd_hessian(image, sigma))[inner]
    ok = ~np.isnan(a) & ~np.isnan(b)
    CLASS_CHANGE[sigma] = np.mean(np.digitize(a[ok], BINS) != np.digitize(b[ok], BINS))
    return {"sigma": sigma,
            "pixels changing class": f"{100 * CLASS_CHANGE[sigma]:.1f}%",
            "median |change in s|": f"{np.median(np.abs(a[ok] - b[ok])):.3f}",
            "NaN, FD": int(np.isnan(a).sum()), "NaN, GD": int(np.isnan(b).sum())}


show_table(pd.DataFrame([change_row(PHOTO, s) for s in (0.5, 1.0, 2.0, 4.0)]),
           index="sigma")
```

At `sigma` 0.5, 71% of the interior pixels of the photograph change class. At
`sigma` 4, 5% do. Neither route produces NaN on this photograph.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.2), layout="constrained")
VIEW_SIGMA = 1.0
inner = interior(VIEW_SIGMA)
sfd = s_from(fd_hessian(PHOTO, VIEW_SIGMA))[inner]
sgd = s_from(gd_hessian(PHOTO, VIEW_SIGMA))[inner]

for ax, data, title in ((axes[0], sfd, "finite differences"),
                        (axes[1], sgd, "Gaussian derivatives")):
    bare(ax, title).imshow(data, cmap=DIV, norm=TwoSlopeNorm(0, -1, 1))
limit = np.abs(sgd - sfd).max()
third = bare(axes[2], "difference").imshow(
    sgd - sfd, cmap=DIV, norm=TwoSlopeNorm(0, -limit, limit))
fig.colorbar(third, ax=axes[2], fraction=0.046).outline.set_visible(False)
fig.suptitle(f"shape index on camera at sigma = {VIEW_SIGMA}, cup (blue) to cap (orange)",
             color=MUTED, fontsize=9)
```

The difference follows the structure of the picture. Split the pixels by how
far the eigenvalue pair is from isotropic:

```{code-cell} ipython3
pair = hessian_matrix_eigvals(gd_hessian(PHOTO, VIEW_SIGMA))
with np.errstate(divide="ignore", invalid="ignore"):
    anisotropy = (np.abs(pair[0] - pair[1])
                  / np.maximum(np.abs(pair[0]), np.abs(pair[1])))[inner]
gap = np.abs(sgd - sfd)
usable = np.isfinite(gap) & np.isfinite(anisotropy)
edges = np.nanpercentile(anisotropy[usable], [0, 20, 40, 60, 80, 100])

show_table(pd.DataFrame(
    [{"|l1 - l2| / max|l|": f"{edges[k]:.2f} to {edges[k+1]:.2f}",
      "median |change in s|":
          f"{np.median(gap[usable & (anisotropy >= edges[k]) & (anisotropy <= edges[k+1])]):.4f}"}
     for k in range(5)]), index="|l1 - l2| / max|l|")
print(f"Spearman rank correlation over pixels: "
      f"{spearmanr(anisotropy[usable], gap[usable]).statistic:+.2f}")
```

The disagreement is smallest in the middle quintiles and largest in the top
one. At the top of the range the two eigenvalues are close to equal and
opposite, which is a saddle. The rank correlation over pixels is $+0.21$, so
this is a tendency, not a rule.

+++

## 7. Which one is right

+++

### 7.1 The blob and the far field

Let $v = w^2 + \sigma^2$ be the variance of the smoothed blob, $f$ its value,
and $\rho = |\mathbf{x}|/\sqrt{v}$ the distance from its centre in units of
$\sqrt{v}$. The exact Hessian is $f\,(\mathbf{x}\mathbf{x}^T/v^2 - I/v)$. Its
eigenvalues are $(f/v)(\rho^2 - 1)$ along the radius and $-f/v$ across it. So
the exact shape index depends on $\rho$ only:

$$
s = -\frac{2}{\pi}\operatorname{atan2}(\rho^2 - 2,\ \rho^2).
$$

It is $+1$ at the centre, $0$ at $\rho = \sqrt 2$, and goes to $-1/2$ as
$\rho$ grows. There the radial eigenvalue is much larger than the other one, and
the Hessian is close to rank one. The left panel shows the exact $s$ of the
blob, with the saddle ring dotted and the split at `FLOOR` solid:

```{code-cell} ipython3
ZONE_SIGMA = 1.0
_, H_zone = blob_truth(BLOB_SIZE, BLOB_WIDTH, ZONE_SIGMA)
s_zone = s_from(H_zone)
spread = BLOB_WIDTH**2 + ZONE_SIGMA**2
rho = radius(BLOB_SIZE) / np.sqrt(spread)            # distance in units of sqrt(v)
rho_floor = np.sqrt(2 * np.log(1 / FLOOR))
middle = BLOB_SIZE // 2

fig, (left, right) = plt.subplots(1, 2, figsize=(8.4, 3.4),
                                  gridspec_kw=dict(width_ratios=[1, 1.6]))
view = slice(middle - 40, middle + 41)
bare(left, f"exact s of the smoothed blob, sigma = {ZONE_SIGMA}").imshow(
    s_zone[view, view], cmap=DIV, norm=TwoSlopeNorm(0, -1, 1))
left.contour(rho[view, view], levels=[rho_floor], colors=[INK], linewidths=1)
left.contour(rho[view, view], levels=[np.sqrt(2)], colors=[MUTED], linewidths=0.8,
             linestyles="dotted")

row = slice(middle, BLOB_SIZE - 1)
right.plot(rho[middle, row], s_zone[middle, row], color=C_ONE, linewidth=2)
right.axhline(-0.5, color=MUTED, linewidth=0.8, linestyle="dashed")
right.axvline(rho_floor, color=INK, linewidth=1)
right.axvline(np.sqrt(2), color=MUTED, linewidth=0.8, linestyle="dotted")
right.annotate("blob", (rho_floor - 0.15, 0.8), ha="right", color=MUTED, fontsize=8)
right.annotate("far field", (rho_floor + 0.15, 0.8), color=MUTED, fontsize=8)
right.annotate("saddle, s = 0", (np.sqrt(2) + 0.1, 0.05), color=MUTED, fontsize=8)
right.annotate("rank one: s = -1/2", (8, -0.45), color=MUTED, fontsize=8)
right.set_xlim(0, 10); right.set_ylim(-1.05, 1.05)
recede(right, "exact s along a radius")
right.set_xlabel(r"distance from centre / $\sqrt{w^2 + \sigma^2}$", color=MUTED)
fig.tight_layout()
```

The level $e^{-\rho^2/2}$ equals the floor at $\rho^2 = 2\ln(1/\text{floor})$.
So the split is a distance in units of $\sqrt{v}$, and the eigenvalue ratio
there is $\rho^2 - 1$:

```{code-cell} ipython3
_, H_check = blob_truth(BLOB_SIZE, BLOB_WIDTH, ZONE_SIGMA)
s_formula = -(2 / np.pi) * np.arctan2(rho**2 - 2, rho**2)
print(f"max |s_exact - formula| over the image: "
      f"{np.max(np.abs(s_from(H_check) - s_formula)):.1e}")


def split_row(floor):
    """Where the blob ends, for a split at `floor` of the peak."""
    rho2 = 2 * np.log(1 / floor)
    return {"floor": f"{floor:g}", "rho": f"{np.sqrt(rho2):.2f}",
            "radial / tangential eigenvalue": f"{rho2 - 1:.1f}",
            "exact s": f"{-(2 / np.pi) * np.arctan2(rho2 - 2, rho2):+.3f}"}


show_table(pd.DataFrame([split_row(f) for f in (1e-2, 1e-3, 1e-4, 1e-6)]),
           index="floor")
```

At $10^{-3}$, the radial eigenvalue is about 13 times the tangential one, and
the exact $s$ is within 0.05 of $-1/2$. So the floor marks where the Hessian of
the blob becomes close to rank one. It is a choice about shape, and §7.2 shows
how the result moves with it.

### 7.2 Error against the closed form

Per-pixel error against the closed form, as a function of distance from the
blob centre:

```{code-cell} ipython3
def s_errors(route, width, sigma, size=BLOB_SIZE):
    """|s - s_exact| at every pixel of the smoothed blob."""
    image, H_true = blob_truth(size, width, sigma)
    return np.abs(s_from(route(image, sigma)) - s_from(H_true))


def radial_median(values, mask, r):
    """Median of `values` over `mask`, at each integer radius."""
    ri = np.rint(r).astype(int)
    radii = np.unique(ri[mask])
    return radii, np.array([np.median(values[mask & (ri == k)]) for k in radii])


fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), sharey=True)
r = radius(BLOB_SIZE)
for ax, sigma in zip(axes, (0.5, 1.0)):
    inside = zones(BLOB_SIZE, BLOB_WIDTH, sigma)
    inside = inside[0] | inside[1]
    for (name, route), colour in zip(ROUTES, (C_TWO, C_ONE)):
        radii, med = radial_median(s_errors(route, BLOB_WIDTH, sigma), inside, r)
        ax.semilogy(radii, np.maximum(med, 1e-16), color=colour, linewidth=2,
                    label=name)
    edge = np.sqrt(2 * (BLOB_WIDTH**2 + sigma**2) * np.log(1 / FLOOR))
    ax.axvline(edge, color=RULE, linewidth=1, zorder=0)
    ax.annotate(f"blob falls to {FLOOR:g} of peak", (edge + 1, 1e-14),
                color=MUTED, fontsize=8)
    recede(ax, f"sigma = {sigma}")
    ax.set_xlabel("distance from blob centre (pixels)", color=MUTED)
axes[0].set_ylabel("median |s - s_exact|", color=MUTED)
axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="center right")
fig.tight_layout()
```

At `sigma` 0.5 the finite-difference error is about $4 \times 10^{-3}$ over
most radii. The Gaussian-derivative error is smaller inside the blob, and climbs
past the finite-difference error only beyond the floor line. At `sigma` 1 it is
smaller everywhere. Split the interior at the floor and average each part:

```{code-cell} ipython3
SIGMAS = (0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 4.0)
BLOB_ERR = {}


def zone_row(sigma, width=BLOB_WIDTH):
    """Mean |s - s_exact| of each route, inside the blob and in the far field."""
    blob, far = zones(BLOB_SIZE, width, sigma)
    errs = {SHORT[name]: s_errors(route, width, sigma) for name, route in ROUTES}
    row = {"sigma": sigma}
    for zone, mask in (("blob", blob), ("far field", far)):
        means = {k: np.mean(e[mask]) for k, e in errs.items()}
        if zone == "blob":
            BLOB_ERR[sigma] = means
        for k, v in means.items():
            row[f"{zone}, {k}"] = f"{v:.1e}"
        row[f"{zone}, closer"] = min(means, key=means.get)
    return row


show_table(pd.DataFrame([zone_row(s) for s in SIGMAS]), index="sigma")
```

Inside the blob the Gaussian-derivative shape index is closer at every
`sigma`: by a factor of about 2 at `sigma` 0.3, 5 at 0.5, and $10^{10}$ or more
from 1.5. In the far field finite differences are closer at `sigma` 0.3 and
0.5. An average over the whole interior is mostly far field, because the far
field has most of the pixels: at `sigma` 0.5, the blob has 1581 of the 36481
interior pixels (floor 0 in the next table).

The split depends on the floor. Move it:

```{code-cell} ipython3
def floor_row(floor):
    """Which route is closer, when the blob ends at `floor` of the peak."""
    row = {"floor": f"{floor:g}"}
    for sigma in (0.3, 0.5):
        errs = {SHORT[n]: s_errors(r, BLOB_WIDTH, sigma) for n, r in ROUTES}
        blob, _ = zones(BLOB_SIZE, BLOB_WIDTH, sigma, floor)
        means = {k: np.mean(e[blob]) for k, e in errs.items()}
        row[f"sigma {sigma}, pixels"] = int(blob.sum())
        row[f"sigma {sigma}, FD / GD"] = f"{means['FD'] / means['GD']:.2f}"
    return row


show_table(pd.DataFrame([floor_row(f) for f in (1e-2, 1e-3, 1e-4, 1e-6, 0)]),
           index="floor")
```

A ratio above 1 means Gaussian derivatives are closer. At `sigma` 0.3,
finite differences win only when the blob extends down to $10^{-6}$ of the
peak. At `sigma` 0.5 they win only with a floor of 0, which is the whole
interior.

The result holds over blob widths:

```{code-cell} ipython3
def width_row(width):
    """Mean |s - s_exact| inside the blob, for one blob width."""
    size = 201 if width <= 10 else 301
    row = {"blob width": width}
    for sigma in (0.3, 0.5):
        blob, _ = zones(size, width, sigma)
        for name, route in ROUTES:
            row[f"sigma {sigma}, {SHORT[name]}"] = \
                f"{np.mean(s_errors(route, width, sigma, size)[blob]):.1e}"
    return row


show_table(pd.DataFrame([width_row(w) for w in (2.0, 3.0, 6.0, 10.0, 16.0)]),
           index="blob width")
```

The finite-difference error barely moves with `sigma`. The stencil error of a
central difference is second order in the pixel spacing, relative to the width
of the smoothed blob, $\sqrt{w^2 + \sigma^2}$. So the error times
$w^2 + \sigma^2$ should be near constant:

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"blob width": w, "sigma": s,
      "FD error": f"{np.mean(s_errors(fd_hessian, w, s, 301)[zones(301, w, s)[0]]):.2e}",
      "FD error * (w**2 + sigma**2)":
          f"{np.mean(s_errors(fd_hessian, w, s, 301)[zones(301, w, s)[0]]) * (w**2 + s**2):.3f}"}
     for w in (3.0, 6.0, 12.0) for s in (0.5, 2.0)]), index="blob width")
```

The scaled error stays in a narrow band while the raw error changes by a
factor of more than ten. At a fixed blob width, `sigma` below 2 adds little to
$w^2$, so the finite-difference error stays flat.

The Hessian ranks the routes the same way as the shape index inside the blob:

```{code-cell} ipython3
def hessian_error(route, sigma):
    """Worst element error over the interior, as a fraction of the peak."""
    image, H_true = blob_truth(BLOB_SIZE, BLOB_WIDTH, sigma)
    inner = interior(sigma)
    peak = max(np.abs(h[inner]).max() for h in H_true)
    return max(np.abs(a[inner] - b[inner]).max()
               for a, b in zip(route(image, sigma), H_true)) / peak


show_table(pd.DataFrame(
    [{"sigma": s,
      **{f"H, {SHORT[name]}": f"{hessian_error(route, s):.2e}" for name, route in ROUTES},
      "H closer": SHORT[min(ROUTES, key=lambda nr: hessian_error(nr[1], s))[0]],
      "s closer, blob": min(BLOB_ERR[s], key=BLOB_ERR[s].get)}
     for s in (0.3, 0.5, 0.8, 1.0, 2.0)]), index="sigma")
```

### 7.3 An RMS test of the whole Hessian

The worst-element error above depends on one element at one pixel. A test of
overall quality should use the whole matrix at every pixel, should not change
when the image axes rotate, and should not need the floor of §7.1. The
relative root mean square (RMS) error of the matrix meets all three:

$$
E = \sqrt{\frac{\sum_{\mathbf{x}} \lVert \hat H(\mathbf{x}) - H(\mathbf{x}) \rVert_F^2}
               {\sum_{\mathbf{x}} \lVert H(\mathbf{x}) \rVert_F^2}},
\qquad
\lVert H \rVert_F^2 = H_{ii}^2 + 2 H_{ij}^2 + H_{jj}^2 = \lambda_1^2 + \lambda_2^2.
$$

$\hat H$ is the computed Hessian, $H$ the exact one, and the sums are over the
interior pixels. $\lVert \cdot \rVert_F$ is the
[Frobenius norm](https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm).
$H_{ij}$ counts twice, because it is in the matrix twice. A rotation $R$ of the
image axes changes each matrix to $R H R^T$, which has the same eigenvalues, so
the norm does not change. $\lVert H \rVert_F^2$ is $2c^2$, where $c$ is the
curvedness of §4, so each pixel counts in proportion to its curvature. If every
element has the same gain error $1 + \epsilon$, then $E = |\epsilon|$. So $E$
reads as a fractional error.

```{code-cell} ipython3
FROBENIUS_WEIGHTS = (1, 2, 1)          # H_ii, H_ij, H_jj: H_ij is in the matrix twice


def hessian_rms(H, H_true, mask):
    """Relative RMS error of the whole Hessian over `mask`, by the Frobenius norm."""
    err = sum(w * (a - b)**2 for w, a, b in zip(FROBENIUS_WEIGHTS, H, H_true))
    ref = sum(w * b**2 for w, b in zip(FROBENIUS_WEIGHTS, H_true))
    return np.sqrt(err[mask].sum() / ref[mask].sum())


def rotate_hessian(H, angle):
    """R H R^T at every pixel, for a rotation of the axes by `angle`."""
    co, si = np.cos(angle), np.sin(angle)
    hii, hij, hjj = H
    return [co*co*hii - 2*co*si*hij + si*si*hjj,
            co*si*(hii - hjj) + (co*co - si*si)*hij,
            si*si*hii + 2*co*si*hij + co*co*hjj]
```

Three checks of the test itself, on the blob at `sigma` 1, with the
finite-difference Hessian as $\hat H$:

```{code-cell} ipython3
image, H_true = blob_truth(BLOB_SIZE, BLOB_WIDTH, 1.0)
blob, far = zones(BLOB_SIZE, BLOB_WIDTH, 1.0)
inner = blob | far
H_fd = fd_hessian(image, 1.0)
show_table(pd.DataFrame([
    {"check": "exact Hessian against itself, want 0",
     "E": f"{hessian_rms(H_true, H_true, inner):.1e}"},
    {"check": "exact Hessian times 1.01, want 0.01",
     "E": f"{hessian_rms([1.01 * h for h in H_true], H_true, inner):.6f}"},
    {"check": "FD Hessian",
     "E": f"{hessian_rms(H_fd, H_true, inner):.6f}"},
    {"check": "FD and exact, both rotated by 35 degrees, want the line above",
     "E": f"{hessian_rms(rotate_hessian(H_fd, np.deg2rad(35)), rotate_hessian(H_true, np.deg2rad(35)), inner):.6f}"},
]), index="check")
```

On the blob, for each route, over the whole interior and over the blob zone of
§7.1 only:

```{code-cell} ipython3
RMS_BLOB = {}


def rms_blob_row(sigma):
    """E for each route on the blob, over the interior and over the blob zone."""
    image, H_true = blob_truth(BLOB_SIZE, BLOB_WIDTH, sigma)
    blob, far = zones(BLOB_SIZE, BLOB_WIDTH, sigma)
    row = {"sigma": sigma}
    for name, route in ROUTES:
        H = route(image, sigma)
        whole, zone = hessian_rms(H, H_true, blob | far), hessian_rms(H, H_true, blob)
        RMS_BLOB[sigma, SHORT[name]] = whole
        row[f"E, {SHORT[name]}"] = f"{whole:.2e}"
        row[f"{SHORT[name]}, blob zone / interior"] = f"{zone / whole:.4f}"
    return row


show_table(pd.DataFrame([rms_blob_row(s) for s in (0.3, 0.5, 1.0, 2.0, 4.0)]),
           index="sigma")
```

The finite-difference Hessian is about 2% out at every `sigma` from 0.3 to 2,
and 1.7% at 4. The Gaussian-derivative Hessian is 0.6% out at `sigma` 0.3,
0.3% at 0.5, $10^{-6}$ at 1, and near $10^{-13}$ from 2. The blob zone and the whole
interior give the same $E$ to within 0.4%, because the far field has little
curvature, and $E$ weights each pixel by its curvature. So $E$ needs no floor.
The mean error in $s$ of §7.2 weights each pixel equally, and so it sees the
far field, where the Hessian is close to rank one.

A smooth blob has few high frequencies (§10). The stripes of §2 have narrow
bumps, and an angle puts curvature into $H_{ij}$. The image is the raw 1-D
profile laid down at each angle; the exact Hessian is that of its smoothing:

```{code-cell} ipython3
STRIPE_GRID = np.indices((161, 161), dtype=float) - 80


def rms_stripes_row(degrees, sigma, routes=ROUTES):
    """E for each route on the stripes of §2, at one angle and one scale."""
    image, _ = stripes_truth(STRIPE_GRID, np.deg2rad(degrees), 0)
    _, H_true = stripes_truth(STRIPE_GRID, np.deg2rad(degrees), sigma)
    mask = np.zeros(image.shape, bool)
    mask[interior(sigma)] = True
    return {"angle (degrees)": degrees, "sigma": sigma,
            **{f"E, {SHORT.get(n, n)}": f"{hessian_rms(r(image, sigma), H_true, mask):.2e}"
               for n, r in routes}}


show_table(pd.DataFrame([rms_stripes_row(d, s) for d in (0, 35) for s in (0.5, 1.0, 2.0)]),
           index="angle (degrees)")
```

On the stripes the finite-difference Hessian is 11% to 18% out at `sigma` 0.5
and 1, against about 2% on the blob. It is 8% to 13% out at `sigma` 2, where
the Gaussian-derivative Hessian is exact to $10^{-13}$. The finite-difference
error changes with the angle, from 18% at 0 degrees to 11% at 35 degrees at
`sigma` 0.5. $E$ does not depend on the angle of the axes, so that change is in
the route.

## 8. Why finite differences win in the far field

Far from its centre, the blob is close to an exponential. About a point
$\mathbf{x}_0$, $f(\mathbf{x}_0 + \mathbf{h}) = f(\mathbf{x}_0)\exp(\mathbf{a}
\cdot \mathbf{h} - |\mathbf{h}|^2 / 2v)$ with $\mathbf{a} = -\mathbf{x}_0 / v$
and $v = w^2 + \sigma^2$. Far out, $|\mathbf{a}|^2 \gg 1/v$, so the Hessian
$f\,(\mathbf{a}\mathbf{a}^T - I/v)$ is close to rank one. The shape index there
depends on the small eigenvalue, $-f/v$, and so on small errors.

Take a pure exponential, $f = e^{ax + by}$. Its Hessian $f\,\mathbf{a}
\mathbf{a}^T$ has rank one, so $s = -1/2$ exactly, whatever $a$ and $b$. Let
$\delta$ be the central difference, $(f(x+1) - f(x-1))/2$. The finite-difference
route uses $\delta$ twice for every element. At small `sigma`, the corrected
kernels approach $(1, -2, 1)$ for the diagonal elements and $\delta$ along
each axis for the mixed element:

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"order": o, "taps at sigma 0.3": np.array2string(taps(0.3, o), precision=4)}
     for o in (0, 1, 2)]), index="order")
```

The determinant of each discrete Hessian, divided by $f^2$:

```{code-cell} ipython3
x, y, a, b = sp.symbols("x y a b", real=True)
f = sp.exp(a * x + b * y)


def d1(e, v):
    """Central difference along v."""
    return (e.subs(v, v + 1) - e.subs(v, v - 1)) / 2


def d2(e, v):
    """The (1, -2, 1) second difference along v."""
    return e.subs(v, v + 1) - 2 * e + e.subs(v, v - 1)


H_fd = sp.Matrix([[d1(d1(f, x), x), d1(d1(f, x), y)],
                  [d1(d1(f, x), y), d1(d1(f, y), y)]])
H_gd0 = sp.Matrix([[d2(f, x), d1(d1(f, x), y)],
                   [d1(d1(f, x), y), d2(f, y)]])
det_fd = sp.simplify((H_fd.det() / f**2).rewrite(sp.exp))
target = (16 * sp.sinh(a / 2)**2 * sp.sinh(b / 2)**2
          * (1 - sp.cosh(a / 2)**2 * sp.cosh(b / 2)**2))
residue = sp.simplify((H_gd0.det() / f**2 - target).rewrite(sp.exp))
print("finite differences:   det / f**2 =", det_fd)
print("small-sigma GD limit: det / f**2 =", target, "  (residue", residue, ")")
```

The finite-difference Hessian of any exponential is $f\,\mathbf{s}\mathbf{s}^T$
with $\mathbf{s} = (\sinh a, \sinh b)$, which has rank one. A sampled Gaussian
multiplies an exponential by a constant, so the finite-difference route gives
$s = -1/2$ exactly, at every `sigma`. The small-`sigma` limit of the Gaussian
derivative kernels has a negative determinant for any non-zero $a$ and $b$.
That is one positive and one negative eigenvalue, so $s$ moves from $-1/2$
towards the saddle.

On an image:

```{code-cell} ipython3
def ramp_error(route, sigma, a, b, size=64):
    """Median |s + 1/2| on the exponential ramp exp(a r + b c)."""
    r, c = np.indices((size, size), dtype=float)
    s = s_from(route(np.exp(a * r + b * c), sigma))[interior(sigma)]
    return np.median(np.abs(s + 0.5))


show_table(pd.DataFrame(
    [{"ramp (a, b)": f"({a_}, {b_})", "sigma": s,
      **{SHORT[n]: f"{ramp_error(r_, s, a_, b_):.1e}" for n, r_ in ROUTES}}
     for a_, b_ in ((0.2, 0.1), (0.6, 0.3)) for s in (0.3, 0.5, 1.0, 2.0)]),
    index="ramp (a, b)")
```

Finite differences are exact to rounding on both ramps. The Gaussian-derivative
error grows with the steepness of the ramp and falls with `sigma`. In the
blob's far field at radius 30, $|\mathbf{a}| = 30/v \approx 0.8$ per pixel, for
$w = 6$ and small `sigma`.

Which element carries the error is decidable: replace one element with its
exact value and measure the angle of §4 again.

```{code-cell} ipython3
def angle_error(H, H_true, mask):
    """Median angular error of the eigenvalue pair over `mask`, in radians."""
    lt, lc = hessian_matrix_eigvals(H_true), hessian_matrix_eigvals(H)
    turn = np.arctan2(lc[1], lc[0]) - np.arctan2(lt[1], lt[0])
    return np.median(np.abs(np.arctan2(np.sin(turn), np.cos(turn)))[mask])


def swap_rows(sigma):
    """Angular error with each element in turn replaced by its exact value."""
    image, H_true = blob_truth(BLOB_SIZE, BLOB_WIDTH, sigma)
    blob, far = zones(BLOB_SIZE, BLOB_WIDTH, sigma)
    out = []
    for zone, mask in (("blob", blob), ("far field", far)):
        for name, route in ROUTES:
            H = route(image, sigma)
            out.append({
                "sigma": sigma, "zone": zone, "route": SHORT[name],
                "as computed": f"{angle_error(H, H_true, mask):.1e}",
                "exact mixed": f"{angle_error([H[0], H_true[1], H[2]], H_true, mask):.1e}",
                "exact diagonals":
                    f"{angle_error([H_true[0], H[1], H_true[2]], H_true, mask):.1e}"})
    return out


show_table(pd.DataFrame([r for s in (0.3, 0.5) for r in swap_rows(s)]), index="sigma")
```

Inside the blob, the finite-difference error is mostly in the diagonal
elements. Exact diagonals cut it by a factor of about 4. The doubled central
difference, $(\tfrac14, 0, -\tfrac12, 0, \tfrac14)$, is a wide stencil. At
`sigma` 0.3 the Gaussian-derivative error is split between the diagonals and
the mixed element, and neither substitution alone removes it.

In the far field the pattern is different. One exact element makes the
finite-difference angle about 10 to 40 times worse, because it breaks the
rank-one structure that the shared operator $\delta$ keeps. At `sigma` 0.3 the
Gaussian-derivative far-field error is almost all in the mixed element: the
exact mixed element cuts it by a factor of about 50.

+++

## 9. A more careful mixed element

The mixed element is two passes of the first-derivative kernel. For
$f = x^n$, `correlate1d` returns $\sum_x k(x)\, x^n$ at the origin. Smoothing $x^3$ at scale
$\sigma$ gives $x^3 + 3\sigma^2 x$, whose derivative at the origin is
$3\sigma^2$. So the kernel owes three conditions, and the shipped one imposes
the first two:

```{code-cell} ipython3
def first_moments(sigma):
    """What the shipped first-derivative kernel gets right and wrong."""
    k = taps(sigma, 1)
    radius = len(k) // 2
    x = np.arange(-radius, radius + 1, dtype=float)
    third, want = (k * x**3).sum(), 3 * sigma**2
    return {"sigma": sigma, "taps": len(k),
            "sum(k), want 0": f"{k.sum():.0e}",
            "sum(k * x), want 1": f"{(k * x).sum():.6f}",
            "sum(k * x**3)": f"{third:.4f}",
            "want 3 sigma**2": f"{want:.4f}",
            "relative error": f"{abs(third - want) / want:.1%}"}


show_table(pd.DataFrame([first_moments(s) for s in (0.3, 0.5, 0.8, 1.0, 2.0)]),
           index="sigma")
```

The third moment is 270% out at `sigma` 0.3 and correct to 0.1% from 0.8.
Below `sigma` of about 0.5 the kernel is close to the plain central difference,
whose third moment is 1 at every `sigma`.

Two odd basis functions carry two conditions:

```{code-cell} ipython3
def taps1_third(sigma, truncate=8):
    """First-derivative kernel with the first and third moments imposed."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g /= g.sum()
    basis = (g * (x / sigma**2), g * (x**3 / sigma**4))
    M = np.array([[(b * x).sum() for b in basis],
                  [(b * x**3).sum() for b in basis]])
    weights = np.linalg.solve(M, np.array([1.0, 3.0 * sigma**2]))
    return weights[0] * basis[0] + weights[1] * basis[1], np.linalg.cond(M)


show_table(pd.DataFrame(
    [{"sigma": s, "cond(M)": f"{taps1_third(s)[1]:.1e}",
      "largest change vs shipped": f"{np.abs(taps1_third(s)[0] - taps(s, 1)).max():.3f}"}
     for s in (0.3, 0.5, 0.8, 1.0, 2.0)]), index="sigma")
```

The change is 0.12 of a tap at `sigma` 0.3 and below $10^{-3}$ from 0.8. The
$2 \times 2$ system has a condition number of $10^7$ at `sigma` 0.3.

```{code-cell} ipython3
def gd_third(image, sigma, mode="constant", cval=0, truncate=8):
    """gd_hessian with the third moment imposed on the first derivative."""
    image = np.asarray(image, dtype=float)
    ndim = image.ndim
    kernels = {0: taps(sigma, 0, truncate), 1: taps1_third(sigma, truncate)[0],
               2: taps(sigma, 2, truncate)}
    elements = []
    for ax0, ax1 in combinations_with_replacement(range(ndim), 2):
        orders = [0] * ndim
        orders[ax0] += 1
        orders[ax1] += 1
        result = image
        for axis, o in enumerate(orders):
            result = ndi.correlate1d(result, kernels[o], axis=axis,
                                     mode=mode, cval=cval)
        elements.append(result)
    return elements


def mixed_row(sigma):
    """Mixed element and shape-index error inside the blob, for three kernels."""
    image, H_true = blob_truth(BLOB_SIZE, BLOB_WIDTH, sigma)
    blob, _ = zones(BLOB_SIZE, BLOB_WIDTH, sigma)
    peak = max(np.abs(h[blob]).max() for h in H_true)
    truth = s_from(H_true)
    row = {"sigma": sigma}
    for name, route in (("FD", fd_hessian), ("GD", gd_hessian), ("GD, third", gd_third)):
        H = route(image, sigma)
        row[f"Hij, {name}"] = f"{np.abs(H[1] - H_true[1])[blob].max() / peak:.1e}"
        row[f"s, {name}"] = f"{np.mean(np.abs(s_from(H) - truth)[blob]):.1e}"
    return row


show_table(pd.DataFrame([mixed_row(s) for s in (0.3, 0.5, 0.8, 1.0)]), index="sigma")
```

The RMS test of §7.3 gives the same picture for the whole Hessian. On the
stripes at 35 degrees, $H_{ij}$ carries part of the curvature:

```{code-cell} ipython3
THIRD_ROUTES = (("GD", gd_hessian), ("GD, third", gd_third))
rows = []
for sigma in (0.3, 0.5, 1.0):
    image, H_true = blob_truth(BLOB_SIZE, BLOB_WIDTH, sigma)
    blob, far = zones(BLOB_SIZE, BLOB_WIDTH, sigma)
    stripes = rms_stripes_row(35, sigma, THIRD_ROUTES)
    rows.append({"sigma": sigma,
                 **{f"blob, E, {n}": f"{hessian_rms(r(image, sigma), H_true, blob | far):.2e}"
                    for n, r in THIRD_ROUTES},
                 **{f"stripes 35 degrees, {k}": v for k, v in stripes.items()
                    if k.startswith("E")}})
show_table(pd.DataFrame(rows), index="sigma")
```

At `sigma` 0.3 the third moment cuts $E$ by a factor of about 2.6 on the blob
and 6 on the stripes. At 0.5 the factors are about 1.4 and 1.9. At 1 the change
is below 2%.

The third moment improves the mixed element by a factor of about 50 at `sigma`
0.3 and 0.5, and the shape index inside the blob by about 2. From `sigma` 0.8
the shape index changes by less than 15%, although the mixed element still
improves by a factor of about 13 at 0.8 and 6 at 1.

+++

## 10. The blob flatters finite differences

A central difference errs on high spatial frequencies, and a smoothed blob has
almost none. On the photograph, at `sigma` 2 and above, the Gaussian-derivative
shape index is the reference: §7 puts its blob error at about $10^{-13}$ or
below from `sigma` 1.5.

```{code-cell} ipython3
def fd_cost(image, sigma, truth=None, mask=None):
    """FD's distance from a reference, and how often it changes the class."""
    a = s_from(fd_hessian(image, sigma))
    b = s_from(gd_hessian(image, sigma) if truth is None else truth)
    if mask is None:
        mask = np.zeros(image.shape, bool)
        mask[interior(sigma)] = True
    ok = mask & np.isfinite(a) & np.isfinite(b)
    return (f"{np.median(np.abs(a - b)[ok]):.4f}",
            f"{100 * np.mean(np.digitize(a[ok], BINS) != np.digitize(b[ok], BINS)):.2f}%")


rows = []
for sigma in (2.0, 4.0, 8.0):
    blob_image, blob_H = blob_truth(401, 12.0, sigma)
    med_b, cls_b = fd_cost(blob_image, sigma, truth=blob_H,
                           mask=zones(401, 12.0, sigma)[0])
    med_p, cls_p = fd_cost(PHOTO, sigma)
    rows.append({"sigma": sigma, "blob, median |ds|": med_b, "blob, wrong class": cls_b,
                 "camera, median |ds|": med_p, "camera, wrong class": cls_p})
show_table(pd.DataFrame(rows), index="sigma")
```

At `sigma` 2 the finite-difference error on the photograph is about 20 times
its error inside the blob. It moves 16% of the photograph's pixels across a
class boundary, against 0.6% of the blob's. By `sigma` 8 it is under 1% on both
images.

A photograph could have more pixels near a class boundary, which would inflate
its class changes. Measure the distance to the nearest edge:

```{code-cell} ipython3
def boundary_distance(image, sigma, mask=None):
    """Median distance from the shape index to the nearest class edge."""
    s = s_from(gd_hessian(image, sigma))
    if mask is None:
        mask = np.zeros(image.shape, bool)
        mask[interior(sigma)] = True
    s = s[mask & np.isfinite(s)]
    return np.median(np.min(np.abs(s[..., None] - BINS), axis=-1))


blob_image, _ = blob_truth(401, 12.0, 2.0)
show_table(pd.DataFrame(
    [{"image": "camera", "median distance to a class edge":
      f"{boundary_distance(PHOTO, 2.0):.4f}"},
     {"image": "Gaussian blob", "median distance to a class edge":
      f"{boundary_distance(blob_image, 2.0, zones(401, 12.0, 2.0)[0]):.4f}"}]),
    index="image")
```

Inside the blob the shape index is closer to a class edge than on the
photograph, so boundary proximity does not explain the gap. The difference is
in the size of the error. Read §7's finite-difference numbers as a best case.

+++

## 11. Noise and cost

The finite-difference shape index is steadier under noise, by a factor of
about 1.8 at `sigma` 0.5 and 1, and 1.2 at `sigma` 2, at the larger noise
amplitude. The test adds white noise to the blob and measures the spread of
$s$ over pixels that carry the top 70% of curvature.

```{code-cell} ipython3
NOISE = {}


def noise_row(sigma, amplitude, draws=12):
    """Spread of s under repeated noise, where the blob has real curvature."""
    image, _ = blob_truth(BLOB_SIZE, BLOB_WIDTH, sigma)
    _, H_true = blob_truth(BLOB_SIZE, BLOB_WIDTH, sigma)
    strength = np.sqrt(H_true[0]**2 + 2 * H_true[1]**2 + H_true[2]**2)
    inner = interior(sigma)
    mask = (strength > 0.3 * strength.max())[inner]
    rng = np.random.default_rng(0)
    draws_by_route = {SHORT[name]: [] for name, _ in ROUTES}
    for _ in range(draws):
        noisy = image + amplitude * rng.normal(size=image.shape)
        for name, route in ROUTES:
            draws_by_route[SHORT[name]].append(s_from(route(noisy, sigma))[inner])
    spread = {k: np.nanmean(np.nanstd(np.stack(v), axis=0)[mask])
              for k, v in draws_by_route.items()}
    NOISE[sigma, amplitude] = spread["GD"] / spread["FD"]
    return {"sigma": sigma, "noise": amplitude,
            **{k: f"{v:.4f}" for k, v in spread.items()},
            "GD / FD": f"{NOISE[sigma, amplitude]:.2f}"}


show_table(pd.DataFrame([noise_row(s, a) for s in (0.5, 1.0, 2.0)
                         for a in (0.005, 0.02)]), index="sigma")
```

Both routes have a gain of 1 on a quadratic, so the noise each element passes
is a fair comparison. White noise of unit variance gives an output standard
deviation equal to the root sum of squares of the element's impulse response:

```{code-cell} ipython3
def noise_gain(route, sigma, size=81):
    """Root sum of squares of each Hessian element's impulse response."""
    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1
    return [np.sqrt((h**2).sum()) for h in route(impulse, sigma)]


def quadratic_gain(route, sigma, size=81):
    """Response of H_ii to r**2 / 2 at the centre: 1 for a correct gain."""
    r = np.arange(size, dtype=float)[:, None] - size // 2
    image = np.broadcast_to(r**2 / 2, (size, size)).copy()
    return route(image, sigma)[0][size // 2, size // 2]


show_table(pd.DataFrame(
    [{"sigma": s, "route": SHORT[name],
      "gain on r**2 / 2": f"{quadratic_gain(route, s):.6f}",
      **{f"noise gain, {e}": f"{g:.3f}"
         for e, g in zip(("Hii", "Hij", "Hjj"), noise_gain(route, s))}}
     for s in (0.5, 1.0, 2.0) for name, route in ROUTES]), index="sigma")
```

At equal gain on a quadratic, the finite-difference elements pass less noise,
by a factor of about 5 on the diagonals at `sigma` 0.5 and about 2 at
`sigma` 1. The frequency responses of the 1-D second derivatives show why. The
response of the doubled central difference is $-\sin^2\omega$, which is zero
at the [Nyquist frequency](https://en.wikipedia.org/wiki/Nyquist_frequency),
$\omega = \pi$. The response of $(1, -2, 1)$ is
$-4\sin^2(\omega/2)$, which is largest there:

```{code-cell} ipython3
def frequency_response(route_1d, sigma, order=2, size=256):
    """Real DFT of the 1-D impulse response, against frequency / Nyquist."""
    h = np.fft.ifftshift(impulse_response(route_1d, sigma, order, size))
    return np.linspace(0, 1, size // 2 + 1), np.fft.rfft(h).real


fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
for ax, sigma in zip(axes, (0.5, 1.0)):
    nu, _ = frequency_response(fd_1d, sigma)
    omega = np.pi * nu
    ax.plot(nu, -omega**2 * np.exp(-(sigma * omega)**2 / 2), color=RULE,
            linewidth=3, label="continuous", zorder=0)
    for route_1d, colour, label in ((fd_1d, C_TWO, "finite differences"),
                                    (gd_1d, C_ONE, "Gaussian derivatives")):
        ax.plot(*frequency_response(route_1d, sigma), color=colour, linewidth=2,
                label=label)
    ax.axhline(0, color=RULE, linewidth=0.8, zorder=0)
    recede(ax, f"second derivative, sigma = {sigma}")
    ax.set_xlabel("frequency / Nyquist", color=MUTED)
axes[0].set_ylabel("response", color=MUTED)
axes[0].legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="lower left")
fig.tight_layout()
```

```{code-cell} ipython3
def nyquist_row(sigma):
    """Each second-derivative response at the Nyquist frequency."""
    row = {"sigma": sigma,
           "continuous": f"{-np.pi**2 * np.exp(-(sigma * np.pi)**2 / 2):+.1e}"}
    for name, route_1d in (("FD", fd_1d), ("GD", gd_1d)):
        row[name] = f"{frequency_response(route_1d, sigma)[1][-1]:+.1e}"
    return row


show_table(pd.DataFrame([nyquist_row(s) for s in (0.5, 1.0, 2.0)]), index="sigma")
```

At the lowest frequencies the three curves agree; that is the gain of 1 on a
quadratic. At `sigma` 0.5 the Gaussian-derivative response at Nyquist is larger
than the continuous one, and the finite-difference response is zero. By
[Parseval's theorem](https://en.wikipedia.org/wiki/Parseval%27s_theorem), the
noise gain above is the root mean square of the
response over all frequencies. So the large high-frequency response of the
Gaussian-derivative kernel passes more white noise.

Cost:

```{code-cell} ipython3
TIME_RATIO = {}
rows = []
for sigma in (0.5, 1.0, 2.0, 4.0):
    timing = {}
    for name, route in ROUTES:
        runs = []
        for _ in range(3):                      # best of three, not the mean
            started = time.perf_counter()
            route(PHOTO, sigma)
            runs.append(time.perf_counter() - started)
        timing[SHORT[name]] = min(runs)
    TIME_RATIO[sigma] = timing["GD"] / timing["FD"]
    rows.append({"sigma": sigma, **{n: f"{t * 1e3:.1f} ms" for n, t in timing.items()},
                 "GD / FD": f"{TIME_RATIO[sigma]:.1f}x"})
show_table(pd.DataFrame(rows), index="sigma")
```

The finite-difference route smooths once, with a kernel of about $8\sigma$
taps, and then takes cheap differences. The Gaussian-derivative route makes two
passes for each of three elements, with kernels of about $16\sigma$ taps. So
the ratio grows with `sigma`.

+++

## 12. What other libraries do

The shape index appears elsewhere almost only on explicit surfaces, such as
triangle meshes. There a quadratic is fitted to a vertex neighbourhood, so the
choice between finite differences and Gaussian derivatives does not arise. The
two implementations read here are FreeSurfer's `mris_curvature_stats` and
`pycurv`. Neither OpenCV nor Pillow has a shape index or a curvedness:

```{code-cell} ipython3
def matching(module):
    """Public names in `module` that mention the shape index or curvedness."""
    return [n for n in dir(module)
            if "shapeindex" in n.lower().replace("_", "") or "curvedness" in n.lower()]


show_table(pd.DataFrame(
    [{"library": label, "matching names": ", ".join(matching(m)) or "none"}
     for label, m in (("cv2", cv2), ("PIL.ImageFilter", PIL.ImageFilter))]),
    index="library")
```

The two surface implementations disagree with scikit-image in two ways. The
first is the expression:

```{code-cell} ipython3
DOME = (-1.0, -2.0)            # k1 >= k2, both negative: a dome


def library_forms(k1, k2):
    """The same dome, under each library's published expression."""
    return [
        {"library": "scikit-image, after Koenderink",
         "expression": "(2/pi) atan((k2+k1)/(k2-k1))",
         "s": f"{2 / pi * atan((k2 + k1) / (k2 - k1)):+.3f}"},
        {"library": "pycurv",
         "expression": "(2/pi) atan((k1+k2)/(k1-k2))",
         "s": f"{2 / pi * atan((k1 + k2) / (k1 - k2)):+.3f}"},
        {"library": "FreeSurfer",
         "expression": "atan((k1+k2)/(k2-k1))",
         "s": f"{atan((k1 + k2) / (k2 - k1)):+.3f}"},
    ]


show_table(pd.DataFrame(library_forms(*DOME)), index="library")
```

`pycurv` reverses the denominator, so with the same curvatures its value is the
negative of Koenderink's. Whether its output differs in sign also depends on
its sign convention for curvature, which this notebook does not test.
FreeSurfer omits the $2/\pi$, so its range is $[-\pi/2, +\pi/2]$.

The second is the umbilic of §5. FreeSurfer returns 0 when the two curvatures
are equal (`mris_curvature_stats.cpp`):

```c
return (af_k1 == af_k2 ? 0 : atan((af_k1+af_k2)/(af_k2 - af_k1)));
```

The comment in the same file says that "The 'Shape Index' can be problematic
due to the atan calculations". `pycurv` returns 0 only when both curvatures
are zero (`curvature_definitions.py`). At an umbilic its Python float division
raises:

```{code-cell} ipython3
def pycurv_shape_index(kappa_1, kappa_2):
    """`pycurv.curvature_definitions.calculate_shape_index`, transcribed."""
    if kappa_1 == 0 and kappa_2 == 0:
        return 0
    return 2 / pi * atan((kappa_1 + kappa_2) / (kappa_1 - kappa_2))


try:
    pycurv_shape_index(-1.0, -1.0)
except ZeroDivisionError as err:
    print(f"ZeroDivisionError: {err}")
```

None of the three gives the $\pm 1$ of the correct sign that the paper asks
for at an umbilic. At the planar point, scikit-image gives NaN, and FreeSurfer
and `pycurv` give 0.

+++

### 12.1 Published forms

The papers of §15 use four forms of the same quantity. They differ in range, in
sign convention, and in what happens at a tie. The cell below evaluates each
form as printed, with NumPy floats, so that division by zero gives an infinity.
A form is correct at a tie when the cap tie has the same sign as the dome.

```{code-cell} ipython3
def published_forms(k_max, k_min):
    """Each published shape index expression, for curvatures k_max >= k_min."""
    k1, k2 = np.float64(k_max), np.float64(k_min)
    H, K = (k1 + k2) / 2, k1 * k2
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "Koenderink 1992; skimage; Bonde 2013":
                (2 / np.pi) * np.arctan((k2 + k1) / (k2 - k1)),
            "Dorai & Jain 1997; Yoshida 2001 (0 to 1)":
                0.5 - np.arctan((k1 + k2) / (k1 - k2)) / np.pi,
            "Worthington & Hancock 2001; Goncalves 2016":
                (2 / np.pi) * np.arctan((k1 + k2) / (k1 - k2)),
            "Yang 2017, Eq. 5 with the mean curvature":
                (2 / np.pi) * np.arctan(H / np.sqrt(H**2 - K)),
        }


shapes = {"dome (-1, -2)": (-1.0, -2.0), "cap, tie (-1, -1)": (-1.0, -1.0),
          "cup, tie (+1, +1)": (1.0, 1.0)}
forms = pd.DataFrame({name: published_forms(*k) for name, k in shapes.items()})
centre = forms.copy()
centre.loc["Dorai & Jain 1997; Yoshida 2001 (0 to 1)"] -= 0.5   # midpoint of 0 to 1
same_side = np.sign(centre["cap, tie (-1, -1)"]) == np.sign(centre["dome (-1, -2)"])
table = forms.map(lambda v: f"{v:+.3f}")
table["cap tie on the dome's side"] = np.where(same_side, "yes", "NO")
show_table(table.rename_axis("form"), index=True)
```

Only Koenderink's printed form fails at a tie, because its denominator,
$\kappa_2 - \kappa_1$, is never positive. scikit-image follows that form. The
other three put $\kappa_{max} - \kappa_{min}$ in the denominator, which is never
negative, so a tie gives the infinity of the right sign. The form of Dorai and
Jain is $(s + 1)/2$. The forms of Worthington and Hancock, Gonçalves and
colleagues, and Yang and colleagues are $-s$, which matches their own sign
conventions for curvature.

The form of Yang and colleagues has a second weakness: it needs
$H^2 - K \geq 0$, and near a tie rounding can make that difference negative.
`hessian_matrix_eigvals` takes the root of a sum of squares, which cannot be
negative:

```{code-cell} ipython3
rng = np.random.default_rng(1)
kappa = rng.uniform(-1, 1, 100_000)
near = kappa * (1 + 1e-12 * rng.uniform(-1, 1, kappa.size))   # a near tie
H_mean, K_gauss = (kappa + near) / 2, kappa * near
pairs = [kappa, np.zeros_like(kappa), near]
show_table(pd.DataFrame([
    {"form": "Yang 2017, Eq. 5",
     "NaN fraction": f"{np.mean(H_mean**2 - K_gauss < 0):.1%}"},
    {"form": "scikit-image, from hessian_matrix_eigvals",
     "NaN fraction": f"{np.mean(np.isnan(s_library(pairs))):.1%}"}]), index="form")
```

The sample is 100 000 pairs whose relative gap is below $10^{-12}$. The form of
Yang and colleagues gives NaN for about a fifth of them.

As printed, Eq. 3 of that paper omits the factor $\tfrac12$ of the mean
curvature. Then Eq. 4 does not give the curvatures of a sphere:

```{code-cell} ipython3
a_ = -1.0                       # f_xx = f_yy = a, f_xy = f_x = f_y = 0: a sphere cap
H_printed, K_printed = a_ + a_, a_ * a_
root = np.sqrt(H_printed**2 - K_printed)
print(f"Eq. 3 and 4 as printed: k1, k2 = {H_printed + root:+.3f}, "
      f"{H_printed - root:+.3f}; want {a_:+.3f}, {a_:+.3f}")
```

This notebook cannot tell whether the error is in the paper only or also in
the code behind its results.

+++

### 12.2 The Hessian is not the intensity surface

The docstring describes the image as a surface "with intensities representing
heights". The principal curvatures of that surface equal the eigenvalues of
the Hessian only where the gradient is zero. Yang and colleagues compute the
curvature of that surface, with the gradient terms. Yoshida and Näppi compute
the curvature of the iso-surface. scikit-image takes the Hessian eigenvalues.
§4 shows that the Hessian shape index does not change when the intensity is
scaled. The surface shape index does:

```{code-cell} ipython3
def blob_gradient(size, width, sigma):
    """The exact gradient of the smoothed blob."""
    r, c = [x - size // 2 for x in np.indices((size, size), dtype=float)]
    spread = width**2 + sigma**2
    amp = (width**2 / spread) * np.exp(-(r**2 + c**2) / (2 * spread))
    return -amp * r / spread, -amp * c / spread


def surface_s(H, grad):
    """Shape index of the intensity surface z = f(r, c), from its K and H."""
    hrr, hrc, hcc = H
    gr, gc = grad
    w = 1 + gr**2 + gc**2
    K = (hrr * hcc - hrc**2) / w**2
    M = (hrr * (1 + gc**2) - 2 * gr * gc * hrc + hcc * (1 + gr**2)) / (2 * w**1.5)
    return -(2 / np.pi) * np.arctan2(2 * M, 2 * np.sqrt(np.maximum(M**2 - K, 0)))


_, H_blob = blob_truth(BLOB_SIZE, BLOB_WIDTH, 1.0)
grad_blob = blob_gradient(BLOB_SIZE, BLOB_WIDTH, 1.0)
blob_zone, _ = zones(BLOB_SIZE, BLOB_WIDTH, 1.0)
rows = []
for amplitude in (0.01, 1.0, 10.0, 100.0):
    H_a = [amplitude * h for h in H_blob]
    s_h = s_from(H_a)
    s_g = surface_s(H_a, [amplitude * g for g in grad_blob])
    gap = np.abs(s_g - s_h)[blob_zone]
    changed = np.digitize(s_g[blob_zone], BINS) != np.digitize(s_h[blob_zone], BINS)
    rows.append({"blob amplitude": amplitude,
                 "median |s_surface - s_Hessian|": f"{np.median(gap):.1e}",
                 "pixels changing class": f"{100 * np.mean(changed):.1f}%"})
show_table(pd.DataFrame(rows), index="blob amplitude")
```

At a blob amplitude of 1 or less the two agree to $2 \times 10^{-5}$. At an
amplitude of 10 the surface shape index moves 13% of the blob pixels to another
class. At 100 it moves 58%. So the surface shape index of one image depends on
the unit of intensity, for example a 0 to 1 scale or a 0 to 255 scale. Yang
and colleagues state that the shape index "can work on different variation
range of intensity" because it "is a scale-invariant descriptor". That holds
for the Hessian form, not for the surface form that they compute.

+++

## 13. Ways forward

+++

### 13.1 Resolve the umbilic

Replace the quotient with the two-argument arctangent of §5, and return NaN
only where both eigenvalues are zero. This changes only exact ties, and the
photograph has none. The docstring example changes from $-1$ to $+1$ at the
bright pixel.

+++

### 13.2 Move `shape_index` to Gaussian derivatives

The measured gains and costs, from §6, §7 and §11:

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"sigma": s,
      "blob error, FD / GD": f"{BLOB_ERR[s]['FD'] / BLOB_ERR[s]['GD']:.1e}",
      "camera pixels changing class": f"{100 * CLASS_CHANGE[s]:.0f}%",
      "noise spread, GD / FD (0.02)": f"{NOISE[s, 0.02]:.2f}",
      "time, GD / FD": f"{TIME_RATIO[s]:.1f}x"}
     for s in (0.5, 1.0, 2.0)]), index="sigma")
```

The change makes the shape index closer to the closed form at every `sigma`.
It also reclassifies most of a photograph at small `sigma` and passes more
noise. It costs more time from `sigma` 1; at 0.5 the timing ratio is close to 1.

+++

### 13.3 Impose the third moment on the first-derivative kernel

§9 measures a factor of about 2 in the shape index inside the blob at `sigma`
0.3 and 0.5, and little change above 0.8. The cost is one more condition per
kernel and a badly conditioned solve at `sigma` 0.3.

+++

## 14. What not to conclude

+++

### 14.1 That finite differences are more accurate at small `sigma`

An average over the whole interior of the blob says so, and §7 shows why: most
of the pixels are in the far field, below $10^{-3}$ of the peak. There the blob
is close to an exponential, and §8 proves that the finite-difference shape
index is exact on any exponential. That is a property of the exponential
tail, where the Hessian is close to rank one, not a general accuracy. §10
measures the finite-difference error on a photograph.

+++

### 14.2 That a switch of route at some `sigma` gets the best of both

§7 finds no `sigma` where finite differences are closer inside the blob. A
switch would also make the shape index jump across the switch point:

```{code-cell} ipython3
def step(low, high):
    """What a reader sees across a sigma step, switching or not."""
    inner = interior(high)
    before = s_from(fd_hessian(PHOTO, low))[inner]
    after = s_from(gd_hessian(PHOTO, high))[inner]
    unswitched = s_from(gd_hessian(PHOTO, low))[inner]
    ok = np.isfinite(before) & np.isfinite(after) & np.isfinite(unswitched)
    return {"sigma step": f"{low} to {high}",
            "median |ds|, switching route": f"{np.median(np.abs(after - before)[ok]):.3f}",
            "median |ds|, staying on one": f"{np.median(np.abs(after - unswitched)[ok]):.3f}"}


show_table(pd.DataFrame([step(0.79, 0.81), step(0.75, 0.85)]), index="sigma step")
```

Across a `sigma` step of 0.02, a switch moves the median shape index by about
11 times as much as staying on one route. Across a step of 0.1, it moves it by
about twice as much.

+++

### 14.3 That lower noise means a better estimate

§11 shows the finite-difference elements pass less noise at equal gain on a
quadratic. They also respond less to the high frequencies in real structure,
which is the error §10 measures on the photograph.

+++

## 15. What the literature says

+++

### 15.1 The measure

Koenderink and van Doorn define the shape index and the curvedness together, as
the polar angle and radius of the $(\kappa_1, \kappa_2)$ plane
[@koenderink1992surface_shape]. §4 and §5 rest on that paper.

Its first large use was on range data. Dorai and Jain describe free-form
objects by maximal patches of constant shape index and match views through a
"shape spectrum", the histogram of the index over a surface
[@dorai1997cosmosa_representation; @dorai1997shape_spectrum]. They fit a
surface over a $5 \times 5$ neighbourhood of the range image, smooth the
curvatures for 15 iterations, and label planar points with a symbol or the
value 2.0. Their index runs from 0 to 1. Worthington and Hancock bring the
index to images: they recover a field of surface normals by shape-from-shading,
take the curvatures as the eigenvalues of the matrix of derivatives of the
normals, and recognise objects from histograms of the index
[@worthington2001object_recognition]. Bonde, Badrinarayanan and Cipolla select
a characteristic scale at each point of a point cloud before they compute the
index [@bonde2013multi_scale]. §12.1 compares the forms of the index that these
papers print.

+++

### 15.2 The volumetric Hessian

Yoshida and Näppi compute the curvature of the iso-surface through each voxel
of a CT volume, from the first and second derivatives of the volume, and keep
voxels by hysteresis thresholds on the shape index and the curvedness
[@yoshida2001threedimensional_computeraided; @yoshida2002computerized_detection].
Their index is the 0 to 1 form of Dorai and Jain, defined "where
$\kappa_1 \neq \kappa_2$". So the definition leaves out the umbilic, which is
the top of the cap class they look for. The 2002 paper smooths the volume with
a 3-D Gaussian. Neither paper says how the derivatives are taken.

Gonçalves, Novo and Campilho compute the Hessian by convolution with Gaussian
second derivatives, at `sigma` from 0.5 to 3.5, and take the index from the
largest and smallest of the three eigenvalues [@goncalves2016hessian_based].
This is the setting closest to scikit-image. They do not say how the kernels
are sampled.

Yang, Han and Chen smooth each slice with an $11 \times 11$ Gaussian kernel and
compute the curvature of the intensity surface in a $7 \times 7$ neighbourhood
[@yang2017automatic_segmentation]. §12.1 and §12.2 test their formulas.

Ben-Noah, Hidalgo and Dentz do not use the shape index
[@ben_noah2024porespace_partitioning]. They use the determinant and the
eigenvalues of the Hessian of a distance map, to find maxima and saddles. Their
derivatives are central differences in the interior and one-sided differences
at the edges, which is the scheme of `np.gradient`, applied without smoothing.

Of this lineage, Gonçalves and colleagues and Ben-Noah and colleagues name
their derivative operator. None discusses an exact tie.

+++

### 15.3 The discretisation

Lindeberg's discrete analogue of the Gaussian and its central-difference
derivatives are from 1990 and 1993 [@lindeberg1990scalespace_discrete;
@lindeberg1993discrete_derivative]. The 2024 paper compares sampled, integrated and
discrete-analogue kernels with the continuous theory
[@lindeberg2024discrete_gaussian]. Its abstract puts the limit for sampled Gaussian
derivatives at a standard deviation of about 0.75, below which their estimates
"are, however, not numerically accurate or consistent". The paper measures the
responses of each kernel to monomials, but it does not impose them. So the
moment-corrected kernels of §3 are not among the discretisations it compares.

The 2025 paper defines "hybrid" discretisations: a normalised sampled Gaussian
followed by central differences [@lindeberg2025approximation_properties]. Its
second-order operator is $\delta_{xx} = (1, -2, 1)$, with $\delta_x$ for odd
orders. The finite-difference route of §3 is a hybrid in that sense, with one
difference: it applies $\delta_x$ twice for the diagonal elements. That is the
shared operator that §8 shows to be exact on exponentials. Lindeberg's hybrid
pairs $(1, -2, 1)$ on the diagonals with $\delta_x \delta_y$ on the mixed
element, which is the pair whose determinant §8 shows to be negative.

Both papers use the mixed derivative inside the determinant of the Hessian and
the ridge strength $L_{pp}$. Their test patterns are a Gaussian blob, measured
at its centre, and a ridge aligned with an axis. At both, $L_{xy} = 0$. This is
a reading of the test design, not a result of the papers: the experiments do
not test whether the mixed and pure second-derivative discretisations agree.

Farid and Simoncelli design the prefilter and the derivative filters together,
so that directional derivatives are rotation invariant
[@farid2004differentiation_discrete]. For second order, the directional derivative
combines $D_{xx}$, $D_{xy}$ and $D_{yy}$, so the constraint couples the pure
and mixed filters. That coupling is the property the shape index needs; the
reading that it is the same property is this notebook's, not theirs.

Rey-Otero and Delbracio compare three discrete versions of Gaussian smoothing
at low blur, including one computed through the discrete Fourier transform
[@rey_otero2016computing_exact]. DIPlib offers Gaussian filters by direct
convolution, recursive filtering and the Fourier transform
[@diplib_linear_filters]. The reference implementation of the discrete
analogue is Lindeberg's own toolbox [@lindeberg_pyscsp].

Lindeberg's $\gamma$-normalised derivatives select a scale by the maximum of a
normalised response over $\sigma$ [@lindeberg1998feature_detection;
@lindeberg1998edge_detection]. Majer shows that the choice of $\gamma$
decides whether a second-derivative filter detects ridges or edges
[@majer2004influence_scale]. The shape index discards the
magnitude that normalisation acts on, so scale selection for it has to use the
curvedness.

+++

## 16. Summary

| question | answer |
| --- | --- |
| how much does the shape index move | 71% of photograph pixels change class at `sigma` 0.5, 46% at 1, 16% at 2, 5% at 4 (§6) |
| which is closer to a closed form, inside the blob | Gaussian derivatives at every `sigma` tested, from 0.3 to 4; also at every blob width from 2 to 16, tested at `sigma` 0.3 and 0.5 (§7) |
| where finite differences are closer | in the far field, where the blob's Hessian is close to rank one (below $10^{-3}$ of the peak), at `sigma` 0.3 and 0.5 (§7) |
| why | the far field is close to an exponential, and finite differences give the exact shape index of any exponential (§8) |
| overall Hessian error, relative RMS by the Frobenius norm | FD about 2% on the blob at every `sigma` from 0.3 to 4, 8% to 18% on the stripes; GD 0.6% at `sigma` 0.3, $10^{-6}$ at 1, $10^{-13}$ from 2; needs no floor (§7.3) |
| where the finite-difference error comes from inside the blob | mostly the diagonal elements, from the doubled central difference (§8) |
| what the formula does at an umbilic | where the eigenvalues tie exactly it gives the wrong end of the scale, in the library docstring example too (§5) |
| the fix for the umbilic | $s = -(2/\pi)\,\operatorname{atan2}(\lambda_1 + \lambda_2, \lambda_1 - \lambda_2)$, NaN only at the planar point; no change on the photograph (§5) |
| does a third-moment mixed kernel help | about 2 times inside the blob at `sigma` 0.3 and 0.5, little above 0.8 (§9) |
| reasons to keep finite differences | less noise per unit gain, lower cost, and reproduction of earlier results (§11) |
| other libraries | `pycurv` reverses the denominator and raises at an umbilic; FreeSurfer omits $2/\pi$ and returns 0 at an umbilic (§12) |
| published forms | of four printed forms, only Koenderink's, which scikit-image uses, fails at a tie; Yang's form gives NaN at near ties (§12.1) |
| Hessian or intensity surface | the Hessian shape index is invariant to intensity scale; the surface shape index is not (§12.2) |

+++

### Limits

One photograph (`camera`, decimated to 256 by 256) and one family
of isotropic Gaussian blobs, at `sigma` from 0.3 to 8, in two dimensions only.
The split at $10^{-3}$ of the peak is a choice about shape (§7.1); §7.2
shows how the result moves with it. The closed form exists only for the blob, so the photograph has no
truth, and §10 uses the Gaussian-derivative route as its reference. The noise
figures are 12 draws at two amplitudes on pixels that carry the top 70% of
curvature. The timings are the best of three runs on one machine. `float32`
input and `mode` other than `constant` are not tested. The Gaussian-derivative
route is a transcription, because the installed scikit-image does not have the
branch. The sign convention of `pycurv` is not tested. §12.1 evaluates the printed
formulas of the papers, not their code.

```{code-cell} ipython3
installed_is_fd = np.array_equal(
    np.nan_to_num(shape_index(PHOTO, sigma=1.0), nan=-9),
    np.nan_to_num(s_library(fd_hessian(PHOTO, 1.0)), nan=-9))

print(f"scikit-image  {ski.__version__}")
print(f"scipy         {__import__('scipy').__version__}")
print(f"opencv        {cv2.__version__}")
print("installed shape_index route: "
      f"{'finite differences' if installed_is_fd else 'Gaussian derivatives'}")
```
