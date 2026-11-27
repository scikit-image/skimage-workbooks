---
title: 'On second-order Gaussian taps'
date: 2026-11-21
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

:::{related}
https://dsp.stackexchange.com/questions/78280/are-scipy-second-order-gaussian-derivatives-correct | Cris Luengo's mean-subtract recipe; compared in [§4](#sec-diplib)
https://github.com/scikit-image/scikit-image/issues/6451 | Hessian Gaussian-derivative scaling; discrete moments in [§3](#sec-problem)
https://diplib.org/diplib-docs/linear.html | DIPlib `Gauss` / `GaussFIR`; recipe in [§4](#sec-diplib)
:::


:::{changelog}
2026-11-21 | cursor:composer | Create workbook; fold citations into library.bib and misc.bib
:::

A sampled second-order Gaussian kernel on a finite grid does not keep the two
discrete moments that a continuous $G''$ has. Two repairs both restore
$\sum k = 0$ and $\sum k\,x^{2} = 2$, but they disagree at moderate $\sigma$
when the window is wide. Lindeberg's discrete analogue plus
$\delta_{xx} = (1,-2,1)$ is a third construction: it does not sample $G''$
at all. This notebook settles where the raw sample fails, how the two
repairs differ, that sampling continuous $G''$ matches the discrete-$k_0$
route after the same repair, how the discrete-analogue kernel compares to
the repaired sample, and which route tracks continuous $G''$ more closely.

The coordinate $x$ is an integer offset from the centre tap, in samples.
Kernels are 1-D. The default window is SciPy-style
`radius = int(truncate * sigma + 0.5)` with `truncate = 8`. For the discrete
analogue, Lindeberg's scale parameter is the variance $s$; this notebook sets
$s = \sigma^{2}$ so that $\sigma$ matches the continuous standard deviation
used in his plots.

## 1. Setup

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import iv
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette; validated all-pairs
# in the workbook set (light mode; every series is also labelled).
C_ONE = "#2a78d6"
C_TWO = "#eb6834"
C_THREE = "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd5"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.titlecolor": MUTED,
        "figure.facecolor": "white",
    }
)
```

```{code-cell} ipython3
TRUNCATE = 8
SIGMAS_TABLE = (0.1, 0.5, 1.0, 2.0)
SIGMAS_GRID = np.linspace(0.1, 3.0, 30)
```

## 2. Helpers

```{code-cell} ipython3
def grid(sigma, truncate=TRUNCATE):
    """Integer offsets and a discrete-normalized Gaussian $k_0$.

    Discrete ``sum`` sets $\\sum k_0 = 1$; the continuous $1/(\\sigma\\sqrt{2\\pi})$
    factor cancels under that normalisation.
    """
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    k0 = np.exp(-(x**2) / (2 * sigma**2))
    k0 /= k0.sum()
    return x, k0


def raw_k2(sigma, truncate=TRUNCATE):
    """Sample $k_0 \\cdot (x^2 - \\sigma^2) / \\sigma^4$ with no moment repair."""
    x, k0 = grid(sigma, truncate)
    k2 = k0 * ((x**2 - sigma**2) / sigma**4)
    return x, k0, k2


def taps_centre(sigma, truncate=TRUNCATE):
    """Sum-zero along $k_0$, scale $\\sum k x^2 = 2$, rebuild the centre tap."""
    x, k0, k2 = raw_k2(sigma, truncate)
    k2 = k2 - k2.sum() * k0  # sum(k) == 0
    center = len(k2) // 2
    k2 /= (k2 * x**2).sum() / 2  # sum(k * x**2) == 2
    k2[center] = -(k2[:center].sum() + k2[center + 1 :].sum())
    return x, k2


def taps_diplib(sigma, truncate=TRUNCATE):
    """DIPlib-style: subtract the mean, then scale $\\sum k x^2 = 2$."""
    x, _, k2 = raw_k2(sigma, truncate)
    k2 = k2 - k2.mean()
    k2 /= (k2 * x**2).sum() / 2
    return x, k2


def taps_double_rebuild(sigma, truncate=TRUNCATE):
    """Sum-zero, rebuild, scale, rebuild again (first rebuild redundant)."""
    x, k0, k2 = raw_k2(sigma, truncate)
    k2 = k2 - k2.sum() * k0  # sum(k) == 0
    center = len(k2) // 2
    k2[center] = -(k2[:center].sum() + k2[center + 1 :].sum())
    k2 /= (k2 * x**2).sum() / 2  # sum(k * x**2) == 2
    k2[center] = -(k2[:center].sum() + k2[center + 1 :].sum())
    return x, k2


def taps_from_Gpp(sigma, truncate=TRUNCATE):
    """Sample continuous $G''$, then the same centre-rebuild repair."""
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    G = np.exp(-(x**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
    k2 = G * ((x**2 - sigma**2) / sigma**4)
    g = G / G.sum()
    k2 = k2 - k2.sum() * g  # sum(k) == 0
    center = len(k2) // 2
    k2 /= (k2 * x**2).sum() / 2  # sum(k * x**2) == 2
    k2[center] = -(k2[:center].sum() + k2[center + 1 :].sum())
    return x, k2


def moments(k, x):
    """Return $(\\sum k,\\, \\sum k x^2 / 2)$."""
    return float(k.sum()), float((k * x**2).sum() / 2)


def taps_disc_xx(sigma, truncate=TRUNCATE):
    """Discrete analogue T_disc, then delta_xx = (1, -2, 1). Returns (x, T, k)."""
    s = sigma**2
    radius = max(int(truncate * sigma + 0.5), 1)
    n = np.arange(-radius, radius + 1, dtype=float)
    # T_disc(n; s) = exp(-s) * I_|n|(s), with s = sigma**2.
    T = np.exp(-s) * iv(np.abs(n), s)
    Tpad = np.pad(T, 1)
    k = Tpad[:-2] - 2 * Tpad[1:-1] + Tpad[2:]
    return n, T, k


def kernel_response(k, x, f):
    """Response at the origin for an even or odd test function f."""
    return float(np.sum(k * f(x)))


def continuous_Gpp_cos(sigma, omega):
    """Continuous $G'' * \\cos(\\omega x)$ at $x=0$: $-\\omega^2 e^{-\\sigma^2 \\omega^2 / 2}$."""
    return float(-(omega**2) * np.exp(-0.5 * sigma**2 * omega**2))
```

(sec-problem)=
## 3. The problem

A continuous second derivative of a Gaussian has zero mass and a fixed second
moment. On a short grid those identities fail. At $\sigma = 0.1$ with
`truncate = 8` the support is three taps; the raw sample is almost a single
negative spike.

```{code-cell} ipython3
x, k0, k2_raw = raw_k2(0.1)
pd.DataFrame(
    {
        "x": x.astype(int),
        "k0": k0,
        "raw k2": k2_raw,
    }
)
```

```{code-cell} ipython3
pd.DataFrame(
    [
        {"kernel": "k0", "sum": k0.sum(), "sum k x^2 / 2": (k0 * x**2).sum() / 2},
        {"kernel": "raw k2", "sum": k2_raw.sum(), "sum k x^2 / 2": (k2_raw * x**2).sum() / 2},
    ]
)
```

The centre tap of `raw k2` is about $-100$. The wing taps are on the order of
$10^{-18}$. So $\sum k$ equals the centre tap to machine precision:

```{code-cell} ipython3
center = len(k2_raw) // 2
float(k2_raw.sum() - k2_raw[center])
```

$k_0$ has centre tap $1$ and the same tiny wings. The product
$(\sum k_2)\,k_0$ therefore puts almost the whole mass of $\sum k_2$ on the
centre:

```{code-cell} ipython3
pd.DataFrame({"x": x.astype(int), "(sum k2) * k0": k2_raw.sum() * k0})
```

The sum-zero step $k \leftarrow k - (\sum k)\,k_0$ then zeros the centre
while the wings stay tiny and positive. After the second-moment scale the
wings are $\sim 1$ and the centre is still $0$, so $\sum k$ is wrong; rebuild
the centre from the wings so $\sum k = 0$. The result is $[1, -2, 1]$:

```{code-cell} ipython3
_, k2_fixed = taps_centre(0.1)
pd.DataFrame(
    {
        "x": x.astype(int),
        "repaired": k2_fixed,
        "sum": k2_fixed.sum(),
        "sum k x^2 / 2": (k2_fixed * x**2).sum() / 2,
    }
)
```

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
for ax in (ax0, ax1):
    ax.axhline(0, color=GRID, lw=0.8)
ax0.plot(x, k2_raw, "o-", color=C_TWO, ms=5)
ax0.set_title(r"raw sample")
ax0.set_ylabel("tap weight")
ax1.plot(x, k2_fixed, "s-", color=C_ONE, ms=5)
ax1.set_title(r"centre rebuild")
for ax in (ax0, ax1):
    ax.set_xlabel("x")
fig.suptitle(r"$\sigma = 0.1$: raw $k_2$ vs repaired", y=1.02)
fig.tight_layout()
```

Lindeberg [@lindeberg2024discrete_approximations] compares three
discretisations of Gaussian derivatives: sampling $G^{(\alpha)}$, integrating
$G^{(\alpha)}$ over each pixel, and the discrete analogue of the Gaussian
followed by small-support central differences. For sampled second-order
kernels he finds numerically poor derivative estimates when $\sigma$ is a bit
below about $0.75$, and good agreement with the continuous theory when
$\sigma$ is greater than about $1$. At fine $\sigma$ he prefers the discrete
analogue plus central differences, not a repaired sample. Sections 3–8 study
two moment repairs on the sampling route. Section 9 compares those repairs to
the discrete-analogue operator. The same sampling defect shows up as a
non-zero response to a constant strip in [@dspse78280] and in the Hessian
discussion around [@skimage6451].

(sec-diplib)=
## 4. Two repairs: centre rebuild and DIPlib mean

DIPlib [@diplib_gauss] samples $G''$, subtracts the mean of the taps, then
scales so $\sum k\,x^{2} = 2$ [@dspse78280]. The centre-rebuild route
subtracts a multiple of $k_0$ instead of a constant, then rebuilds only the
centre tap after scaling.

```{code-cell} ipython3
rows = []
for s in SIGMAS_TABLE:
    x, kc = taps_centre(s)
    _, kd = taps_diplib(s)
    sc, mc = moments(kc, x)
    sd, md = moments(kd, x)
    rows.append(
        {
            "sigma": s,
            "n": len(x),
            "centre sum": sc,
            "diplib sum": sd,
            "centre mom": mc,
            "diplib mom": md,
            "max |diff|": float(np.max(np.abs(kc - kd))),
            "centre tap": float(kc[len(kc) // 2]),
            "diplib centre tap": float(kd[len(kd) // 2]),
        }
    )
pd.DataFrame(rows)
```

At $\sigma = 0.1$ both kernels are $[1, -2, 1]$ and `max |diff|` is $0$. At
$\sigma = 0.5$ they diverge. At $\sigma = 2$ they agree again to floating
point on this truncate; at $\sigma = 1$ the difference is below $10^{-6}$.

```{code-cell} ipython3
x, kc = taps_centre(0.5)
_, kd = taps_diplib(0.5)
pd.DataFrame({"x": x.astype(int), "centre rebuild": kc, "DIPlib mean": kd})
```

The far DIPlib wings at $\sigma = 0.5$ stay near the value printed below; the
centre-rebuild wings have already decayed with $k_0$.

```{code-cell} ipython3
wing = int(x[0])
pd.DataFrame(
    [
        {
            "|x|": abs(wing),
            "centre rebuild": float(kc[0]),
            "DIPlib mean": float(kd[0]),
        }
    ]
)
```

```{code-cell} ipython3
sigmas = SIGMAS_TABLE
fig, axes = plt.subplots(1, len(sigmas), figsize=(10.5, 2.8), sharey=False)
for ax, s in zip(axes, sigmas):
    x, kc = taps_centre(s)
    _, kd = taps_diplib(s)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.plot(x, kc, "o-", color=C_ONE, ms=3.5, label="centre rebuild")
    ax.plot(x, kd, "s--", color=C_TWO, ms=3.5, label="DIPlib mean")
    ax.set_title(rf"$\sigma = {s}$")
    ax.set_xlabel("x")
axes[0].set_ylabel("tap weight")
axes[0].legend(fontsize=8, loc="best")
fig.suptitle("Order-2 kernels: centre rebuild vs DIPlib mean subtract", y=1.05)
fig.tight_layout()
```

```{code-cell} ipython3
x, kc = taps_centre(0.5)
_, kd = taps_diplib(0.5)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.5, 3.0))
ax0.axhline(0, color=GRID, lw=0.8)
ax0.plot(x, kc, "o-", color=C_ONE, ms=5, label="centre rebuild")
ax0.plot(x, kd, "s--", color=C_TWO, ms=5, label="DIPlib mean")
ax0.set_title(r"$\sigma = 0.5$: taps")
ax0.set_xlabel("x")
ax0.set_ylabel("tap weight")
ax0.legend(fontsize=8)
ax1.semilogy(x, np.maximum(np.abs(kc), 1e-20), "o-", color=C_ONE, ms=5, label="centre rebuild")
ax1.semilogy(x, np.maximum(np.abs(kd), 1e-20), "s--", color=C_TWO, ms=5, label="DIPlib mean")
ax1.set_title(r"$\sigma = 0.5$: $|\mathrm{tap}|$")
ax1.set_xlabel("x")
ax1.set_ylabel("|tap weight|")
ax1.legend(fontsize=8)
fig.tight_layout()
```

## 5. How often, and how big

Corpus: thirty values of $\sigma$ in $[0.1, 3]$, `truncate = 8`, one 1-D
kernel per $\sigma$. The score is $\max |k_{\mathrm{centre}} - k_{\mathrm{DIPlib}}|$.

```{code-cell} ipython3
diffs = []
for s in SIGMAS_GRID:
    x, kc = taps_centre(s)
    _, kd = taps_diplib(s)
    diffs.append(float(np.max(np.abs(kc - kd))))
diffs = np.asarray(diffs)
pd.DataFrame(
    [
        {
            "n sigma": len(SIGMAS_GRID),
            "max |diff|": diffs.max(),
            "sigma at max": float(SIGMAS_GRID[int(np.argmax(diffs))]),
            "share with |diff| > 1e-6": float(np.mean(diffs > 1e-6)),
            "share with |diff| > 1e-3": float(np.mean(diffs > 1e-3)),
        }
    ]
)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.semilogy(SIGMAS_GRID, np.maximum(diffs, 1e-20), "o-", color=C_ONE, ms=3)
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel(r"max $|k_{\mathrm{centre}} - k_{\mathrm{DIPlib}}|$")
ax.set_title("Kernel disagreement vs sigma (truncate = 8)")
fig.tight_layout()
```

DIPlib's own windows are shorter ($\sim 3\sigma$ to $4\sigma$), so the mean
pedestal is smaller there. With `truncate = 100` at $\sigma = 0.5$ the same
$\sum k$ residue spreads over many near-zero taps:

```{code-cell} ipython3
x100, kc100 = taps_centre(0.5, truncate=100)
_, kd100 = taps_diplib(0.5, truncate=100)
pd.DataFrame(
    [
        {
            "truncate": 100,
            "n": len(x100),
            "max |diff|": float(np.max(np.abs(kc100 - kd100))),
            "centre wing at x min": float(kc100[0]),
            "DIPlib wing at x min": float(kd100[0]),
        }
    ]
)
```

## 6. Properties that ought to hold

Both repairs promise $\sum k = 0$, $\sum k\,x^{2}/2 = 1$, and even symmetry.
Check the thirty-$\sigma$ corpus for each route.

```{code-cell} ipython3
prop_rows = []
for name, fn in (("centre", taps_centre), ("DIPlib", taps_diplib)):
    sums, moms, odd = [], [], []
    for s in SIGMAS_GRID:
        x, k = fn(s)
        sums.append(abs(k.sum()))
        moms.append(abs((k * x**2).sum() / 2 - 1.0))
        odd.append(np.max(np.abs(k - k[::-1])))
    prop_rows.append(
        {
            "route": name,
            "max |sum|": max(sums),
            "max |mom - 1|": max(moms),
            "max |k - flip(k)|": max(odd),
        }
    )
pd.DataFrame(prop_rows)
```

## 7. Sampled $G''$ as comparator

The centre-rebuild helper starts from a discrete-normalized $k_0$ times the
analytic factor $(x^{2}-\sigma^{2})/\sigma^{4}$. Sampling the continuous
second derivative uses the same factor on the analytically normalized
Gaussian:

$$
G''(x)
  = \frac{1}{\sigma\sqrt{2\pi}}\,e^{-x^2/(2\sigma^2)}
    \cdot\frac{x^2-\sigma^2}{\sigma^4}.
$$

Before repair the two samples differ by normalization. After the same
centre-rebuild repair they agree.

```{code-cell} ipython3
rows = []
for s in SIGMAS_TABLE:
    x, kk = taps_centre(s)
    _, kg = taps_from_Gpp(s)
    raw = np.exp(-(x**2) / (2 * s**2))
    Z = np.sqrt(2 * np.pi) * s
    # Unrepaired continuous sample (no moment fix).
    G = raw / Z
    k_unrepaired = G * ((x**2 - s**2) / s**4)
    rows.append(
        {
            "sigma": s,
            "sum(raw) / Z": float(raw.sum() / Z),
            "unrepaired G'' sum": float(k_unrepaired.sum()),
            "max |k0 - G''| after repair": float(np.max(np.abs(kk - kg))),
            "repaired k0 sum": float(kk.sum()),
            "repaired G'' sum": float(kg.sum()),
        }
    )
pd.DataFrame(rows)
```

Unrepaired $|\sum G''|$ is large at $\sigma = 0.1$ and $\sigma = 0.5$ on this
truncate. At $\sigma = 2$ it is already near zero. The repair is still the
safe default: the same code path must work at fine $\sigma$.

```{code-cell} ipython3
gpp_diffs = np.array(
    [
        float(np.max(np.abs(taps_centre(s)[1] - taps_from_Gpp(s)[1])))
        for s in SIGMAS_GRID
    ]
)
pd.DataFrame(
    [
        {
            "n sigma": len(SIGMAS_GRID),
            "max |k0 - G''| after repair": gpp_diffs.max(),
            "sigma at max": float(SIGMAS_GRID[int(np.argmax(gpp_diffs))]),
        }
    ]
)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.semilogy(SIGMAS_GRID, np.maximum(gpp_diffs, 1e-20), "o-", color=C_ONE, ms=3)
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel(r"max $|k_{k_0} - k_{G''}|$ after repair")
ax.set_title(r"Disagreement after the same repair (truncate $= 8$)")
fig.tight_layout()
```

```{code-cell} ipython3
sigmas = SIGMAS_TABLE
fig, axes = plt.subplots(1, len(sigmas), figsize=(10.5, 2.8), sharey=False)
for ax, s in zip(axes, sigmas):
    x, kk = taps_centre(s)
    _, kg = taps_from_Gpp(s)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.plot(x, kk, "o-", color=C_ONE, ms=3.5, label=r"from discrete $k_0$")
    ax.plot(x, kg, "s--", color=C_TWO, ms=3.5, label=r"sample $G''$")
    ax.set_title(rf"$\sigma = {s}$")
    ax.set_xlabel("x")
axes[0].set_ylabel("tap weight")
axes[0].legend(fontsize=8, loc="best")
fig.suptitle(r"Moment-repaired kernels: discrete $k_0$ route vs sampled $G''$", y=1.05)
fig.tight_layout()
```

On this grid the max disagreement after repair is on the order of
$10^{-16}$. Either starting form still needs the repair at fine $\sigma$.

## 8. Why one centre rebuild is enough

After the sum-zero step $k \leftarrow k - (\sum k)\,k_0$, a pipeline can
rebuild the centre so $\sum k = 0$, scale the second moment, then rebuild the
centre again. The first rebuild is redundant: the finished kernel depends
only on the wings.

Write the taps as a centre value $c$ (at $x = 0$) and wing values $w$ (all
taps with $x \neq 0$). The two discrete moments split as

$$
\sum k = c + \sum w,
\qquad
\sum k\,x^{2} = \sum w\,x^{2},
$$

because $x = 0$ at the centre, so $c$ never enters the second moment.

**Scale then rebuild:**

$$
\alpha = \frac{2}{\sum w\,x^{2}},
\qquad
\tilde{w} = \alpha w,
\qquad
\tilde{c} = -\sum \tilde{w}.
$$

Then $\sum \tilde{w}\,x^{2} = 2$ and $\sum(\tilde{c} + \tilde{w}) = 0$ by
construction. The output $(\tilde{c}, \tilde{w})$ is a function of $w$ alone.

**Rebuild, scale, rebuild:** first set $c' = -\sum w$, then apply the same
$\alpha$ (still independent of $c'$), then rebuild again. Same $\alpha$, same
$\tilde{w}$, same $\tilde{c}$.

```{code-cell} ipython3
rebuild_diffs = np.array(
    [
        float(np.max(np.abs(taps_centre(s)[1] - taps_double_rebuild(s)[1])))
        for s in SIGMAS_GRID
    ]
)
pd.DataFrame(
    [
        {
            "n sigma": len(SIGMAS_GRID),
            "max |single - double rebuild|": rebuild_diffs.max(),
        }
    ]
)
```

On this corpus the two pipelines match to floating point. The first rebuild
only replaces a centre that the scale ignores and that the second rebuild
overwrites.

(sec-disc)=
## 9. Discrete analogue plus $\delta_{xx}$

The constructions above all start from a sample of a continuous second
derivative (or from $k_0$ times that analytic factor) and then repair
moments. Lindeberg [@lindeberg2024discrete_approximations] uses a different
operator. Smooth with the discrete analogue of the Gaussian

$$
T_{\mathrm{disc}}(n; s) = e^{-s}\,I_{|n|}(s),
\qquad s = \sigma^{2},
$$

then apply the central second difference $\delta_{xx} = (1,-2,1)$ to the
smoothed signal. The equivalent taps are
$T_{\mathrm{disc},xx} = \delta_{xx} * T_{\mathrm{disc}}$. That kernel is not
a repaired sample of $G''$.

```{code-cell} ipython3
rows = []
for s in SIGMAS_TABLE:
    x, kc = taps_centre(s)
    n, T, kd = taps_disc_xx(s)
    sc, mc = moments(kc, x)
    sd, md = moments(kd, n)
    rows.append(
        {
            "sigma": s,
            "n": len(x),
            "sum T_disc": float(T.sum()),
            "centre sum": sc,
            "disc sum": sd,
            "centre mom": mc,
            "disc mom": md,
            "max |centre - disc|": float(np.max(np.abs(kc - kd))),
            "centre tap": float(kc[len(kc) // 2]),
            "disc centre tap": float(kd[len(kd) // 2]),
        }
    )
pd.DataFrame(rows)
```

At $\sigma = 0.1$ both kernels sit near $(1,-2,1)$, but they are not equal:
centre rebuild forces exact $[1,-2,1]$, while $\delta_{xx}*T_{\mathrm{disc}}$
keeps the truncated Bessel mass. At $\sigma = 0.5$ the centre taps differ by
about $0.55$ on this truncate. The disc route does not enforce
$\sum k = 0$ by a separate step; a small $\sum k$ residue remains at fine
$\sigma$ from truncating $T_{\mathrm{disc}}$.

```{code-cell} ipython3
# Response to monomials at the origin (even kernels: sum k(n) f(n)).
rows = []
for s in SIGMAS_TABLE:
    x, kc = taps_centre(s)
    n, _, kd = taps_disc_xx(s)
    rows.append(
        {
            "sigma": s,
            "centre on 1": float(kc.sum()),
            "disc on 1": float(kd.sum()),
            "centre on x": float((kc * x).sum()),
            "disc on x": float((kd * n).sum()),
            "centre on x^2": float((kc * x**2).sum()),
            "disc on x^2": float((kd * n**2).sum()),
        }
    )
pd.DataFrame(rows)
```

Lindeberg's monomial test asks for response $2$ on $x^{2}$ and $0$ on
constants. Centre rebuild meets both by construction. The disc route meets
them up to truncation of $T_{\mathrm{disc}}$: on this truncate the $x^{2}$
response is already within about $2\%$ at $\sigma = 0.1$ and within
$4\cdot 10^{-4}$ at $\sigma = 0.5$.

```{code-cell} ipython3
sigmas = SIGMAS_TABLE
fig, axes = plt.subplots(1, len(sigmas), figsize=(10.5, 2.8), sharey=False)
for ax, s in zip(axes, sigmas):
    x, kc = taps_centre(s)
    n, _, kd = taps_disc_xx(s)
    _, km = taps_diplib(s)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.plot(x, kc, "o-", color=C_ONE, ms=3.5, label="centre rebuild")
    ax.plot(n, kd, "s--", color=C_THREE, ms=3.5, label=r"disc + $\delta_{xx}$")
    ax.plot(x, km, "^:", color=C_TWO, ms=3.5, label="DIPlib mean")
    ax.set_title(rf"$\sigma = {s}$")
    ax.set_xlabel("x")
axes[0].set_ylabel("tap weight")
axes[0].legend(fontsize=7, loc="best")
fig.suptitle(r"Repaired samples vs discrete analogue + $\delta_{xx}$", y=1.05)
fig.tight_layout()
```

```{code-cell} ipython3
disc_diffs = np.array(
    [
        float(np.max(np.abs(taps_centre(s)[1] - taps_disc_xx(s)[2])))
        for s in SIGMAS_GRID
    ]
)
pd.DataFrame(
    [
        {
            "n sigma": len(SIGMAS_GRID),
            "max |centre - disc|": disc_diffs.max(),
            "sigma at max": float(SIGMAS_GRID[int(np.argmax(disc_diffs))]),
            "share with |diff| > 1e-3": float(np.mean(disc_diffs > 1e-3)),
        }
    ]
)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.0, 3.0))
ax.semilogy(SIGMAS_GRID, np.maximum(disc_diffs, 1e-20), "o-", color=C_THREE, ms=3)
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel(r"max $|k_{\mathrm{centre}} - k_{\mathrm{disc}}|$")
ax.set_title(r"Centre rebuild vs disc + $\delta_{xx}$ (truncate $= 8$)")
fig.tight_layout()
```

On this thirty-$\sigma$ corpus the peak disagreement and the share above
$10^{-3}$ are the values in the table above. The discrete-analogue route is
therefore a different operator from either moment repair, not a rephrasing of
them. DIPlib and the centre-rebuild path stay on the sampling route;
Lindeberg prefers the disc route at fine $\sigma$
[@lindeberg2024discrete_approximations].

(sec-accuracy)=
## 10. Accuracy against continuous $G''$

Two contracts differ. Lindeberg's monomial test asks an order-$2$ operator for
response $2$ on $x^{2}$ and $0$ on $1$ and $x$
[@lindeberg2024discrete_approximations]. Continuous scale-space asks the
kernel to match $G'' * f$ for smooth $f$. Centre rebuild meets the monomial
contract by construction. The cosine

$$
(G'' * \cos(\omega\,\cdot\,))(0)
  = -\omega^{2}\,e^{-\sigma^{2}\omega^{2}/2}
$$

tests the continuous contract. Corpus: $\sigma$ on the thirty-point grid,
$\omega \in \{0.2, 0.5, 1.0\}$, `truncate = 8`. Relative error uses
denominator $\max(|target|, 10^{-12})$.

```{code-cell} ipython3
def route_kernel(name, sigma):
    """Return (x, k) for a named second-order route."""
    if name == "centre":
        return taps_centre(sigma)
    if name == "DIPlib":
        return taps_diplib(sigma)
    if name == "disc":
        x, _, k = taps_disc_xx(sigma)
        return x, k
    if name == "raw sample":
        x, _, k = raw_k2(sigma)
        return x, k
    raise ValueError(name)


ROUTE_NAMES = ("centre", "DIPlib", "disc", "raw sample")
OMEGAS = (0.2, 0.5, 1.0)
```

```{code-cell} ipython3
# Lindeberg monomials of order <= 2, and continuous target on x^4 at 0.
mono_rows = []
for name in ROUTE_NAMES:
    err_low, err_x2, err_x4 = [], [], []
    for s in SIGMAS_GRID:
        x, k = route_kernel(name, s)
        err_low.append(abs(kernel_response(k, x, lambda t: np.ones_like(t))))
        err_low.append(abs(kernel_response(k, x, lambda t: t)))
        err_x2.append(abs(kernel_response(k, x, lambda t: t**2) - 2.0))
        err_x4.append(abs(kernel_response(k, x, lambda t: t**4) - 12.0 * s**2))
    mono_rows.append(
        {
            "route": name,
            "max |resp on 1 or x|": max(err_low),
            "max |resp on x^2 - 2|": max(err_x2),
            "max |resp on x^4 - 12 sigma^2|": max(err_x4),
            "median |resp on x^4 - 12 sigma^2|": float(np.median(err_x4)),
        }
    )
pd.DataFrame(mono_rows)
```

Centre rebuild and DIPlib keep the order-$\le 2$ monomial errors at floating
point. The disc route is close, with a truncation residue at fine $\sigma$.
Raw sampling fails the monomial test. On $x^{4}$ the continuous target at the
origin is $12\sigma^{2}$. The disc route sits near $12\sigma^{2}+2$: the
central difference satisfies $\delta_{xx}(x^{4}) = 12x^{2}+2$, so after unit
mass smoothing the extra $+2$ remains. That gap is structural to
$\delta_{xx}$, not to truncation.

```{code-cell} ipython3
cos_rows = []
cos_rel = {}
for name in ROUTE_NAMES:
    rel = []
    for s in SIGMAS_GRID:
        x, k = route_kernel(name, s)
        for w in OMEGAS:
            tgt = continuous_Gpp_cos(s, w)
            got = kernel_response(k, x, lambda t, ww=w: np.cos(ww * t))
            rel.append(abs(got - tgt) / max(abs(tgt), 1e-12))
    cos_rel[name] = np.asarray(rel)
    cos_rows.append(
        {
            "route": name,
            "median rel err": float(np.median(rel)),
            "mean rel err": float(np.mean(rel)),
            "max rel err": float(np.max(rel)),
        }
    )
pd.DataFrame(cos_rows)
```

```{code-cell} ipython3
# Relative error vs sigma at omega = 0.5 (one slice of the corpus).
w = 0.5
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for name, color, style in (
    ("centre", C_ONE, "o-"),
    ("DIPlib", C_TWO, "s--"),
    ("disc", C_THREE, "^-"),
):
    rel = []
    for s in SIGMAS_GRID:
        x, k = route_kernel(name, s)
        tgt = continuous_Gpp_cos(s, w)
        got = kernel_response(k, x, lambda t: np.cos(w * t))
        rel.append(abs(got - tgt) / max(abs(tgt), 1e-12))
    ax.semilogy(SIGMAS_GRID, np.maximum(rel, 1e-20), style, color=color, ms=3, label=name)
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel(r"relative error vs continuous $G''*\cos$")
ax.set_title(rf"Continuous cosine accuracy ($\omega = {w}$, truncate $= 8$)")
ax.legend(fontsize=8)
fig.tight_layout()
```

Against continuous $G''$ on this cosine corpus, centre rebuild has a lower
mean relative error than DIPlib or disc. DIPlib's median is comparable to
centre rebuild, but its mean rises because of the moderate-$\sigma$ pedestal.
The disc route keeps a larger median error across the grid: $\delta_{xx}$
matches Lindeberg's discrete monomials of order $\le 2$, but it is not the
continuum second derivative. For overall agreement with continuous
scale-space $G''$, centre rebuild is the more accurate of the two on this
corpus. For Lindeberg's discrete contract at fine $\sigma$, the disc route is
the operator he recommends; centre rebuild matches the low-order monomial
numbers by a different mechanism.

## 11. Ways forward

Four runnable choices:

1. Centre rebuild after a $k_0$-shaped sum-zero step (`taps_centre`).
2. DIPlib mean subtract (`taps_diplib`).
3. Sample continuous $G''$, then the same centre rebuild (`taps_from_Gpp`).
4. Discrete analogue plus $\delta_{xx}$ (`taps_disc_xx`).

Choice 3 matches choice 1 after repair. Choice 2 matches choice 1 at fine and
coarse $\sigma$ on `truncate = 8`, and diverges in the middle of the corpus
above. Choice 4 disagrees with choice 1 on most of the corpus. Use choice 1
when the target is continuous $G''$ (section 10). Use choice 4 when the
contract is Lindeberg's discrete scale space. For a SciPy-wide window on the
sampling route, put the $\sum k$ residue in the centre (or subtract a
multiple of $k_0$) so far taps stay near zero.

Drop a first centre rebuild before scaling. Section 8 shows it does not change
the kernel.

## 12. What each choice costs

Runtime of a 1-D correlate is the same when the support length matches.
Order-2 taps are even ($k_{-i} = k_i$), so correlation and convolution
agree; the correlate-vs-convolve sign flip that matters for $g_x$ is a no-op
here. The disc route can also run as smooth-then-$\delta_{xx}$ (two passes).
The cost that matters for the sampling repairs is how many kernels change
when the sum-zero step changes.

```{code-cell} ipython3
import time
import scipy.ndimage as ndi

img = np.random.default_rng(0).random((256, 256))
x, kc = taps_centre(1.0)
_, kd = taps_diplib(1.0)
n, T, kdisc = taps_disc_xx(1.0)
delta_xx = np.array([1.0, -2.0, 1.0])


def timed(k, nrep=30):
    t0 = time.perf_counter()
    for _ in range(nrep):
        ndi.correlate1d(img, k, axis=0, mode="reflect")
    return (time.perf_counter() - t0) / nrep


def timed_two_pass(nrep=30):
    t0 = time.perf_counter()
    for _ in range(nrep):
        sm = ndi.correlate1d(img, T, axis=0, mode="reflect")
        ndi.correlate1d(sm, delta_xx, axis=0, mode="reflect")
    return (time.perf_counter() - t0) / nrep


pd.DataFrame(
    [
        {
            "route": "centre rebuild",
            "n taps": len(kc),
            "seconds / call": timed(kc),
        },
        {
            "route": "DIPlib mean",
            "n taps": len(kd),
            "seconds / call": timed(kd),
        },
        {
            "route": "disc combined kernel",
            "n taps": len(kdisc),
            "seconds / call": timed(kdisc),
        },
        {
            "route": "disc smooth then delta_xx",
            "n taps": f"{len(T)}+3",
            "seconds / call": timed_two_pass(),
        },
    ]
)
```

```{code-cell} ipython3
pd.DataFrame(
    [
        {
            "corpus": "30 sigma in [0.1, 3], truncate=8",
            "share centre vs DIPlib |diff| > 1e-6": float(np.mean(diffs > 1e-6)),
            "share centre vs G'' |diff| > 1e-12": float(np.mean(gpp_diffs > 1e-12)),
            "share centre vs disc |diff| > 1e-3": float(np.mean(disc_diffs > 1e-3)),
            "share single vs double rebuild |diff| > 0": float(np.mean(rebuild_diffs > 0)),
        }
    ]
)
```

## 13. What not to do

**Trust an unrepaired sample of $G''$.** Section 7 shows
`unrepaired G'' sum` far from zero at $\sigma = 0.1$ and $0.5$. A constant
image then gets a non-zero second-derivative response.

**Use mean subtract with a very wide truncate.** Section 5 shows
`truncate = 100` at $\sigma = 0.5$: the DIPlib wing at the window edge is
about $2\cdot 10^{-5}$ while the centre-rebuild wing is $0$.

**Keep a first centre rebuild before scaling.** Section 8 shows it is an
identity on the finished kernel; it only adds a write.

**Treat disc + $\delta_{xx}$ as the same as a repaired sample.** Section 9
measures the kernel gap; section 10 measures the continuous-accuracy gap.

**Pick disc + $\delta_{xx}$ for continuous $G''$ accuracy.** Section 10 shows
centre rebuild tracks the analytic cosine response more closely on this
corpus.

## 14. Summary

| question | answer | section |
| --- | --- | --- |
| Does a raw sample keep $\sum k = 0$ and $\sum k x^2 = 2$? | No at fine $\sigma$; at $\sigma = 0.1$ the raw sum is the centre tap | §3 |
| Do centre rebuild and DIPlib mean both restore the moments? | Yes, on the thirty-$\sigma$ corpus | §6 |
| When do the two repairs disagree? | Peak near moderate $\sigma$ on `truncate = 8`; wider truncate spreads a mean pedestal | §4, §5 |
| Does sampled $G''$ match the $k_0$ route after repair? | Yes: max $|diff|$ on the order of $10^{-16}$ over 30 $\sigma$ | §7 |
| Is a first centre rebuild needed? | No: max $|single - double| = 0$ on the corpus | §8 |
| Is disc + $\delta_{xx}$ the same as centre rebuild? | No: peak $\max\|diff\| \approx 0.55$ near $\sigma = 0.5$; $>90\%$ of the grid above $10^{-3}$ | §9 |
| Which tracks continuous $G''$ more closely? | Centre rebuild (lower mean relative cosine error than disc or DIPlib on this corpus) | §10 |

### Limits

- Corpus: 1-D kernels only; $\sigma \in [0.1, 3]$; default `truncate = 8`,
  plus one `truncate = 100` check at $\sigma = 0.5$.
- Continuous accuracy uses three cosines
  $\omega \in \{0.2, 0.5, 1.0\}$ and monomials through degree $4$; other
  signals were not tested.
- Discrete-analogue comparison uses $s = \sigma^{2}$ and the same truncate
  rule as the sampled kernels; Lindeberg's own windows may differ.
- No image-level Hessian or ridge filter was measured here.
- Versions: NumPy, pandas, SciPy and Matplotlib as installed in the workbook
  environment.
- DIPlib itself was not called; the mean-subtract recipe follows
  [@dspse78280] and [@diplib_gauss].
