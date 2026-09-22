---
title: 'On the Laplacian pyramid'
date: 2026-09-22
options:
  updated: 2026-09-22
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Assisted-by: claude-code:claude-fable-5-1

`skimage.transform.pyramid_laplacian` cites Burt and Adelson (1983). Their
Laplacian pyramid is invertible: the layers, plus one small low-pass residual,
rebuild the input exactly. A user report on the image.sc forum
([Elena Pascal, 2025](https://forum.image.sc/t/unclarity-about-laplacian-pyramid-transform/115976))
shows that the scikit-image pyramid does not rebuild the input. This notebook
shows what the function computes instead, why that breaks reconstruction, what
the correct construction is, and whether the two pull requests that tried to fix
it are correct.

Notation follows Torralba, Isola and Freeman, *Foundations of Computer Vision*,
chapter 23 (reference at the end). $\mathbf{g}_k$ is level $k$ of the Gaussian
pyramid, $\mathbf{l}_k$ is level $k$ of the Laplacian pyramid, and bold capitals
are linear operators. Arrays are in `(row, column)` order.

## 1. Setup

```{code-cell} ipython3
import math
import time

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
# The subject under test.
import skimage as ski
from skimage import data
from skimage.transform import (
    pyramid_gaussian,
    pyramid_laplacian,
    pyramid_reduce,
    pyramid_expand,
    resize,
)
from _skimage2.transform.pyramids import _smooth

print(ski.__version__)
```

```{code-cell} ipython3
# Comparator.
import cv2
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette; validate_palette.js: ALL CHECKS PASS
# (light mode; #1baf7a contrast WARN, so every series is also labelled).
C_ONE = "#2a78d6"
C_TWO = "#eb6834"
C_THREE = "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dedcd5"

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.titlesize": 9,
     "axes.titlecolor": MUTED, "figure.facecolor": "white",
     "image.cmap": "gray"}
)


def bare(ax, title=None):
    """Strip an axis down to the data."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title)
    return ax


def recede(ax, title=None):
    """Keep the ticks, but make the frame recede."""
    ax.tick_params(labelsize=8, colors=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    if title is not None:
        ax.set_title(title)
    return ax
```

## 2. Helpers

The default `sigma` of the pyramid functions is `2 * downscale / 6`, which is
$2/3$ for `downscale=2`. Every helper below uses that value, and the boundary
mode `'reflect'`, so that it matches what `pyramid_reduce` and `pyramid_expand`
compute.

```{code-cell} ipython3
SIGMA = 2 * 2 / 6.0


def smooth(x):
    """Gaussian blur with the pyramid's default sigma and boundary mode."""
    return _smooth(x, SIGMA, "reflect", 0, None)


def expand_to(x, shape):
    """Upsample x to an exact shape, then blur: the F operator of the book."""
    up = resize(x, shape, order=1, mode="reflect", anti_aliasing=False)
    return smooth(up)


def laplacian_pyramid(image, expand=expand_to, max_layer=-1):
    """Burt-Adelson pyramid: l_k = g_k - F g_{k+1}, last layer is g_N."""
    g = list(pyramid_gaussian(image, max_layer=max_layer))
    layers = [g[k] - expand(g[k + 1], g[k].shape) for k in range(len(g) - 1)]
    layers.append(g[-1])
    return layers


def collapse(layers, expand=expand_to):
    """Rebuild g_0 from a Laplacian pyramid whose last layer is the residual."""
    out = layers[-1]
    for layer in layers[-2::-1]:
        out = layer + expand(out, layer.shape)
    return out


def op_matrix(f, n_in, n_out):
    """Matrix of a linear map on 1-D signals, one column per unit impulse."""
    m = np.zeros((n_out, n_in))
    for i in range(n_in):
        e = np.zeros(n_in)
        e[i] = 1.0
        m[:, i] = f(e)
    return m


def rms(x):
    """Root mean square of an array."""
    return float(np.sqrt(np.mean(np.square(x))))
```

## 3. Background: the two pyramids

A [Gaussian pyramid](https://en.wikipedia.org/wiki/Pyramid_(image_processing))
is a sequence of images, each a blurred and half-size copy of the one before.
Level $0$ is the input. Level $k+1$ is

$$
\mathbf{g}_{k+1} = \mathbf{D}\,\mathbf{B}\,\mathbf{g}_k = \mathbf{G}\,\mathbf{g}_k ,
$$

where $\mathbf{B}$ blurs and $\mathbf{D}$ keeps every second sample. Burt and
Adelson call $\mathbf{G}$ `REDUCE`; scikit-image calls it `pyramid_reduce`.

A Laplacian pyramid stores, at each level, what level $k$ of the Gaussian
pyramid contains and level $k+1$ does not. To compare the two, level $k+1$ must
first come back to the size of level $k$. The book writes this as
$\mathbf{F} = \mathbf{B}\,\mathbf{U}$: insert zeros between samples
($\mathbf{U}$), then blur. Burt and Adelson call it `EXPAND`; scikit-image calls
it `pyramid_expand`. The Laplacian level is

$$
\mathbf{l}_k = \mathbf{g}_k - \mathbf{F}\,\mathbf{g}_{k+1}
             = (\mathbf{I} - \mathbf{F}\,\mathbf{G})\,\mathbf{g}_k
\qquad (k < N),
$$

and the last level is the coarsest Gaussian level itself, the *low-pass
residual*:

$$
\mathbf{l}_N = \mathbf{g}_N .
$$

Reconstruction is the same recurrence read backwards, from $k = N-1$ down to
$0$:

$$
\mathbf{g}_k = \mathbf{l}_k + \mathbf{F}\,\mathbf{g}_{k+1} .
$$

Substitute the definition of $\mathbf{l}_k$ into the right-hand side and the
$\mathbf{F}\,\mathbf{g}_{k+1}$ terms cancel, for any $\mathbf{F}$ and any
$\mathbf{G}$. The book states this directly: "the reconstruction property of the
Laplacian pyramid does not depend on the filters used for subsampling and
upsampling."

No property of the filters is needed because the pyramid is predictive
coding. The encoder stores $\mathbf{g}_N$ and, at each level, the error of
predicting $\mathbf{g}_k$ from $\mathbf{g}_{k+1}$. The decoder holds
$\mathbf{g}_{k+1}$ when it reaches level $k$, recomputes the same prediction
$\mathbf{F}\,\mathbf{g}_{k+1}$, and adds the stored error. $\mathbf{F}$ need
not be linear; it must be deterministic and the same on both sides. The price
is redundancy: the pyramid holds about $4/3$ as many coefficients as the image
has pixels. A critically sampled transform, such as a wavelet or QMF pyramid,
has no such slack, and there the analysis and synthesis filters must satisfy
perfect-reconstruction conditions (book, section 23.3). The filters do decide
what the layers look like: whether each $\mathbf{l}_k$ is band-pass, near zero
mean and low in entropy. Section 6 tests invertibility with a random
$\mathbf{F}$; this notebook does not measure band separation.

:::{figure}
:label: fig-analysis

```{mermaid}
flowchart TB
  g0((g0)) -->|G| g1((g1)) -->|G| g2((g2))
  g0 -->|"− F g1"| l0[/"l0"/]
  g1 -->|"− F g2"| l1[/"l1"/]
  g2 -->|copy| l2[/"l2 (residual)"/]
```

Analysis, top to bottom. Each Gaussian level (circle) feeds the next through
$\mathbf{G}$, and gives one stored layer (slanted box) by subtracting the
expanded next level. The coarsest level is stored as the residual.
:::

:::{figure}
:label: fig-synthesis

```{mermaid}
flowchart BT
  l2[/"l2"/] -->|copy| g2((g2))
  g2 -->|"+ F"| g1((g1))
  l1[/"l1"/] -->|"+"| g1
  g1 -->|"+ F"| g0((g0))
  l0[/"l0"/] -->|"+"| g0
```

Synthesis, bottom to top. Start from the residual, expand with the same
$\mathbf{F}$, and add the stored layer at each level.
:::

Two facts follow and both are tested below. First, a pyramid without the
residual cannot be inverted, because the residual is the only place the mean
brightness is stored. Second, the layers must be defined with the *same*
$\mathbf{F}$ that the reconstruction uses.

The name comes from the shape of the operator $\mathbf{I} - \mathbf{F}\mathbf{G}$.
Its rows are a difference of two Gaussian-like blurs, which approximates the
[Laplacian of Gaussian](https://en.wikipedia.org/wiki/Difference_of_Gaussians).

## 4. The problem

The forum post applies the recurrence with $N = 1$: layer $0$ of
`pyramid_laplacian` plus `pyramid_expand` of level $1$ of `pyramid_gaussian`
should give the image back.

```{code-cell} ipython3
image = ski.util.img_as_float(data.camera())

l0_skimage = next(pyramid_laplacian(image, max_layer=0))
g = list(pyramid_gaussian(image, max_layer=1))
rebuilt = l0_skimage + pyramid_expand(g[1])

print(f"rms(rebuilt - image) = {rms(rebuilt - image):.4f}")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
bare(axes[0], "image").imshow(image, vmin=0, vmax=1)
bare(axes[1], "layer 0 + expand(g_1)").imshow(rebuilt, vmin=0, vmax=1)
bare(axes[2], "difference").imshow(rebuilt - image, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
fig.tight_layout()
```

The rebuilt image is softer than the input, and the difference image is
every edge of the scene. The rms difference is printed above; on a $[0, 1]$
scale it is about a fiftieth of the range. Section 5 says where it went.

## 5. What `pyramid_laplacian` computes

The docstring says each layer is
`resize(prev_layer) - smooth(resize(prev_layer))`. In the notation above that is
$(\mathbf{I} - \mathbf{B})\,\mathbf{g}_k$: the Gaussian level minus a blurred
copy of *itself*, at the *same* resolution. The check below confirms this for
every layer.

```{code-cell} ipython3
layers = list(pyramid_laplacian(image))
g = list(pyramid_gaussian(image))

same_as_high_pass = [np.allclose(lk, gk - smooth(gk)) for lk, gk in zip(layers, g)]
print(f"{len(layers)} layers; each equals g_k - smooth(g_k): {all(same_as_high_pass)}")
print("shape of the last layer:", layers[-1].shape)
```

Three things differ from the definition in section 3.

- **The comparison level is wrong.** The layer subtracts $\mathbf{B}\,\mathbf{g}_k$,
  not $\mathbf{F}\,\mathbf{G}\,\mathbf{g}_k$. The two are equal only if
  downsampling then upsampling is the identity, and it is not.
- **There is no residual.** The last layer is also a difference. The coarsest
  Gaussian level is never returned.
- **Every layer has zero mean.** A Gaussian blur with reflecting boundaries
  preserves the sum of an image, so $(\mathbf{I} - \mathbf{B})$ removes the
  mean at every level. The mean brightness of the input appears in no layer.

The last point has a one-line demonstration. A constant image produces a pyramid
of zeros:

```{code-cell} ipython3
flat = np.full((64, 64), 0.7)
print("largest |value| over all layers of a constant image:",
      f"{max(np.abs(lk).max() for lk in pyramid_laplacian(flat)):.1e}")
```

The first point is the forum author's second guess: that the code assumes
`smooth(image)` equals `expand(reduce(image))`. Measured on the camera image:

```{code-cell} ipython3
b_g0 = smooth(image)
fg_g0 = pyramid_expand(pyramid_reduce(image))
print(f"rms(B g0 - F G g0) = {rms(b_g0 - fg_g0):.4f}   (rms of g0 itself: {rms(image):.4f})")
```

Blurring, downsampling and upsampling blurs more than blurring alone. The
layer-0 difference between the two constructions is that same array, because
both subtract from the same $\mathbf{g}_0$.

## 6. The construction the definition asks for

`laplacian_pyramid` in section 2 follows section 3 with the scikit-image
operators: `pyramid_gaussian` for $\mathbf{G}$, and `resize` to the exact shape
of the finer level followed by the same blur for $\mathbf{F}$. `collapse` runs
the recurrence backwards. Reconstruction is exact to rounding.

```{code-cell} ipython3
layers_ok = laplacian_pyramid(image)
rebuilt_ok = collapse(layers_ok)

print(f"{len(layers_ok)} layers, last has shape {layers_ok[-1].shape}")
print(f"max |rebuilt - image| = {np.abs(rebuilt_ok - image).max():.1e}")
```

For even shapes, `expand_to` is `pyramid_expand`; the check below shows the two
agree. The exact-shape form is needed for odd shapes, where `pyramid_expand`
returns one row or column too many. Section 9 comes back to this.

```{code-cell} ipython3
print("expand_to == pyramid_expand on a 256x256 level:",
      np.allclose(expand_to(g[1], g[0].shape), pyramid_expand(g[1])))

odd = np.random.default_rng(0).random((31, 27))
g_odd = list(pyramid_gaussian(odd))
print("odd input, Gaussian shapes:", [x.shape for x in g_odd])
print("pyramid_expand(g_1).shape:", pyramid_expand(g_odd[1]).shape, " g_0.shape:", g_odd[0].shape)
print(f"exact-shape reconstruction error: {np.abs(collapse(laplacian_pyramid(odd)) - odd).max():.1e}")
```

The book's claim that the filters do not matter is also testable. Replace
$\mathbf{F}$ by nearest-neighbour upsampling followed by a fixed random 5-by-5
kernel. Reconstruction stays exact, as long as analysis and synthesis use the
same kernel.

```{code-cell} ipython3
kernel = np.random.default_rng(1).random((5, 5))
kernel /= kernel.sum()


def random_expand(x, shape):
    """Nearest-neighbour upsampling followed by a fixed random blur."""
    up = resize(x, shape, order=0, mode="reflect", anti_aliasing=False)
    return ndi.convolve(up, kernel, mode="reflect")


layers_rand = laplacian_pyramid(image, expand=random_expand)
print(f"random F, same F in synthesis:      {np.abs(collapse(layers_rand, random_expand) - image).max():.1e}")
print(f"random F, default F in synthesis:   {np.abs(collapse(layers_rand, expand_to) - image).max():.3f}")
```

## 7. The operators as matrices

The book writes each step as a matrix acting on a 1-D signal. The same can be
done for the scikit-image operators, by feeding unit impulses through them. The
signal length is 16.

```{code-cell} ipython3
n = 16
B = op_matrix(smooth, n, n)
G = op_matrix(pyramid_reduce, n, n // 2)
F = op_matrix(lambda e: expand_to(e, (n,)), n // 2, n)
I = np.eye(n)

np.set_printoptions(precision=3, suppress=True, linewidth=120)
print("row 8 of B          :", B[8])
print("row 4 of G = D B    :", G[4])
print("rows 7, 8 of F      :", F[7], F[8], sep="\n                      ")
```

Each row of $\mathbf{G}$ averages two neighbouring input samples with equal
weight, because `resize` places the output sample between them. $\mathbf{F}$
interpolates linearly and then blurs. All three have unit row sums, so all
three preserve a constant signal.

The layer-0 operator of the current code is $\mathbf{I} - \mathbf{B}$. The
definition asks for $\mathbf{I} - \mathbf{F}\mathbf{G}$. Their middle rows are
the impulse responses of the two constructions.

```{code-cell} ipython3
L_current = I - B
L_correct = I - F @ G

fig, axes = plt.subplots(1, 2, figsize=(9, 3))
x = np.arange(n)
for ax, row, color, name in [
    (axes[0], L_current[8], C_ONE, "I − B  (pyramid_laplacian)"),
    (axes[1], L_correct[8], C_TWO, "I − F G  (Burt–Adelson)"),
]:
    ax.axhline(0, color=GRID, lw=0.8)
    ax.stem(x, row, linefmt=color, markerfmt="o", basefmt=" ")
    ax.set_xlim(2, 14)
    ax.set_ylim(-0.3, 1)
    ax.set_xlabel("sample")
    recede(ax, f"row 8 of {name}")
axes[0].set_ylabel("weight")
fig.tight_layout()
```

The current operator subtracts a narrow blur. The correct operator subtracts a
wider one, and its negative lobe covers more samples. It is also not
symmetric, because sample 8 sits off-centre between two samples of the coarse
grid. Both rows sum to zero, so both remove the mean; the difference is the
scale at which they do it.

### Rank of the whole transform

Stack every layer's operator into one analysis matrix, as the book does for
$\mathbf{P}^\mathsf{T}$ in section 23.3. For the current pyramid, each block is
$(\mathbf{I} - \mathbf{B})$ applied after $k$ reductions, down to a single
sample. For the correct pyramid, the blocks are
$(\mathbf{I} - \mathbf{F}\mathbf{G})$ after $k$ reductions, and the last block
is the residual $\mathbf{G}^N$.

```{code-cell} ipython3
def stack_current(n):
    """Analysis matrix of pyramid_laplacian on a length-n signal."""
    rows, P, m = [], np.eye(n), n
    while True:
        Bm = op_matrix(smooth, m, m)
        rows.append((np.eye(m) - Bm) @ P)
        if m == 1:
            break
        m2 = math.ceil(m / 2)
        P = op_matrix(pyramid_reduce, m, m2) @ P
        m = m2
    return np.vstack(rows)


def stack_correct(n):
    """Analysis matrix of the Burt-Adelson pyramid on a length-n signal."""
    rows, P, m = [], np.eye(n), n
    while m > 1:
        m2 = math.ceil(m / 2)
        Gm = op_matrix(pyramid_reduce, m, m2)
        Fm = op_matrix(lambda e: expand_to(e, (m,)), m2, m)
        rows.append((np.eye(m) - Fm @ Gm) @ P)
        P = Gm @ P
        m = m2
    rows.append(P)
    return np.vstack(rows)


A_current = stack_current(n)
A_correct = stack_correct(n)
for name, A in [("pyramid_laplacian", A_current), ("Burt-Adelson", A_correct)]:
    print(f"{name:18s} analysis matrix {A.shape}, rank {np.linalg.matrix_rank(A)}")

_, s, vt = np.linalg.svd(A_current)
print("smallest singular value of the current matrix:", f"{s[-1]:.1e}")
print("its null vector:", vt[-1])
```

Both matrices have 31 rows for 16 inputs, the overcompleteness the book
describes. The correct one has full rank 16. The current one has rank 15, and
its null space is spanned by the constant signal. So the current pyramid
does contain the input up to its mean, in the linear-algebra sense: a
least-squares solve recovers the mean-free signal. What it lacks is the mean,
and a synthesis rule. No telescoping sum rebuilds the input, because the layers
were not formed with any $\mathbf{F}$.

```{code-cell} ipython3
rng = np.random.default_rng(2)
sig = rng.random(n)
coeffs = A_current @ sig
lstsq, *_ = np.linalg.lstsq(A_current, coeffs, rcond=None)
print(f"least squares from current layers, error after removing the mean: "
      f"{np.abs((lstsq - lstsq.mean()) - (sig - sig.mean())).max():.1e}")
print(f"error of the mean itself: {abs(lstsq.mean() - sig.mean()):.3f}")
```

## 8. Side by side

Layer 0 of both constructions on the camera image, with the row through the
middle of the picture.

```{code-cell} ipython3
l0_ok = layers_ok[0]
row = image.shape[0] // 2

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), gridspec_kw={"height_ratios": [3, 1.3]})
lim = 0.15
bare(axes[0, 0], "layer 0, pyramid_laplacian").imshow(l0_skimage, cmap="RdBu_r", vmin=-lim, vmax=lim)
bare(axes[0, 1], "layer 0, Burt–Adelson").imshow(l0_ok, cmap="RdBu_r", vmin=-lim, vmax=lim)
for ax in axes[0]:
    ax.axhline(row, color=INK, lw=0.6)
axes[1, 0].plot(l0_skimage[row], color=C_ONE, lw=1, label="pyramid_laplacian")
axes[1, 1].plot(l0_ok[row], color=C_TWO, lw=1, label="Burt–Adelson")
for ax in axes[1]:
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlim(0, image.shape[1])
    ax.legend(frameon=False, loc="upper right")
    recede(ax, f"row {row}")
fig.tight_layout()

print(f"rms layer 0: pyramid_laplacian {rms(l0_skimage):.4f}, Burt-Adelson {rms(l0_ok):.4f}")
```

The correct layer has about 1.8 times the rms amplitude, because it holds
everything between the input resolution and half of it. The current layer holds
only what a $\sigma = 2/3$ blur removes.

## 9. How often, and how big

The corpus is every 2-D grey image in `skimage.data` that loads without a
network fetch, plus the first channel of the colour ones. Two quantities per
image: the reconstruction error of the current layers when the coarsest
Gaussian level is appended as a residual (the best a user can do with the
current output), and the reconstruction error of the corrected pyramid.

```{code-cell} ipython3
names = ["camera", "astronaut", "coins", "moon", "text", "page", "checkerboard",
         "horse", "brick", "grass", "gravel", "cell", "coffee", "chelsea",
         "rocket", "clock", "hubble_deep_field", "immunohistochemistry",
         "logo", "microaneurysms", "retina"]

records = []
for name in names:
    im = getattr(data, name)()
    if im.ndim == 3:
        im = im[..., 0]
    im = ski.util.img_as_float(im)

    cur = list(pyramid_laplacian(im))
    res = list(pyramid_gaussian(im))[-1]
    rebuilt_cur = collapse(cur[:-1] + [res])

    t0 = time.perf_counter()
    ok = laplacian_pyramid(im)
    t_ok = time.perf_counter() - t0
    t0 = time.perf_counter()
    list(pyramid_laplacian(im))
    t_cur = time.perf_counter() - t0

    records.append({
        "image": name, "shape": f"{im.shape[0]}x{im.shape[1]}",
        "rms current + residual": rms(rebuilt_cur - im),
        "max |err| corrected": np.abs(collapse(ok) - im).max(),
        "time ratio corrected / current": t_ok / t_cur,
    })

corpus = pd.DataFrame(records).set_index("image")
corpus.style.format({"rms current + residual": "{:.3f}",
                     "max |err| corrected": "{:.0e}",
                     "time ratio corrected / current": "{:.1f}"})
```

```{code-cell} ipython3
print(f"{len(corpus)} images. Current + residual: rms error between "
      f"{corpus['rms current + residual'].min():.3f} and {corpus['rms current + residual'].max():.3f} "
      f"on a [0, 1] scale.")
print(f"Corrected: largest error {corpus['max |err| corrected'].max():.0e}.")
print(f"Corrected takes {corpus['time ratio corrected / current'].min():.1f}x to "
      f"{corpus['time ratio corrected / current'].max():.1f}x the time.")
```

The failure is on every image, and its size is a large fraction of the
intensity range. It is not a boundary or a precision effect.

## 10. Properties that ought to hold

- **Reconstruction.** Sections 6 and 9: exact to rounding for the corrected
  pyramid, off by 0.02 to 0.3 rms for the current one.
- **The residual carries the mean.** For the corrected pyramid, the mean of the
  residual equals the mean of the image to the accuracy of `resize`; the mean of
  every other layer is close to zero. For the current pyramid, every layer has
  zero mean and nothing carries the image mean.
- **Layer count.** Both give `max_layer + 1` images for the default
  `max_layer=-1`. Section 12 measures the one case where they differ.

```{code-cell} ipython3
print(f"image mean {image.mean():.4f}; corrected residual mean {layers_ok[-1].mean():.4f}")
print("corrected, |mean| of layers 0..3:", [f"{abs(l.mean()):.1e}" for l in layers_ok[:4]])
print("current,   |mean| of layers 0..3:", [f"{abs(l.mean()):.1e}" for l in layers[:4]])
```

## 11. Comparators

### The book's matrix

Section 23.5 of the book prints $256\,(\mathbf{I} - \mathbf{F}_0\mathbf{G}_0)$
for a length-8 signal with zero boundaries, the binomial filter
$[1, 4, 6, 4, 1]/16$, and $\mathbf{F}_0 = 2\,\mathbf{B}_0\mathbf{U}_0$. Building
those matrices from the definitions reproduces the printed one entry for entry.

```{code-cell} ipython3
def binomial_matrix(n):
    """Blur by [1, 4, 6, 4, 1] / 16 with zero boundaries."""
    b = np.array([1, 4, 6, 4, 1]) / 16
    M = np.zeros((n, n))
    for i in range(n):
        for m in range(-2, 3):
            if 0 <= i + m < n:
                M[i, i + m] = b[m + 2]
    return M


n8 = 8
B0 = binomial_matrix(n8)
D0 = np.zeros((n8 // 2, n8))
D0[np.arange(n8 // 2), 2 * np.arange(n8 // 2)] = 1
U0 = D0.T
F0 = 2 * B0 @ U0
G0 = D0 @ B0

book = np.array([
    [182, -56, -24,  -8,  -2,   0,   0,   0],
    [-56, 192, -56, -32,  -8,   0,   0,   0],
    [-24, -56, 180, -56, -24,  -8,  -2,   0],
    [ -8, -32, -56, 192, -56, -32,  -8,   0],
    [ -2,  -8, -24, -56, 180, -56, -24,  -8],
    [  0,   0,  -8, -32, -56, 192, -56, -32],
    [  0,   0,  -2,  -8, -24, -56, 182, -48],
    [  0,   0,   0,   0,  -8, -32, -48, 224],
])
print("256 (I - F0 G0) equals the book's matrix:",
      np.array_equal(np.rint(256 * (np.eye(n8) - F0 @ G0)), book))
```

### OpenCV

`cv2.pyrDown` and `cv2.pyrUp` implement Burt and Adelson's `REDUCE` and
`EXPAND` with the binomial kernel. Building the pyramid with them and collapsing
it is exact to float32 rounding, with no scikit-image code involved.

```{code-cell} ipython3
def cv_pyramid(im, levels=5):
    """Laplacian pyramid from OpenCV's pyrDown and pyrUp."""
    g = [im]
    for _ in range(levels):
        g.append(cv2.pyrDown(g[-1]))
    out = [g[k] - cv2.pyrUp(g[k + 1], dstsize=g[k].shape[::-1]) for k in range(levels)]
    out.append(g[-1])
    return out


def cv_collapse(layers):
    """Rebuild the image from an OpenCV Laplacian pyramid."""
    out = layers[-1]
    for layer in layers[-2::-1]:
        out = layer + cv2.pyrUp(out, dstsize=layer.shape[::-1])
    return out


cv_layers = cv_pyramid(image.astype(np.float32))
print(f"OpenCV (float32) reconstruction, max |err| = {np.abs(cv_collapse(cv_layers) - image).max():.1e}")
```

## 12. The two pull requests

Two pull requests proposed the same change and both were closed by their
authors after a maintainer asked whether they understood the algorithm. Neither
was reviewed on technical grounds. The relevant question here is whether the
code was right.

- [#8274](https://github.com/scikit-image/scikit-image/pull/8274) (August 2026,
  closes #8007): builds the Gaussian pyramid with `pyramid_gaussian`, forms each
  layer as `prev - expand(next)` with a new private `_pyramid_expand(image,
  out_shape, ...)` that `pyramid_expand` also calls, and yields the last
  Gaussian level. Tests reconstruction for 1-D, 2-D and RGB inputs of odd size.
- [#8306](https://github.com/scikit-image/scikit-image/pull/8306) (August 2026,
  closes #7921): the same construction with `resize` and `_smooth` inlined.
  Tests reconstruction on one 40-by-40 image.

The cell below is the loop shared by both, with the same argument handling.

```{code-cell} ipython3
def pyramid_laplacian_pr(image, max_layer=-1, downscale=2, sigma=None,
                         order=1, mode="reflect", cval=0, channel_axis=None):
    """The construction proposed in PRs 8274 and 8306."""
    image = ski.util.img_as_float(image)
    if sigma is None:
        sigma = 2 * downscale / 6.0
    g = pyramid_gaussian(image, max_layer=max_layer, downscale=downscale,
                         sigma=sigma, order=order, mode=mode, cval=cval,
                         preserve_range=True, channel_axis=channel_axis)
    prev = next(g)
    for layer in g:
        up = resize(layer, prev.shape, order=order, mode=mode, cval=cval,
                    anti_aliasing=False)
        yield prev - _smooth(up, sigma, mode, cval, channel_axis)
        prev = layer
    yield prev
```

Tests, in the order a reviewer would ask them.

```{code-cell} ipython3
rng = np.random.default_rng(3)
cases = {
    "1-D, 31": (rng.random(31), None),
    "2-D, 31x27": (rng.random((31, 27)), None),
    "RGB, 31x27x3": (rng.random((31, 27, 3)), -1),
    "camera 512x512": (image, None),
}
rows = []
for name, (im, cax) in cases.items():
    pr = list(pyramid_laplacian_pr(im, channel_axis=cax))
    out = pr[-1]
    for layer in pr[-2::-1]:
        up = resize(out, layer.shape, order=1, mode="reflect", anti_aliasing=False)
        out = layer + _smooth(up, SIGMA, "reflect", 0, cax)
    n_old = len(list(pyramid_laplacian(im, channel_axis=cax)))
    rows.append({"case": name, "layers PR": len(pr), "layers current": n_old,
                 "max |err| PR": np.abs(out - im).max()})
pd.DataFrame(rows).set_index("case").style.format({"max |err| PR": "{:.0e}"})
```

```{code-cell} ipython3
print("PR construction equals laplacian_pyramid (section 2) on camera:",
      all(np.allclose(a, b) for a, b in zip(pyramid_laplacian_pr(image), laplacian_pyramid(image))))
```

The PR code is correct: exact reconstruction for odd sizes and for a channel
axis, and the same layer count as today for the default `max_layer`.

Two points remain.

**An explicit `max_layer` past the natural end returns fewer layers.**
`pyramid_gaussian` stops when a reduction no longer changes the shape.
The current `pyramid_laplacian` keeps yielding 1-by-1 zero layers until it
reaches `max_layer`. The PRs inherit the `pyramid_gaussian` behaviour. The
docstring promises `max_layer + 1` images, so this is a documented-contract
change for one corner case. The automated review on #8306 raised it; neither PR
answered.

```{code-cell} ipython3
tiny = rng.random((4, 4))
print("4x4 input, max_layer=5:",
      f"current yields {len(list(pyramid_laplacian(tiny, max_layer=5)))} layers,",
      f"PR yields {len(list(pyramid_laplacian_pr(tiny, max_layer=5)))}.")
```

**Users cannot collapse the pyramid with public functions.** Both PRs, and
both of their tests, reach for `resize` plus the private `_smooth`, because
`pyramid_expand(upscale=2)` cannot produce an odd target shape (section 6). A
fix that makes the pyramid invertible should also give users a public inverse.

## 13. Ways forward

The construction is settled: it is the one in section 6, which both PRs
implement. What remains is API.

1. **Adopt the PR construction** for `pyramid_laplacian`, with the docstring
   rewritten around $\mathbf{l}_k = \mathbf{g}_k - \mathbf{F}\,\mathbf{g}_{k+1}$
   and the residual. #8274's private `_pyramid_expand(image, out_shape, ...)`
   helper is the cleaner of the two, because `pyramid_expand` then shares it.
2. **Give `pyramid_expand` an exact target.** Either an `out_shape` keyword or
   acceptance of a shape tuple for `upscale`, so that a user can write the
   synthesis loop with public functions. Without this, the reconstruction test
   in the PRs can only be written against private helpers.
3. **Add the inverse.** A function that takes the list of layers and returns
   $\mathbf{g}_0$, and a test that round-trips odd, even, 1-D, and
   channel-axis inputs. Both closed PRs contain usable tests.
4. **Decide the `max_layer` corner case** explicitly: either keep yielding
   1-by-1 layers to honour the docstring, or change the docstring.

## 14. What each fix costs

- **Every layer changes.** All layers of all inputs differ from the current
  output. Any downstream code that used the current layers as a fixed-scale
  high-pass filter will see about twice the energy per layer (section 8).
- **Time.** The corrected pyramid runs two to three times slower on the corpus
  in section 9, because it computes the Gaussian pyramid and one
  upsample-and-blur per level rather than one blur per level.
- **Layer count** is unchanged except for an explicit `max_layer` beyond the
  natural end (section 12).

## 15. What not to do

**Append the residual and keep the current layers.** Section 9 measures this:
the reconstruction error stays between 0.02 and 0.3 rms, because the layers
were formed with $\mathbf{B}$ rather than $\mathbf{F}\mathbf{G}$.

**Collapse with `pyramid_expand(upscale=2)`.** It returns the wrong shape for
any odd level (section 6), and the addition fails. The forum thread's first
code snippet fails this way on a colour image, where the channel axis is also
upsampled.

```{code-cell} ipython3
rgb = ski.util.img_as_float(data.astronaut())
g1_rgb = list(pyramid_gaussian(rgb, max_layer=1, channel_axis=-1))[1]
print("pyramid_expand on an RGB level without channel_axis:",
      pyramid_expand(g1_rgb).shape, "; with channel_axis=-1:",
      pyramid_expand(g1_rgb, channel_axis=-1).shape)
```

**Compare a float reconstruction with the integer input.** The pyramid
functions convert to float in $[0, 1]$. The forum thread rescaled the
reconstruction back to $[0, 255]$ with `rescale_intensity`, which stretches the
result to the observed extremes and so hides any offset. Compare in float.

## 16. Summary

| question | answer | cell |
| --- | --- | --- |
| What does `pyramid_laplacian` compute? | $(\mathbf{I} - \mathbf{B})\,\mathbf{g}_k$ at each level, no residual | §5 |
| Does it rebuild the input? | No: rms error 0.02 to 0.3 on 21 images, even with a residual appended | §9 |
| What is lost? | The image mean, exactly; and the synthesis rule | §5, §7 |
| What does the definition ask for? | $\mathbf{g}_k - \mathbf{F}\,\mathbf{g}_{k+1}$, last layer $\mathbf{g}_N$ | §3, §6 |
| Does that rebuild the input? | Yes, to $10^{-16}$, for any $\mathbf{F}$ used consistently | §6 |
| Are PRs #8274 and #8306 correct? | Yes; same construction, exact on odd, RGB, 1-D | §12 |
| What did they miss? | A public inverse; the `max_layer` corner case | §12 |
| Cost of the fix | All layers change; two to three times slower | §9, §14 |

### Limits

- Corpus: 21 images from `skimage.data`, first channel of colour images,
  `downscale=2`, default `sigma`, `order=1`, `mode='reflect'`. Other downscale
  factors and boundary modes were not measured.
- The matrix analysis in section 7 is for a 1-D signal of length 16. The rank
  result was not checked for other lengths.
- Versions: scikit-image printed in section 1; OpenCV, pandas and SciPy as
  installed.
- The choice of $\mathbf{F}$ affects how well each layer isolates one octave.
  This notebook tests only invertibility, not band separation.

## References

- P. J. Burt and E. H. Adelson, "The Laplacian Pyramid as a Compact Image
  Code," *IEEE Transactions on Communications*, vol. COM-31, no. 4,
  pp. 532–540, April 1983.
  [doi:10.1109/TCOM.1983.1095851](https://doi.org/10.1109/TCOM.1983.1095851).
  Author copy: <http://persci.mit.edu/pub_pdfs/pyramid83.pdf>.
- A. Torralba, P. Isola and W. T. Freeman, *Foundations of Computer Vision*,
  Adaptive Computation and Machine Learning series, The MIT Press, Cambridge,
  MA, 2024. ISBN 9780262378666. Chapter 23, "Image Pyramids":
  <https://visionbook.mit.edu/pyramids_new_notation.html>.
- E. Pascal et al., "Unclarity about Laplacian pyramid transform," image.sc
  forum, 2025.
  <https://forum.image.sc/t/unclarity-about-laplacian-pyramid-transform/115976>.
- scikit-image issues [#7921](https://github.com/scikit-image/scikit-image/issues/7921)
  and [#8007](https://github.com/scikit-image/scikit-image/issues/8007);
  pull requests [#8274](https://github.com/scikit-image/scikit-image/pull/8274)
  and [#8306](https://github.com/scikit-image/scikit-image/pull/8306).
- Wikipedia, [Pyramid (image processing)](https://en.wikipedia.org/wiki/Pyramid_(image_processing))
  and [Difference of Gaussians](https://en.wikipedia.org/wiki/Difference_of_Gaussians).
