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

# The Jerman vesselness filter, as proposed in PR 8074

[PR 8074](https://github.com/scikit-image/scikit-image/pull/8074) adds a fifth
ridge filter, from [Jerman *et al.*
(2016)](https://doi.org/10.1109/TMI.2016.2550102). `on_frangi.md` and
`on_meijering.md` catalogue defects in two of the four already there, so the
question for a new one is whether it repeats them.

The short answer: it repeats one and avoids two, and the deviation from the
author's own reference implementation that looks most alarming turns out not to
matter at all. That last point corrects a claim made in `on_meijering.md` §5.4,
which is amended.

Coordinates are in array order.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from nbhelper import show_table
```

```{code-cell} ipython3
import skimage as ski
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import sato, frangi, meijering
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette, validated all-pairs;
# same palette as the other notebooks here.
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

## 1. What the filter computes

Sort the Hessian eigenvalues by magnitude, $|\lambda_1| \le |\lambda_2| \le
|\lambda_3|$. Jerman's measure uses the largest one and a *regularised* version
of it, $\lambda_\rho$, which replaces small values by a floor set at a fraction
$\tau$ of the largest $\lambda_3$ **anywhere in the image** (their equation 13):

$$
\lambda_\rho = \begin{cases}
0 & \lambda_3 \le 0 \\
\tau \max_\mathbf{x} \lambda_3 & 0 < \lambda_3 \le \tau \max_\mathbf{x}\lambda_3 \\
\lambda_3 & \text{otherwise.}
\end{cases}
$$

The response is then equations 14 and 15:

$$
\mathcal{V} = \begin{cases}
0 & \lambda_2 \le 0 \ \text{or}\ \lambda_\rho \le 0\\
1 & \lambda_2 \ge \lambda_\rho/2 > 0\\
\lambda_2^2\,(\lambda_\rho - \lambda_2)\left[\dfrac{3}{\lambda_2+\lambda_\rho}\right]^3
  & \text{otherwise,}
\end{cases}
$$

and the filter takes the maximum over a list of σ. In 2-D there is no third
eigenvalue and the reference sets $\lambda_3 = \lambda_2$.

The PR transcribes this directly. `skimage.filters.jerman` is not in this
notebook's environment, so the version below is a transcription, with a
`power` argument added for section 4.

```{code-cell} ipython3
def jerman_core(image, sigmas, tau=0.75, black_ridges=True, mode="reflect",
                power=0.0, tol=1e-10):
    """PR 8074's `jerman`, with an optional sigma power on the Hessian.

    `power=0` is the PR as written; `power=2` adds the `c = sigma.^2` scaling
    that Jerman's own MATLAB applies. Returns the fused maximum and the
    per-scale responses.
    """
    image = image.astype(float, copy=False)
    if not black_ridges:
        image = -image
    fused, per_scale = np.zeros_like(image), {}
    for sigma in sigmas:
        eigvals = hessian_matrix_eigvals(
            hessian_matrix(image, sigma, mode=mode, use_gaussian_derivatives=True))
        eigvals = np.take_along_axis(eigvals, abs(eigvals).argsort(0), 0)
        eigvals = eigvals * sigma**power
        (lambda2,) = np.maximum(eigvals[1:], tol)      # 2-D: lambda3 = lambda2
        lambda3 = lambda2.copy()

        floor = tau * lambda3.max()                    # eq. 13
        rho = np.full_like(lambda3, floor)
        rho[lambda3 <= tol] = 0
        rho[lambda3 > floor] = lambda3[lambda3 > floor]

        vals = lambda2**2 * (rho - lambda2) * 27 / (lambda2 + rho) ** 3   # eq. 14
        vals[(lambda2 >= rho / 2) & (rho > tol)] = 1                      # eq. 15
        vals[(lambda2 <= tol) | (rho <= tol)] = 0
        per_scale[sigma] = vals
        fused = np.maximum(fused, vals)
    return fused, per_scale
```

```{code-cell} ipython3
PHOTO = ski.util.img_as_float(ski.data.camera())[::2, ::2]
SIGMAS = (1, 3, 5, 7, 9)
N = 241
rows, cols = np.indices((N, N), dtype=float)
MID = N // 2
```

The transcription was checked against the built PR rather than assumed. In a
worktree at PR 8074 (`jerman-vesselness`, built with `spin install`),

```
jerman_core(PHOTO, SIGMAS)[0]  vs  skimage.filters.jerman(PHOTO, sigmas=SIGMAS)
    max |difference| = 0.000e+00
```

which is exact agreement, not agreement to a tolerance. That environment is not
this one, so the comparison cannot be re-run here; everything below is the
transcription.

## 2. What the PR changes against the reference

The PR's docstring says it "was written based on the MATLAB implementation by
Tim Jerman". Against the paper, equations 13, 14 and 15 are transcribed
correctly, including the $\lambda_3 = \lambda_2$ substitution that section B
prescribes for 2-D, the three-way $\lambda_\rho$ rule, and the $27$ that is
$3^3$ from $[3/(\lambda_2+\lambda_\rho)]^3$. The PR's comments cite the
equation numbers and they are the right ones.

One thing in the paper has no counterpart in the PR, and it is not a detail of
the reference implementation but the definition of the Hessian itself.
Equation 1 is

$$
H_{ij}(\mathbf{x}, s) = s^2\, I(\mathbf{x}) * \frac{\partial^2}{\partial x_i \partial x_j} G(\mathbf{x}, s),
$$

with the $s^2$ built in, and Jerman's MATLAB implements it as
`c = sigma.^2; Hxx = c*Hxx; ...`. That is Lindeberg's scale normalisation, the
same $\sigma^2$ whose absence is `on_frangi.md`'s D2 — the defect that leaves
`frangi` with no scale selection and is the subject of the still-open issue
[#7711](https://github.com/scikit-image/scikit-image/issues/7711). Finding it
missing here looks like the same bug arriving in a new filter.

It is not, and the paper says why.

## 3. Why the missing σ² changes nothing

Look at what the response depends on. Every branch of equations 14 and 15 is a
function of $\lambda_2$ and $\lambda_\rho$ **only through their ratio**:
dividing numerator and denominator of the third branch by $\lambda_2^3$ gives

$$
\mathcal{V} = \frac{27\,(r - 1)}{(1 + r)^3}, \qquad r = \lambda_\rho/\lambda_2,
$$

and the two other branches are comparisons — $\lambda_2 \ge \lambda_\rho/2$ is
$r \le 2$ — which are ratio tests too. Now $\lambda_\rho$ is either $\lambda_3$
or $\tau\max\lambda_3$, and both are linear in the Hessian. So multiplying the
Hessian by any positive constant, $\sigma^2$ included, leaves $r$ and therefore
$\mathcal{V}$ exactly unchanged.

```{code-cell} ipython3
without = jerman_core(-PHOTO, SIGMAS, power=0.0)[0]
with_sigma2 = jerman_core(-PHOTO, SIGMAS, power=2.0)[0]

rows_scale = [{"comparison": "no σ power vs σ² on the Hessian",
               "max |difference|": f"{np.abs(without - with_sigma2).max():.2e}",
               "pixels differing at all": f"{(without != with_sigma2).mean():.1%}"}]
for factor in (3.0, 100.0):
    scaled = jerman_core(-PHOTO * factor, SIGMAS)[0]
    rows_scale.append({"comparison": f"image × {factor:g} vs image × 1",
                       "max |difference|": f"{np.abs(scaled - without).max():.2e}",
                       "pixels differing at all": f"{(scaled != without).mean():.1%}"})
show_table(pd.DataFrame(rows_scale), index="comparison")
```

A third of the pixels differ, and every one of them differs in the last bit or
two. Scaling the whole image by a hundred does the same nothing. The filter is
invariant to any positive rescaling of the Hessian, so equation 1's $s^2$ is
inert *for this enhancement function* and the PR loses nothing by dropping it.

"For this enhancement function" is the necessary qualifier. Equation 1 is the
paper's general Hessian, shared with the six established functions it compares
against — Frangi's, Sato's, Li's, Erdt's, Zhou's and plain $\lambda_2$ — and
those are not ratios, so they need the $s^2$ exactly as `sato` does. It is
redundant only in the one function the PR implements.

The paper states the property directly in its discussion, as a design feature
rather than an accident:

> The main benefit of the ratio of eigenvalue magnitudes is that it effectively
> cancels the magnitude decays towards a structure's periphery. It also
> normalizes the response across different scales and thus exhibits a similar
> response on structures of different sizes.

**This corrects `on_meijering.md` §5.4**, which lists the PR as dropping "the
σ² its own reference applies" and warns that merging it would give
scikit-image "a third ridge filter with D2". The first half is true and the
second is wrong: D2 is *about* scale selection, and σ² is not what gives this
filter scale selection — nothing does, which is the subject of the next
section.

## 4. Scale-uniform by design, not scale-selecting

The invariance of section 3 has a consequence worth stating carefully, because
the obvious reading of it is wrong.

Scale normalisation usually exists so that responses at different σ can be
*compared*, and the winner taken as the structure's size. `sato` multiplies by
$\sigma^2$, `meijering` should multiply by $\sigma^{2\gamma}$
(`on_meijering.md` §5), and `frangi` multiplies by nothing, which is why its
finest σ always wins — `on_frangi.md`'s D2. A filter whose response cannot be
changed by *any* rescaling of the Hessian cannot use that mechanism at all, so
it might look as though `jerman` has the same defect.

It does not, because it is not trying to do the same thing. The sentence quoted
above says the aim is "a similar response on structures of different sizes" —
uniformity across scale, not discrimination between scales. Where the response
saturates, every scale returns exactly 1, the maximum over scales is a tie, and
that tie *is* the intended behaviour. Reporting no size is the design.

What is left to measure is the unsaturated regime, which the paper's own
regularisation exists to handle, and there σ does change the answer: through
which structures are tubular at that scale, and through the image-wide maximum
that sets $\lambda_\rho$. Both push one way.

```{code-cell} ipython3
SIGMA_GRID = (1, 2, 3, 4, 6, 8, 10, 12)
# A dominant narrow ridge holds the others below saturation, where V = 1 for
# every sigma and the comparison would be a tie.
scene = -(3.0 * np.exp(-((cols - 20) ** 2) / (2 * 1.5**2))
          + 0.30 * np.exp(-((cols - 80) ** 2) / (2 * 2.0**2))
          + 0.30 * np.exp(-((cols - 140) ** 2) / (2 * 6.0**2))
          + 0.30 * np.exp(-((cols - 200) ** 2) / (2 * 10.0**2)))
_, per = jerman_core(scene, SIGMA_GRID)

rows_pick = []
for column, width in ((80, 2.0), (140, 6.0), (200, 10.0)):
    values = {s: per[s][MID, column] for s in SIGMA_GRID}
    rows_pick.append({"true width": width,
                      "winning σ": max(values, key=values.get),
                      **{f"σ={s}": round(values[s], 3) for s in SIGMA_GRID}})
show_table(pd.DataFrame(rows_pick), index="true width")
```

Three ridges of quite different width, all equal contrast, and every one of
them is won by the coarsest σ on offer. The response rises monotonically with σ
in each row; extend the list and the winner moves with it.

That is not D2 in mirror image, tempting as the symmetry is. `frangi`'s finest-σ
collapse breaks a documented promise — it returns "the maximum of pixels across
all scales" while only one scale can ever win. `jerman` promises no such thing,
and outside this deliberately unsaturated construction the responses are equal
rather than ordered. What the measurement does show is that the useful range of
`sigmas` is bounded from above by something other than the structure size, and
that adding coarser scales to the list will keep raising weak responses. A
caller choosing `smin` and `smax`, which the paper says should follow "the
respective minimal and maximal expected size of the structures of interest",
gets no warning from the filter if they overshoot.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for (column, width), colour in zip(((80, 2.0), (140, 6.0), (200, 10.0)),
                                   (C_ONE, C_TWO, C_THREE)):
    ax.plot(SIGMA_GRID, [per[s][MID, column] for s in SIGMA_GRID], "o-", color=colour,
            lw=1.7, ms=4, label=f"ridge of width {width:g}")
recede(ax, "response against σ: no interior maximum for any width")
ax.set_xlabel("σ", fontsize=8, color=MUTED)
ax.set_ylabel("response", fontsize=8, color=MUTED)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
```

## 5. τ·max(λ₃) is a per-scale global statistic

The floor in equation 13 is a fraction of the largest $\lambda_3$ **over the
whole image**, recomputed at every scale. Structurally that is the defect
`on_meijering.md` is about — a whole-image number inside the scale loop — and
here it is deliberate, with both halves of the choice stated in the paper. Why
a regulariser at all:

> In image regions of uniform intensity, however, all eigenvalue magnitudes are
> low and the ratio is ill-behaved. This problem has been addressed by
> regularizing the eigenvalue with the highest $\lambda_3$ magnitude to a
> fraction τ of the overall highest $\lambda_3$ magnitude.

and why per scale:

> To normalize the response of proposed enhancement function across the scales,
> $\lambda_\rho$ is computed for each scale $s$ separately.

So the global maximum is the price of making a ratio well behaved where both
its terms vanish, and the per-scale recomputation is what replaces the $s^2$
that section 3 showed to be inert. Neither is a transcription error. What the
paper does not discuss is the consequence: the response at a pixel now depends
on pixels arbitrarily far away.

```{code-cell} ipython3
def far_field(fn, image, spot=(0, 0), value=10.0, keep=100):
    """Relative change 100 px away when one distant pixel is brightened."""
    edited = image.copy()
    edited[spot] = value
    before, after = fn(image), fn(edited)
    far = (slice(keep, None), slice(keep, None))
    return (np.abs(before[far] - after[far]).max()
            / max(np.abs(before[far]).max(), 1e-12))


def crop_change(fn, image, size=140, margin=40):
    """Change in the shared interior when the surroundings are cropped away."""
    whole, part = fn(image)[:size, :size], fn(image[:size, :size])
    inner = (slice(margin, -margin),) * 2
    return (np.abs(whole[inner] - part[inner]).max()
            / max(np.abs(whole[inner]).max(), 1e-12))


probes = {}
for label, fn in (
        ("jerman (PR 8074)", lambda im: jerman_core(im, SIGMAS)[0]),
        ("meijering", lambda im: meijering(im, sigmas=SIGMAS)),
        ("frangi", lambda im: frangi(im, sigmas=SIGMAS)),
        ("sato", lambda im: sato(im, sigmas=SIGMAS, mode="reflect"))):
    probes[label] = {"one distant pixel": f"{far_field(fn, PHOTO):.2%}",
                     "crop the surroundings": f"{crop_change(fn, PHOTO):.2%}"}
show_table(pd.DataFrame(probes).T)
```

`sato` is local, as a filter should be. The other three are not, and `jerman`
joins them. The mechanism is direct enough to watch: hold a ridge fixed and
brighten something far away from it.

```{code-cell} ipython3
probe_ridge = np.exp(-((cols - 120) ** 2) / (2 * 4.0**2))

rows_far = []
for amplitude in (0.0, 4.0, 8.0, 16.0, 40.0):
    scene_far = probe_ridge + (
        amplitude * np.exp(-((rows - 30) ** 2 + (cols - 30) ** 2) / (2 * 3.0**2))
        if amplitude else 0.0)
    _, per_far = jerman_core(-scene_far, [4.0])
    eigvals = hessian_matrix_eigvals(
        hessian_matrix(-scene_far, 4.0, mode="reflect",
                       use_gaussian_derivatives=True))
    eigvals = np.take_along_axis(eigvals, abs(eigvals).argsort(0), 0)
    rows_far.append({
        "distant bar amplitude": amplitude,
        "τ·max(λ₃)": round(0.75 * np.maximum(eigvals[1], 1e-10).max(), 5),
        "response at the untouched ridge": round(per_far[4.0][MID, 120], 6)})
show_table(pd.DataFrame(rows_far), index="distant bar amplitude")
```

Not one pixel of the ridge changes. Its score falls from 1.0 to under 0.06
because something in the far corner raised the image-wide maximum. That is the
same failure `on_meijering.md` documents for `meijering`, arriving through a
different formula.

Unlike `frangi`, though, the statistic is recomputed at every scale rather than
frozen at the first, so the ordering defect of `on_frangi.md` D3 does not
appear here.

```{code-cell} ipython3
ascending = jerman_core(-PHOTO, SIGMAS)[0]
descending = jerman_core(-PHOTO, SIGMAS[::-1])[0]
print(f"jerman, max |ascending sigmas − descending| = "
      f"{np.abs(ascending - descending).max():.1e}")
print(f"frangi, the same comparison                = "
      f"{np.abs(frangi(PHOTO, sigmas=SIGMAS) - frangi(PHOTO, sigmas=SIGMAS[::-1])).max():.1e}")
```

## 6. What the PR gets right

Two things it would have been easy to get wrong.

**Wrong-polarity rejection works.** `frangi` clips the eigenvalue denominator
to `1e-10` and relies on the ratio exploding, which fails when the numerator is
also zero — `on_frangi.md` D1, where `black_ridges` becomes inert on an ideal
ridge. `jerman` clips too, but then tests the clipped values explicitly
(`lambda2 <= tol`, `rho <= tol`) and sets those pixels to zero. The test catches
exactly the pixels the clip created.

```{code-cell} ipython3
bright_ridge = np.exp(-((cols - MID) ** 2) / (2 * 3.0**2))

show_table(pd.DataFrame(
    [{"filter": "jerman (PR 8074)",
      "black_ridges=True": jerman_core(bright_ridge, [3.0])[0][MID, MID],
      "black_ridges=False":
          jerman_core(bright_ridge, [3.0], black_ridges=False)[0][MID, MID]},
     {"filter": "frangi",
      "black_ridges=True": frangi(bright_ridge, sigmas=[3.0])[MID, MID],
      "black_ridges=False":
          frangi(bright_ridge, sigmas=[3.0], black_ridges=False)[MID, MID]}]
    ).round(6), index="filter")
```

A bright ridge scored as a dark one reads exactly zero under `jerman` and the
filter's maximum under `frangi`. Note that the probe is the ridge *centre*: a
bright bar has genuinely dark-valley-like shoulders, so the image maximum is
non-zero for both filters and would not show this.

**The response is bounded.** Equation 15 caps it at 1 by construction, so the
output needs no normalisation and carries none — which is why the ordering
check above comes out exactly zero.

```{code-cell} ipython3
fused = jerman_core(-PHOTO, SIGMAS)[0]
print(f"camera, output range: [{fused.min():.4f}, {fused.max():.4f}]")
print(f"fraction saturated at exactly 1.0: {(fused == 1.0).mean():.2%}")
```

That cap has a cost worth naming. Wherever the response saturates, every scale
returns exactly 1 and the maximum over scales is a tie — so on strong
structures the filter reports no scale at all, and `sigmas` is decorative
there. Section 4 had to suppress the saturation deliberately to measure
anything.

## 7. Side by side

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.0))
retina = ski.util.img_as_float(ski.color.rgb2gray(ski.data.retina()))[::4, ::4]
for ax, (name, out) in zip(axes, (
        ("input", retina),
        ("jerman (PR 8074)", jerman_core(retina, SIGMAS)[0]),
        ("frangi", frangi(retina, sigmas=SIGMAS)),
        ("sato", sato(retina, sigmas=SIGMAS, mode="reflect")))):
    bare(ax, name)
    ax.imshow(out, cmap="gray" if name == "input" else SEQ)
fig.suptitle("the dark vessels of `retina`, default parameters", y=1.03)
fig.tight_layout()
```

## 8. The tests the PR adds

`jerman` is added to the existing parametrised cases — null matrix, energy
decrease, linearity, black/white equivalence, dtype, border management — and
one new test of its own, `test_jerman_result_decrease_with_tau_increase`. That
test is carefully done: its docstring derives $\mathcal{V}(r) = 27(r-1)/(1+r)^3$
and $\mathcal{V}'(r) = 27(4-2r)/(1+r)^4 < 0$ for $r > 2$, then asserts
monotonicity over five values of τ on a retina crop.

What is not covered follows from the sections above:

| property | tested? |
| --- | --- |
| τ monotonicity | yes, with a derivation |
| agreement with Jerman's reference implementation | no |
| behaviour across `sigmas` | no |
| locality — does a distant pixel change the answer? | no |
| bounded output, and how often it saturates | no |

The second is the one a reviewer should want most, since the docstring claims
the implementation is based on that reference and §2 found one line of it
missing. §3 shows that particular line is harmless, but a fixture generated
from the MATLAB would say so directly rather than by argument.

## 9. Summary

| finding | verdict | evidence |
| --- | --- | --- |
| faithful transcription of equations 13–15 | yes | §1, exact agreement with the built PR |
| drops the reference's `c = sigma.^2` | yes, and it is immaterial | §3, differences at $10^{-16}$ |
| scale selection | none, and none intended: the response is scale-uniform by design | §4 |
| locality | fails, through τ·max(λ₃) — from the paper, not the port | §5, 92% far-field |
| depends on the order of `sigmas` | no | §5 |
| wrong-polarity rejection | correct, unlike `frangi` | §6 |
| bounded output | yes, at the cost of ties across scales | §6 |

The recommendation this supports is narrow. The PR is a correct port of
equations 13 to 15, and the missing $s^2$ of equation 1 is provably inert for
this function, so a reviewer should not block on it — but the PR should say so,
because the next reader will notice the same gap and have to redo the argument.

The non-locality comes from the published algorithm rather than from the port,
and `meijering` has shipped with the same property for years, so blocking on it
would be inconsistent. It deserves a docstring sentence — `tau` couples every
pixel to the brightest structure in the frame, so cropping the image changes
the answer — exactly as `on_meijering.md` §7 argues for `meijering`.

The one thing I would ask the author to add is a fixture: a small array and its
expected response generated from Jerman's MATLAB. Section 3 argues the port is
faithful; a fixture would demonstrate it, and would have settled the $s^2$
question without any of this.

**Limits.** One 2-D photograph, one retina crop, and synthetic Gaussian ridges,
at `mode='reflect'` and `tau=0.75` unless stated. The 3-D path is not exercised
at all: `lambda3 = lambda2` applies only in 2-D, and the 3-D branch uses two
genuinely different eigenvalues, so none of §3's ratio argument has been
checked there. The agreement with the built PR was measured once, on `camera`
with five σ, in the `jerman-vesselness` worktree; it is quoted here, not
reproduced. No comparison against Jerman's MATLAB was run — §2's reading of it
is from the published source, not from executing it.

```{code-cell} ipython3
print(f"scikit-image {ski.__version__}")
```
