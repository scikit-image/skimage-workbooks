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

# Testing the `frangi` repairs

`frangi` lost its σ² scale normalisation in version 0.20.0. Before that release
it went through a helper that multiplied the Hessian by `sigma ** 2` under the
comment `# Correct for scale`; the rewrite that removed the helper kept that
line in `sato` and dropped it from `frangi`.
[#7711](https://github.com/scikit-image/scikit-image/issues/7711) is the
consequence, reported as "only the smallest scale affects the filter output".

Restoring it is one line. That one line is unsafe until a second change has
landed. `frangi_refactor_plan.md` sets out the two stages and
`on_frangi.md` has the full defect catalogue. This notebook asks the narrower
question the two earlier `_testing` notebooks ask:

> **What should the test suite assert, such that every assertion is a sentence
> about a picture?**

Five tests, and one ordering constraint that a test can state and check.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from itertools import permutations

from nbhelper import show_table
```

```{code-cell} ipython3
import skimage as ski
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import frangi, sato
```

```{code-cell} ipython3
# Slots 1 to 3 of the reference categorical palette, validated all-pairs;
# same palette as `on_frangi.md` and `hessian_testing.md`.
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
```

## 1. The four candidates

The two repairs are independent switches, so there are four things to test, not
two. `parts` caches the per-scale eigenvalues, because the tests below run the
same scale many times in different orders and the Hessian is all the cost.

```{code-cell} ipython3
_CACHE = {}


def parts(name, image, sigma, power=0.0, mode="reflect"):
    """The two ordered eigenvalues and S at one scale, cached by name."""
    key = (name, float(sigma), float(power), mode)
    if key not in _CACHE:
        eigvals = hessian_matrix_eigvals(hessian_matrix(
            np.asarray(image, float), sigma, mode=mode,
            use_gaussian_derivatives=True))
        eigvals = np.take_along_axis(eigvals, abs(eigvals).argsort(0), 0)
        eigvals = eigvals * sigma**power
        _CACHE[key] = (eigvals[0], eigvals[1], np.sqrt((eigvals**2).sum(0)))
    return _CACHE[key]


def frangi_from(name, image, sigmas, power=0.0, hoist=False, gamma=None,
                beta=0.5, sign_test=False, mode="reflect"):
    """`frangi`, with each repair switchable.

    power     : stage B, the sigma ** 2 restore (D2, #7711).
    hoist     : stage A, resolve gamma over all scales, not just sigmas[0] (D3).
    sign_test : the explicit polarity test instead of the clipped divide (D1).
    """
    got = [parts(name, image, s, power, mode) for s in sigmas]
    if gamma is None:
        norms = [s.max() for _, _, s in got]
        gamma = (max(norms) if hoist else norms[0]) / 2 or 1.0
    out = None
    for lambda1, lambda2, s in got:
        if sign_test:
            ok = lambda2 > 0
            r_b = np.abs(lambda1) / np.where(ok, lambda2, 1.0)
            blobness = np.where(ok, np.exp(-r_b**2 / (2 * beta**2)), 0.0)
        else:
            r_b = np.abs(lambda1) / np.maximum(lambda2, 1e-10)
            blobness = np.exp(-r_b**2 / (2 * beta**2))
        vesselness = blobness * (1 - np.exp(-(s**2) / (2 * gamma**2)))
        out = vesselness if out is None else np.maximum(out, vesselness)
    return out


def grid_gamma(name, image, power, sigmas):
    """Half the largest Hessian norm over the whole sigma grid.

    Any scan over sigma must hold gamma fixed like this.  Let each call
    resolve its own gamma from its own single scale and the structuredness
    term is pinned to 1 - exp(-2) at every sigma, which divides out exactly
    the scale dependence the scan is trying to measure (§9.1).
    """
    return max(parts(name, image, s, power)[2].max() for s in sigmas) / 2


CANDIDATES = {"as shipped": {},
              "stage A, hoist gamma": dict(hoist=True),
              "stage B alone, sigma**2": dict(power=2.0),
              "A + B": dict(hoist=True, power=2.0)}
```

The transcription has to be exact or nothing below means anything.

```{code-cell} ipython3
def as_grey(image, target=200):
    """Greyscale float, decimated so the longest side is about `target`."""
    if image.ndim == 3:
        image = ski.color.rgb2gray(image)
    image = ski.util.img_as_float(image)
    step = max(1, max(image.shape) // target)
    return image[::step, ::step]


CORPUS = {"camera": as_grey(ski.data.camera()),
          "retina": as_grey(ski.data.retina()),
          "coins": as_grey(ski.data.coins())}

show_table(pd.DataFrame(
    [{"image": name, "shape": str(image.shape),
      "sigmas": str(sig),
      "identical to the library": str(np.array_equal(
          frangi(image, sigmas=sig), frangi_from(name, image, sig)))}
     for name, image in CORPUS.items() for sig in ((1, 3, 5), (5, 3, 1))]),
    index=False)
```

`as shipped` reproduces `skimage.filters.frangi` bit for bit, in both orders.

+++

## 2. The regression, in one picture

The issue's own figure: elongated features of several widths, dark on light.

```{code-cell} ipython3
SIZE = 200
WIDTHS = (1.0, 2.0, 4.0, 8.0)
CENTRES = (28, 75, 122, 172)
SIGMAS = (1, 2, 3, 4, 6, 8, 10, 12)

columns = np.indices((SIZE, SIZE), dtype=float)[1]
BARS = np.ones((SIZE, SIZE))
for width, centre in zip(WIDTHS, CENTRES):
    BARS -= np.exp(-((columns - centre) ** 2) / (2 * width**2))
PROBES = [(SIZE // 2, centre) for centre in CENTRES]

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
bare(axes[0], "four dark bars, widths 1, 2, 4 and 8")
axes[0].imshow(BARS, cmap="gray")
for ax, label in zip(axes[1:], ("as shipped", "A + B")):
    out = frangi_from("bars", BARS, SIGMAS, **CANDIDATES[label])
    bare(ax, f"{label}\nwidest bar scores {out[PROBES[-1]]:.4f}")
    ax.imshow(out, cmap="gray", vmin=0, vmax=1)
fig.suptitle("the same four bars, found by the same filter", y=1.04)
fig.tight_layout()
```

The shipped filter answers with thin lines where the bars have edges, and says
almost nothing at their centres — which is the reporter's own description, that
"the filter behaves like an edge detector instead". At σ = 1 a wide bar has no
curvature in its middle to respond to, and σ = 1 is the only scale that ever
wins. The next cell is the same fact as numbers.

```{code-cell} ipython3
def bar_scores(label):
    """Vesselness at the centre of each bar, for one candidate."""
    out = frangi_from("bars", BARS, SIGMAS, **CANDIDATES[label])
    return [out[probe] for probe in PROBES]


show_table(pd.DataFrame(
    [{"candidate": label,
      **{f"w = {w:g}": round(v, 4) for w, v in zip(WIDTHS, bar_scores(label))},
      "spread, widest ÷ narrowest":
          f"{max(bar_scores(label)) / min(bar_scores(label)):.0f}x"}
     for label in CANDIDATES]), index="candidate")
```

+++

## 3. Test 1 — a bar's score must not depend on how wide the bar is

This is the plainest statement of what the σ² is for, and it needs no theory:
**four bars of the same contrast should come out of a ridge filter with the
same score.** The filter is given a scale for each of them; its job is to use
it.

As shipped they come out 0.8647, 0.4148, 0.0535 and 0.0039 — a factor of 220
between the narrowest and the widest, on a picture where the only difference is
width. With the σ² they come out within 5% of each other.

```{code-cell} ipython3
# One hue, light to dark, keyed by bar width: the widths are ordered.
ramp = LinearSegmentedColormap.from_list("ramp", ["#bcd3ef", "#123a63"])
shades = [ramp(x) for x in np.linspace(0, 1, len(WIDTHS))]

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
for ax, label in zip(axes, ("as shipped", "A + B")):
    recede(ax, label)
    power = CANDIDATES[label].get("power", 0.0)
    gamma = grid_gamma("bars", BARS, power, SIGMAS)
    for width, probe, shade in zip(WIDTHS, PROBES, shades):
        scores = [frangi_from("bars", BARS, [s], power=power, gamma=gamma)[probe]
                  for s in SIGMAS]
        ax.plot(SIGMAS, scores, color=shade, lw=1.5, marker="o", ms=3,
                label=f"w = {width:g}")
        peak = SIGMAS[int(np.argmax(scores))]
        ax.plot([peak], [max(scores)], marker="o", ms=8, mfc="none", mec=C_TWO)
    ax.set_xlabel("σ")
axes[0].set_ylabel("vesselness at the bar centre")
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("orange rings mark the σ each bar chose", y=1.03)
fig.tight_layout()
```

On the left every ring sits at σ = 1 — every bar chooses the finest scale — and
the four rings are at four different heights, from 0.86 down to 0.004. On the
right the rings march to the right as the bar widens, and all four sit at the
same height. A test can assert either fact; the height is the easier one to
explain and the harder one to satisfy by accident.

+++

## 4. Test 2 — shuffling the scales must not change the answer

`sigmas` is documented as a set of scales. A set has no order, and the
docstring does not say otherwise. Today the answer depends on which scale
happens to be first, because `gamma=None` is resolved on the first iteration of
the loop and never revisited.

```{code-cell} ipython3
TRIO = (1, 3, 5)


def order_spread(label, name, image):
    """Worst disagreement over the six permutations of `TRIO`."""
    base = frangi_from(name, image, TRIO, **CANDIDATES[label])
    return max(np.abs(frangi_from(name, image, order, **CANDIDATES[label])
                      - base).max() for order in permutations(TRIO))


show_table(pd.DataFrame(
    [{"image": name,
      **{label: round(order_spread(label, name, image), 4) for label in CANDIDATES}}
     for name, image in CORPUS.items()]), index="image")
```

Three columns of that table are the whole argument of this notebook, so read
them in order. `as shipped` is wrong by almost the full output range. `stage A`
is exactly zero. `stage B alone` is wrong again — less, but by up to 0.29 — and
`A + B` is exactly zero.

That is the ordering constraint, measured: **the σ² restore reintroduces the
defect that stage A removes, so it must not ship first.** §8 says why.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"image": name,
      "stage A vs shipped, sorted sigmas":
          f"{np.abs(frangi_from(name, image, (1, 3, 5, 7, 9)) - frangi_from(name, image, (1, 3, 5, 7, 9), hoist=True)).max():.1e}",
      "bit-identical": str(np.array_equal(
          frangi_from(name, image, (1, 3, 5, 7, 9)),
          frangi_from(name, image, (1, 3, 5, 7, 9), hoist=True)))}
     for name, image in CORPUS.items()]), index="image")
```

Stage A is bit-identical to today's output for an ascending `sigmas`, on every
corpus image. It ships as a refactor, not as a behaviour change — which is what
makes it safe to land first and on its own.

+++

## 5. Test 3 — `sato` and `frangi` must agree which scale finds a bar

`sato` sits in the same module, takes the same `sigmas`, and kept its σ²
through the rewrite that lost `frangi`'s. Two ridge filters given the same bar
and the same scale list should nominate the same scale. This test needs no
model of either filter — only that they are both ridge filters.

```{code-cell} ipython3
def winning_sigma(score_at):
    """The sigma with the highest score, given a function of sigma."""
    return max(SIGMAS, key=score_at)


def frangi_winner(power, probe):
    """Which sigma frangi nominates, with gamma held fixed across the scan."""
    gamma = grid_gamma("bars", BARS, power, SIGMAS)
    return winning_sigma(lambda s: frangi_from(
        "bars", BARS, [s], power=power, gamma=gamma)[probe])


show_table(pd.DataFrame(
    [{"bar width": width,
      "w√2": round(width * np.sqrt(2), 2),
      "sato": winning_sigma(
          lambda s: sato(BARS, sigmas=[s], mode="reflect")[probe]),
      "frangi as shipped": frangi_winner(0.0, probe),
      "frangi, A + B": frangi_winner(2.0, probe)}
     for width, probe in zip(WIDTHS, PROBES)]), index="bar width")
```

γ is held fixed across the scan, for the reason §9.1 measures.

`sato` and the repaired `frangi` agree on every row, and both land next to
`w√2`, which is where a σ-squared-normalised filter puts the peak. The shipped
`frangi` says 1 four times.

+++

## 6. Test 4 — a bright ridge is not a dark ridge

`black_ridges=True` asks for dark ridges. Hand the filter a bright one and it
should say nothing.

```{code-cell} ipython3
RIDGE_SIZE = 128
ridge_columns = np.indices((RIDGE_SIZE, RIDGE_SIZE), dtype=float)[1]
RIDGE_PROBE = (RIDGE_SIZE // 2, RIDGE_SIZE // 2)


def bright_ridge(width):
    """A bright Gaussian ridge: an exact function of one coordinate."""
    return np.exp(-((ridge_columns - RIDGE_SIZE // 2) ** 2) / (2 * width**2))


show_table(pd.DataFrame(
    [{"ridge width": width,
      "black_ridges=True (want 0)":
          round(float(frangi(bright_ridge(width), sigmas=[3])[RIDGE_PROBE]), 6),
      "black_ridges=False":
          round(float(frangi(bright_ridge(width), sigmas=[3],
                             black_ridges=False)[RIDGE_PROBE]), 6),
      "the two arrays are identical": str(np.array_equal(
          frangi(bright_ridge(width), sigmas=[3]),
          frangi(bright_ridge(width), sigmas=[3], black_ridges=False)))}
     for width in WIDTHS]), index="ridge width")
```

Not merely equal at the probe: the two whole arrays are identical, so
`black_ridges` does nothing at all on this picture. 0.864665 is `1 - exp(-2)`,
the largest score a single-scale `frangi` with the default γ can return — the
wrong-polarity ridge is scoring the ceiling.

The cause is that the rejection is done by a division that is meant to explode,
`abs(lambda1) / max(lambda2, 1e-10)`, and on an ideal ridge `lambda1` is
**exactly** zero, so the division gives 0 rather than infinity. The repair is
an explicit sign test.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"ridge width": width,
      "sign test, bright read as dark": round(float(frangi_from(
          f"bright{width}", bright_ridge(width), [3], sign_test=True,
          gamma=0.02)[RIDGE_PROBE]), 6),
      "sign test, bright read as bright": round(float(frangi_from(
          f"dark{width}", -bright_ridge(width), [3], sign_test=True,
          gamma=0.02)[RIDGE_PROBE]), 6)}
     for width in WIDTHS]), index="ridge width")
```

The defect is triggered by `lambda1` being *exactly* zero, which is a property
of the picture rather than of the filter. §9.2 measures which pictures have it.

+++

## 7. Test 5 — the units of brightness must not matter

With `gamma=None` every quantity in `frangi` is either a ratio of eigenvalues
or is measured against `s.max()`. Multiplying the image by a constant should
therefore change nothing at all.

```{code-cell} ipython3
PHOTO = ski.util.img_as_float(ski.data.camera())[::2, ::2]
photo_base = frangi(PHOTO, sigmas=TRIO)

show_table(pd.DataFrame(
    [{"image ×": f"{factor:.0e}",
      "max difference from ×1": f"{np.abs(frangi(PHOTO * factor, sigmas=TRIO) - photo_base).max():.1e}",
      "as % of the output range":
          f"{np.abs(frangi(PHOTO * factor, sigmas=TRIO) - photo_base).max() / photo_base.max():.2%}"}
     for factor in (1e4, 1e2, 1e-2, 1e-4, 1e-5, 1e-6)]), index="image ×")
```

It does not hold. The `1e-10` in `max(lambda2, 1e-10)` is an absolute constant
in a filter that is otherwise entirely ratios, so once the eigenvalues fall
near it the answer changes. The sign test of §6 removes the clip and with it
this dependence, which is a second reason to make that change.

+++

## 8. Why stage B must not ship before stage A

§4's table shows it; this section says why, because a test suite that knows the
reason can pin it.

Stage A is a no-op today **because** `S` falls with σ, so the largest Hessian
norm sits at the smallest σ, which for a sorted list is also `sigmas[0]`. The
σ² multiplies each scale by a different number and destroys that.

```{code-cell} ipython3
show_table(pd.DataFrame(
    [{"image": name, "sigma**power": f"σ**{power:g}",
      "σ carrying the largest S":
          max(SIGS := (1, 3, 5, 7, 9),
              key=lambda s: parts(name, image, s, power)[2].max()),
      "frozen γ ÷ true γ": round(
          parts(name, image, SIGS[0], power)[2].max()
          / max(parts(name, image, s, power)[2].max() for s in SIGS), 4)}
     for name, image in CORPUS.items() for power in (0.0, 2.0)]), index=False)
```

With `σ**0` the answer is the smallest σ on every image, and the frozen γ is
the true one — the defect is real but invisible. With `σ**2` that stops being
true on `coins`, and the frozen γ is then taken from a scale that is not the
maximum.

Which images move depends on where the image's dominant structure lives, so
this table is about this corpus. §4's permutation column is the claim that does
not depend on that: **with σ² and a frozen γ, every image shows an ordering
spread, and with stage A none of them does.**

+++

## 9. What the tests must avoid

### 9.1 Hold γ fixed while scanning σ

This is the trap that caught three separate cells of this notebook before they
were corrected, so it goes first. `gamma=None` resolves to half the largest
Hessian norm **of the scales it was given**. Hand a single σ to each call and
each call sets its own reference, which pins the structuredness term to
`1 - e⁻²` at the strongest pixel of every scale — dividing out exactly the
scale dependence the scan is there to measure.

```{code-cell} ipython3
def winners(power, gamma):
    """The sigma each bar chooses, at this power and this gamma."""
    return [max(SIGMAS, key=lambda s: frangi_from(
        "bars", BARS, [s], power=power, gamma=gamma)[probe]) for probe in PROBES]


show_table(pd.DataFrame(
    [{"gamma": label,
      "as shipped": str(winners(0.0, gamma(0.0))),
      "A + B": str(winners(2.0, gamma(2.0))),
      "the two agree": str(winners(0.0, gamma(0.0)) == winners(2.0, gamma(2.0)))}
     for label, gamma in (
         ("resolved per call (wrong)", lambda power: None),
         ("resolved over the grid",
          lambda power: grid_gamma("bars", BARS, power, SIGMAS)))]),
    index="gamma")
```

With a per-call γ the shipped filter and the repaired one nominate the **same**
scales, so the test cannot fail and the figure in §3 comes out as two identical
panels. With one γ across the scan they differ on every bar.

### 9.2 Turn the ridge before trusting a rasterised one

§6's defect needs `lambda1` exactly zero, and whether a picture gives that
depends on its *orientation*, not on how smoothly it is drawn. A bar parallel
to an image axis is a function of one coordinate whatever its profile, so the
curvature along it is exactly zero either way. Turn the same bar and the
profile starts to matter.

```{code-cell} ipython3
TURN = 160
turn_rows, turn_cols = np.indices((TURN, TURN), dtype=float)
diagonal = ((turn_rows - 80) * np.cos(np.deg2rad(30))
            - (turn_cols - 80) * np.sin(np.deg2rad(30)))
straight = turn_cols - 80

show_table(pd.DataFrame(
    [{"ridge": label,
      "|lambda1| at the centre":
          f"{abs(parts(label, shape, 3.0)[0][80, 80]):.1e}",
      "black_ridges=True (want 0)":
          round(float(frangi(shape, sigmas=[3])[80, 80]), 6)}
     for label, shape in (
         ("axis-aligned, profile", np.exp(-straight**2 / (2 * 4.0**2))),
         ("axis-aligned, thresholded", (np.abs(straight) < 4.0).astype(float)),
         ("turned 30°, profile", np.exp(-diagonal**2 / (2 * 4.0**2))),
         ("turned 30°, thresholded", (np.abs(diagonal) < 4.0).astype(float)))]),
    index="ridge")
```

The last row is the trap, and it is the opposite of the one to expect: a turned
*rasterised* bar puts `lambda1` at 2e-05, which clears the `1e-10` clip, so the
filter rejects the wrong polarity correctly and the test **passes on a broken
implementation**. The turned exact profile keeps `lambda1` at 3e-18 and the
defect fires.

So a polarity test may use an axis-aligned bar drawn any way at all, or a
turned bar drawn as an exact profile — but a turned bar drawn by thresholding
proves nothing.

### 9.3 Do not test the ordering with a single σ

The ordering defect lives in which scale is *first*. A one-element `sigmas` has
no order, and a test that passes one cannot fail.

### 9.4 Do not assert "close" where the answer is "identical"

Stage A is bit-identical to today's output for sorted `sigmas`, and both stages
make the permutation spread exactly `0.0`. Those are equalities, not
tolerances. A test written with `rtol=1e-5` would pass a filter that still
resolved γ from the wrong scale, as long as the scales were close together.

### 9.5 Do not measure scale selection below σ = 1 yet

`hessian_testing.md` measures the shipped Hessian reporting 8% of a quadratic's
curvature at σ = 0.5. A scale-selecting `frangi` on that operator picks the
wrong scale for any structure narrow enough to need a small σ. Keep this
suite's σ grid at 1 and above until `hessian_matrix` is fixed.

+++

## 10. The suite

```{code-cell} ipython3
def test_a_bars_score_does_not_depend_on_its_width(name, image, **kw):
    """Four bars of one contrast and four widths must score alike."""
    out = frangi_from(name, image, SIGMAS, **kw)
    scores = [out[probe] for probe in PROBES]
    assert min(scores) / max(scores) > 0.9


def test_shuffling_the_scales_changes_nothing(name, image, **kw):
    """`sigmas` is a set of scales, so its order must not matter."""
    base = frangi_from(name, image, TRIO, **kw)
    for order in permutations(TRIO):
        assert np.array_equal(frangi_from(name, image, order, **kw), base)


def test_agrees_with_sato_about_scale(name, image, **kw):
    """Two ridge filters must nominate the same scale for the same bar.

    gamma is resolved once over the whole grid and held fixed, or each
    single-scale call pins its own structuredness and the scan measures
    nothing (§5).
    """
    gamma = grid_gamma(name, image, kw.get("power", 0.0), SIGMAS)
    for probe in PROBES:
        theirs = max(SIGMAS,
                     key=lambda s: sato(image, sigmas=[s], mode="reflect")[probe])
        ours = max(SIGMAS, key=lambda s: frangi_from(
            name, image, [s], gamma=gamma, **kw)[probe])
        assert ours == theirs


def test_a_bright_ridge_is_not_a_dark_ridge(name, image, **kw):
    """`black_ridges=True` must say nothing about a bright ridge."""
    for width in WIDTHS:
        bright = bright_ridge(width)
        scored = frangi_from(f"{name}-bright{width}", bright, [3],
                             gamma=0.02, **kw)[RIDGE_PROBE]
        assert scored < 1e-12


def test_brightness_units_do_not_matter(name, image, **kw):
    """Scaling the image must not change a filter built from ratios.

    Run on the photograph, not the bars: the `1e-10` clip only bites where
    an eigenvalue falls near it, and the bars never get that flat (§10.1).
    """
    reference = frangi_from("photo", PHOTO, TRIO, **kw)
    for factor in (1e-6, 1e-2, 1e2):
        scaled = frangi_from(f"photo-x{factor}", PHOTO * factor, TRIO, **kw)
        assert np.allclose(scaled, reference, atol=1e-6)
```

```{code-cell} ipython3
SUITE = [test_a_bars_score_does_not_depend_on_its_width,
         test_shuffling_the_scales_changes_nothing,
         test_agrees_with_sato_about_scale,
         test_a_bright_ridge_is_not_a_dark_ridge,
         test_brightness_units_do_not_matter]

# Every test runs on the bars picture, which carries all five properties.
STAGES = {"as shipped": {},
          "A": dict(hoist=True),
          "A + B": dict(hoist=True, power=2.0),
          "A + B + sign test": dict(hoist=True, power=2.0, sign_test=True)}


def outcome(test, kw, tag):
    """Run one test against one candidate, and report pass or FAIL."""
    try:
        test(f"bars{tag}", BARS, **kw)
        return "pass"
    except AssertionError:
        return "FAIL"


show_table(pd.DataFrame(
    [{"test": test.__name__.removeprefix("test_").replace("_", " "),
      **{label: outcome(test, kw, label) for label, kw in STAGES.items()}}
     for test in SUITE]), index="test")
```

Each stage turns exactly the tests it is meant to turn, and turns none of the
others back. The last column is the filter this plan proposes.

+++

## 11. Summary

| test | the sentence it asserts | which stage makes it pass |
| --- | --- | --- |
| a bar's score does not depend on its width | four bars of one contrast score alike | B |
| shuffling the scales changes nothing | `sigmas` is a set | A, and A again after B |
| agrees with `sato` about scale | two ridge filters nominate the same σ | B |
| a bright ridge is not a dark ridge | `black_ridges` does what it says | the sign test |
| brightness units do not matter | a filter of ratios has no units | the sign test |

and the rules a test must obey: hold γ fixed while scanning σ (§9.1), never
prove polarity on a turned rasterised bar (§9.2), never test the ordering with
one σ (§9.3), assert equality where the answer is exact (§9.4), and keep σ at 1
and above until `hessian_matrix` is fixed (§9.5).

**Limits.** Three photographs from `skimage.data` decimated to about 200 px,
one synthetic bar picture at 200², one synthetic ridge at 128², `mode='reflect'`
throughout, default `beta`, and σ from 1 to 12. Permutation scans use three
scales, not the full default list of five. Everything here is 2-D: `frangi`'s
3-D branch shares the `gamma` and polarity code and so shares D1 and D3 by
inspection, but that is a reading and not a measurement. §8's table is a
statement about this corpus; §4's permutation column is the claim that is not.
The measurements run against the released two-pass `hessian_matrix`, so the
numbers move a little once that is repaired — the pass and fail pattern in §10
does not.

```{code-cell} ipython3
print(f"scikit-image {ski.__version__}")
```
