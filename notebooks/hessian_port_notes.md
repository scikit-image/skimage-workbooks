---
title: 'Explaining the porting plan for Hessian Matrix'
date: 2026-11-14
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

This notebook supports PR <>.

Unlike other notebooks in this repository, this one is hand-written by me
(Matthew).  I used AI only to check and review.

The purpose of this document is to set out the various trade-offs in the decisions I had to make for the PR.

## The issues

There are two issues that this PR addresses, and then a variety of consequences.

Here's the summary:

* Issue 1: current (<=0.27) `hessian_matrix` uses a workaround for a (sort-of)
  bug in `scipy.ndimage` Gaussian convolution that is inaccurate at image
  edges. I consider this a bug, that needs to be fixed in Skimage1 (and
  Skimage2).
* Issue 2: `hessian_matrix` has two methods to calculate the Hessian, where
  the default is a finite-difference (FD) method that is less accurate than
  the Gaussian convolution method above.  Various scikit-image routines call
  into `hessian_matrix` using this default and user code (e.g. on Github) does
  as well.  But we want to remove the FD method from the standard API for
  Skimage2.  So we need to support users migrating to Skimage2 who want to
  preserve outputs.

## Background

The Skimage1 API `hessian_matrix` has two algorithms for calculating the
Hessian, at a given Gaussian smoothing `sigma`, at each pixel.  The
[Hessian](https://en.wikipedia.org/wiki/Hessian_matrix) $H$ is the second
derivative of the image:

$$
H =
 \begin{pmatrix}
 H_{ii} & H_{ij} \\
 H_{ij} & H_{jj}
 \end{pmatrix},
 \qquad
 H_{ij} = \frac{\partial^{2}}{\partial x_i \, \partial x_j}(G_{\sigma} \ast f)
$$

We can get this second derivative in two ways; call these: finite differences
(FD); and convolution with second derivative of Gaussian (C2DG).

Quoting the docstring for `hessian_matrix`:

> The distributive property of derivatives and convolutions allows us to
restate the derivative of an image, I, smoothed with a Gaussian kernel, G, as
the convolution of the image with the derivative of G.
>
> $$
\frac{\partial }{\partial x_i}(I \ast G) =
I \ast \left( \frac{\partial }{\partial x_i} G \right)
$$

In words, the derivative of $I$ convolved with the Gaussian, is equal to $I$
convolved with the derivative of the Gaussian.

The Hessian is the second derivative, and has (for 2D) four matrix entries, as
above.

[Hessian Review](https://scikit-image.org/skimage-workbooks/hessian-review)
goes into more detail, but here is the summary code from that notebook.

```{note}
The code below assumes that `scipy.ndimage` correctly generates the second
Gaussian derivative; we'll see below that this is — arguably — not the case,
especially for values of `sigma` < 1.
```

```{code-cell} python
import numpy as np
import scipy.ndimage as ndi
```

```{code-cell} python
def fd_route(image, sigma, mode="constant"):
    """Smooth, then difference: the left-hand side, literally."""
    smoothed = ndi.gaussian_filter(image, sigma, mode=mode)
    d_i, d_j = np.gradient(smoothed)
    d_ii, d_ij = np.gradient(d_i)
    _, d_jj = np.gradient(d_j)
    return [d_ii, d_ij, d_jj]


def cd2g_route(image, sigma, mode="constant"):
    """Convolve once with a derivative-of-Gaussian kernel: the right-hand side."""
    # This is the underlying idea; as we'll see later, the scipy
    # implementation lacks accuracy at low sigma.
    return [ndi.gaussian_filter(image, sigma, order=order, mode=mode)
            for order in ([2, 0], [1, 1], [0, 2])]
```

Skimage1 `hessian_matrix` has parameter `use_gaussian_derivatives`, with
default `None`, where `None` currently maps to `False`, giving
a `FutureWarning` that the default will change to `True` "in a future
version".  This parameter and warning arrived with scikit-image version 0.20.0
(February 2023).

`use_gaussian_derivatives=False` (the default) corresponds to the FD algorithm
above; `=True` is a version of the C2DG algorithm (but see below).

I don't go into it here, but it turns out that the
`scipy.ndimage.gaussian_filter` second-order kernels (`order=2`) break
a couple of contracts for the second derivative; please see [Hessian
Review](https://scikit-image.org/skimage-workbooks/hessian-review) for detail,
and [this StackOverflow
discussion](https://dsp.stackexchange.com/questions/78280/are-scipy-second-order-gaussian-derivatives-correct)
for further background.

The broken contracts are:

* The second-derivative (curvature) kernel should give 0 for a constant set of
  values, but `scipy.ndimage` (NDI)'s kernel does not.
* Its output for a set of values following e.g. $x^2$ should be 2 — or, in
  terms of the discussion above, it should have the correct magnitude response
  for a parabolic function.  This is also not true for NDI.

The consequences for these two broken contracts are two-fold:

* Skimage1 does a partial workaround for this problem by applying the
  NDI Gaussian first-derivative (that breaks its contracts rather less) twice,
  to approximate the second derivative.  This in turn leads to edge effects,
  as the boundary mode (`nearest` etc) has to be applied twice, once for each
  iteration.
* In order to fix the Gaussian second-derivative kernels, we have to build our
  own Gaussian second-derivative kernels, so we can't use
  `scipy.ndimage.gaussian_filter` directly.

Now we have the background, let's go to the issues

## Issue 1: bug in calculation of Hessian matrix

Remember, `hessian_matrix` has two modes:

* `use_gaussian_derivatives=True` — C2DG — that hits the bug referred to
  above.
* `use_gaussian_derivatives=False` — FD — the default.

This issue is about the first `use_gaussian_derivatives=True` case, used by us
in `filters`: `meijering`, `frangi` and `sato`, and therefore, via `frangi`,
by the `hessian` filter (see [What changes for the
callers](https://scikit-image.org/skimage-workbooks/hessian-review/#id-9-what-changes-for-the-callers)
in the Hessian Review notebook).

I consider the correction to the calculation with
`use_gaussian_derivatives=True` to be a bug-fix.  The results of the filters
above will change, generally slightly, typically at the image edges, but these
changes are correct, and so need no compatibility wrappers to reproduce the
old behavior.  See the notebook link above for an analysis.

Similarly, anyone using `use_gaussian_derivatives=True` in the wild will see
there results change, but we consider those results correct, and therefore,
they will have a bug-fix.

## Issue 2: removing finite difference (FD) method

`use_gaussian_derivatives=False` is the current default for `hessian_matrix`.
But we can show that the alternative, fixed C2DG method is more accurate than
FD at all sigmas.  So we would prefer to remove the FD algorithm from
`hessian_matrix`.  But this involves a change in default output.  We need to
give users a way to reproduce their previous Skimage1 results using that
default.

### Who uses the FD default in our codebase?

* Any callers of `hessian_matrix` without an explicit
  `use_gaussian_derivatives=True`.  Examples are the two below.
* Any callers of `hessian_matrix_det` either using explicit
  `approximate=False` (the default is `True`) or applying it to a 3D image
  (where `approximate=True` has no effect). `skimage` itself has no examples.
  (`feature.blob_doh` calls `_hessian_matrix_det`, which is the
  `approximate=True` path).
* Any callers of `shape_index` (because it calls `hessian_matrix` with
  default). `skimage` has no examples.

So we expect no changes to the tests, for the upcoming Skimage2 versions of `shape_index` or `hessian_matrix_det`.

### Who uses the FD default in the wild?

In a Claude (Opus 5.5) search, around 37 repos used
`use_gaussian_derivatives=False` in calls to `hessian_matrix`, and around 4500
files call `hessian_matrix` without specifying `use_gaussian_derivatives`.
The 37 will get errors when porting to Skimage2 (no parameter of that name);
the 4500, and the 37, will need some method of reproducing the FD case when
migrating.

I asked for another Claude search of Github for uses of `hessian_matrix_det`
either using `approximate=False` or using it on 3D images, so that the routine
falls back to the `hessian_matrix` default call.  It found no instances.
I don't think we need to cover that case, but we might want to clean up the
API — see below.

There were more users of `shape_index` — and the differences in output with the non-FD `hessian_matrix` were quite substantial.  So these users need a migration path.

## Plan implemented in the PR

1. Refactored Skimage2 `hessian_matrix` to a) drop `use_gaussian_derivatives`
   and b) use the correctly sampled Gaussian second-derivative kernel.  Test.
1. Made a migration helper
   `_skimage2.migration.hessian_matrix_finite_difference` to reproduce the
   `use_gaussian_derivatives=False` behavior in Skimage1.  Point people
   migrating to Skimage2 as this stopgap.
