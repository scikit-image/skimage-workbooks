"""Approximate a greyscale image by a sum of isotropic Gaussians of a few widths.

The model is

    f(r, c) = sum_k h_k exp(-((r - r_k)**2 + (c - c_k)**2) / (2 w_k**2))

with ``h_k`` the peak height and ``w_k`` the standard deviation of Gaussian
``k``.  Coordinates are in pixels, in skimage ``rc`` order: ``r`` down the rows,
``c`` across the columns, with pixel centres at integers.

The Gaussians of one width form a *level*: a regular grid of centres whose
spacing equals the width, reaching one spacing past each image edge.  Levels
are fitted coarse to fine, each to the residual that the coarser levels leave.
Because a level is a grid of separable functions, its least-squares fit splits
into two 1-D problems, ``A_r @ H @ A_c.T ~ residual``, solved with Tikhonov
regularisation on each 1-D factor.

Smoothing the model by a Gaussian of scale ``sigma`` gives the same sum with
width ``sqrt(w**2 + sigma**2)`` and height ``h * w**2 / (w**2 + sigma**2)``.
So the smoothed image and its derivatives have closed forms; `render` and
`hessian` evaluate them without truncation.
"""

from dataclasses import dataclass

import numpy as np

WIDTHS = (16.0, 8.0, 4.0, 2.0, 1.5)
RIDGE = 1e-3


@dataclass
class Level:
    """Gaussians of one width, on a grid of centres."""

    width: float
    row_centres: np.ndarray
    col_centres: np.ndarray
    heights: np.ndarray  # shape (len(row_centres), len(col_centres))


def grid_centres(n, spacing):
    """Centres along one axis of length `n`, one spacing past each edge."""
    count = int(np.ceil((n - 1 + 2 * spacing) / spacing)) + 1
    return -spacing + spacing * np.arange(count)


def basis(coords, centres, width, order=0):
    """1-D Gaussians at `centres`, or a derivative of them, sampled at `coords`."""
    d = np.asarray(coords, dtype=float)[:, None] - centres[None, :]
    g = np.exp(-(d**2) / (2 * width**2))
    if order == 0:
        return g
    if order == 1:
        return -d / width**2 * g
    if order == 2:
        return (d**2 - width**2) / width**4 * g
    raise ValueError(f"order must be 0, 1 or 2, not {order}")


def ridge_pinv(A, ridge):
    """Tikhonov-regularised pseudo-inverse, with ridge relative to the top singular value."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return (Vt.T * (s / (s**2 + ridge * s[0] ** 2))) @ U.T


def fit(image, widths=WIDTHS, ridge=RIDGE):
    """Fit levels coarse to fine; return the levels and the final residual."""
    residual = np.array(image, dtype=float)
    rows = np.arange(residual.shape[0], dtype=float)
    cols = np.arange(residual.shape[1], dtype=float)
    levels = []
    for width in widths:
        rc = grid_centres(len(rows), width)
        cc = grid_centres(len(cols), width)
        A_r, A_c = basis(rows, rc, width), basis(cols, cc, width)
        heights = ridge_pinv(A_r, ridge) @ residual @ ridge_pinv(A_c, ridge).T
        residual -= A_r @ heights @ A_c.T
        levels.append(Level(width, rc, cc, heights))
    return levels, residual


def render(levels, shape, sigma=0.0, order=(0, 0), rows=None, cols=None):
    """The model smoothed by `sigma`, or its (row, col) derivative of `order`.

    Evaluates at pixel centres of an image of `shape`, or at the given 1-D
    `rows` and `cols` coordinates.
    """
    rows = np.arange(shape[0], dtype=float) if rows is None else rows
    cols = np.arange(shape[1], dtype=float) if cols is None else cols
    out = np.zeros((len(rows), len(cols)))
    for level in levels:
        variance = level.width**2 + sigma**2
        width = np.sqrt(variance)
        scale = level.width**2 / variance
        out += scale * (
            basis(rows, level.row_centres, width, order[0])
            @ level.heights
            @ basis(cols, level.col_centres, width, order[1]).T
        )
    return out


def hessian(levels, shape, sigma):
    """Exact Hessian elements [H_rr, H_rc, H_cc] of the model smoothed by `sigma`."""
    return [render(levels, shape, sigma, order) for order in ((2, 0), (1, 1), (0, 2))]
