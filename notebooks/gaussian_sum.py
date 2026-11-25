"""Approximate a greyscale image by a sum of isotropic Gaussians of a few widths.

The model is Mobahi's isotropic Gaussian radial-basis sum

    f(x) = sum_j a_j exp(-||x - x_j||**2 / (2 delta_j**2))

with ``a_j`` the amplitude and ``delta_j`` the standard deviation of Gaussian
``j``.  Coordinates are in pixels, in skimage ``rc`` order: ``r`` down the rows,
``c`` across the columns, with pixel centres at integers.

The Gaussians of one width form a *level*: a regular grid of centres whose
spacing equals the width, reaching one spacing past each image edge.  Levels
are fitted coarse to fine, each to the residual that the coarser levels leave.
Because a level is a grid of separable functions, its least-squares fit splits
into two 1-D problems, ``A_r @ A @ A_c.T ~ residual``, each solved by ridge
(Tikhonov) least squares.

Smoothing the model by a normalised Gaussian of scale ``sigma`` gives the same
sum with width ``sqrt(delta**2 + sigma**2)`` and amplitude
``a * delta**2 / (delta**2 + sigma**2)`` (Mobahi 2016, section 2.2).  So the
smoothed image and its derivatives have closed forms; `render` and `hessian`
evaluate them without truncation.

The names and the fit match ``tests/skimage2/feature/test_hessian_schemes.py``
on the scikit-image ``hessian-refactor`` branch.
"""

from dataclasses import dataclass

import numpy as np

DELTAS = (16.0, 8.0, 4.0, 2.0, 1.5)  # delta, pixels, coarse to fine
RIDGE = 1e-3  # relative ridge penalty for the separable least-squares fit
NDIM = 2  # n = dim(x) for images


@dataclass
class Level:
    """Gaussians of one width delta, on a grid of centres x_j = (r_j, c_j)."""

    delta: float
    row_centres: np.ndarray
    col_centres: np.ndarray
    amplitudes: np.ndarray  # a_j; shape (len(row_centres), len(col_centres))


def grid_centres(n, spacing):
    """Centres along an axis of length `n`, one spacing past each edge."""
    count = int(np.ceil((n - 1 + 2 * spacing) / spacing)) + 1
    return -spacing + spacing * np.arange(count)


def basis(coords, centres, delta, order=0):
    """1-D factors exp(-(t - t_j)**2 / (2 delta**2)), or a derivative of them."""
    offset = np.asarray(coords, dtype=float)[:, None] - centres[None, :]
    gaussian = np.exp(-(offset**2) / (2 * delta**2))
    if order == 0:
        return gaussian
    if order == 1:
        return -offset / delta**2 * gaussian
    if order == 2:
        return (offset**2 - delta**2) / delta**4 * gaussian
    raise ValueError(f"order must be 0, 1 or 2, not {order}")


def ridge_solve(A, rhs, ridge):
    """Ridge least squares, min ||A X - rhs||**2 + lam ||X||**2, lam = ridge * s0**2.

    ``s0`` is the largest singular value of `A`.  Solved as an ordinary
    least-squares problem on `A` augmented with ``sqrt(lam) I``.
    """
    lam = ridge * np.linalg.norm(A, 2) ** 2
    columns = A.shape[1]
    augmented = np.vstack([A, np.sqrt(lam) * np.eye(columns)])
    rhs_augmented = np.vstack([rhs, np.zeros((columns, rhs.shape[1]))])
    return np.linalg.lstsq(augmented, rhs_augmented)[0]


def fit(image, deltas=DELTAS, ridge=RIDGE):
    """Fit levels coarse to fine; return the levels and the final residual."""
    residual = np.array(image, dtype=float)
    rows = np.arange(residual.shape[0], dtype=float)
    cols = np.arange(residual.shape[1], dtype=float)
    levels = []
    for delta in deltas:
        row_centres = grid_centres(len(rows), delta)
        col_centres = grid_centres(len(cols), delta)
        A_r = basis(rows, row_centres, delta)
        A_c = basis(cols, col_centres, delta)
        mid = ridge_solve(A_r, residual, ridge)             # A_r M ~ residual
        amplitudes = ridge_solve(A_c, mid.T, ridge).T       # A_c A.T ~ M.T
        residual = residual - A_r @ amplitudes @ A_c.T
        levels.append(Level(delta, row_centres, col_centres, amplitudes))
    return levels, residual


def render(levels, shape, sigma=0.0, order=(0, 0)):
    """The model smoothed by `sigma`, or its (row, col) derivative of `order`.

    Values are at the pixel centres of an image of `shape`.
    """
    rows = np.arange(shape[0], dtype=float)
    cols = np.arange(shape[1], dtype=float)
    out = np.zeros((len(rows), len(cols)))
    for level in levels:
        delta_s = np.sqrt(level.delta**2 + sigma**2)
        scale = (level.delta / delta_s) ** NDIM
        out += scale * (
            basis(rows, level.row_centres, delta_s, order[0])
            @ level.amplitudes
            @ basis(cols, level.col_centres, delta_s, order[1]).T
        )
    return out


def hessian(levels, shape, sigma):
    """Exact Hessian elements [H_rr, H_rc, H_cc] of the model smoothed by `sigma`."""
    return [render(levels, shape, sigma, order) for order in ((2, 0), (1, 1), (0, 2))]
