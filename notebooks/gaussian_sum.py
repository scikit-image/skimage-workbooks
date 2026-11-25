"""Approximate a greyscale image by a sum of isotropic Gaussians of a few widths.

The model is Mobahi's isotropic Gaussian radial-basis sum [1]

    f(x) = Σ_j a_j exp(−‖x − x_j‖² / (2 δ_j²))

with ``a_j`` the amplitude, ``δ_j`` the width and ``x_j`` the centre of
Gaussian ``j``. We choose the ``δ_j`` and ``x_j``, then do a least-squares fit
to find the ``a_j``. Coordinates are in pixels, in skimage ``rc`` order: ``r``
down the rows, ``c`` across the columns, with pixel centres at integers.

The Gaussians of one width form a *level*: a regular grid of centres whose
spacing equals the width, reaching one spacing past each image edge. Levels
are fitted from coarse (large ``δ``) to fine (small ``δ``), each to the
residual that the coarser levels leave. On the grid, Mobahi's single index
``j`` becomes a pair ``(j, k)``: row centre ``r_j`` and column centre ``c_k``.
Because a level is a grid of separable functions, its least-squares fit splits
into two 1-D problems, ``A_r @ A @ A_c.T ≈ residual`` [2], each solved by ridge
(Tikhonov) least squares.

We use ridge least squares because neighbouring Gaussians, spaced ``δ`` apart
with width ``δ``, overlap heavily. Without a penalty, the fit uses large
amplitudes that nearly cancel. The ridge keeps the amplitudes small, at a
small cost in fit accuracy, and lowers the sigma-1 Hessian error of the
filter.

Smoothing the model by a normalised Gaussian of scale ``σ`` gives the same sum
with width ``√(δ² + σ²)`` and amplitude ``a (δ / √(δ² + σ²))^n`` [1]. So the
smoothed image and its derivatives have closed forms; `render` and `hessian`
evaluate them without truncation.

The code matches ``tests/skimage2/feature/test_hessian_metrics.py`` on the
scikit-image ``hessian-refactor`` branch, with public names: `grid_centres`,
`basis`, `ridge_solve` and `hessian` there are ``_grid_centres``, ``_basis``,
``_ridge_solve`` and ``exact_hessian``.

References
----------
.. [1] H. Mobahi, "Closed Form for Some Gaussian Convolutions,"
    arXiv:1602.05610 [math.CA], 2016.
.. [2] D. W. Fausett and C. T. Fulton, "Large Least Squares Problems Involving
    Kronecker Products," SIAM J. Matrix Anal. Appl., 15(1):219–227, 1994.
    https://doi.org/10.1137/S0895479891222106
"""

from dataclasses import dataclass

import numpy as np
import numpy.linalg as npl

DELTAS = (16.0, 8.0, 4.0, 2.0, 1.5)  # δ, pixels, coarse to fine
RIDGE = 1e-3  # relative ridge penalty for the separable least-squares fit
NDIM = 2  # n = dim(x) for images


@dataclass
class Level:
    """Gaussians of one width δ, on a grid of centres x_j = (r_j, c_j)."""

    delta: float
    row_centres: np.ndarray
    col_centres: np.ndarray
    amplitudes: np.ndarray  # a_j; shape (len(row_centres), len(col_centres))


def grid_centres(n, spacing):
    """Centres along an axis of length ``n``, one spacing past each edge."""
    count = int(np.ceil((n - 1 + 2 * spacing) / spacing)) + 1
    return -spacing + spacing * np.arange(count)


def basis(coords, centres, delta, order=0):
    """1-D factors ``exp(−(t − t_j)² / (2 δ²))``, or a derivative."""
    offset = coords[:, None] - centres[None, :]
    gaussian = np.exp(-(offset**2) / (2 * delta**2))
    if order == 0:
        return gaussian
    if order == 1:
        return -offset / delta**2 * gaussian
    if order == 2:
        return (offset**2 - delta**2) / delta**4 * gaussian
    raise ValueError(f"order must be 0, 1 or 2, not {order}")


def ridge_solve(X, Y, ridge):
    """Ridge regression of ``Y`` on ``X``, as least squares on augmented data.

    In the notation of Wikipedia's "Ridge regression", with design matrix
    ``X``, response ``Y`` (here one column per right-hand side) and
    coefficients ``β``, the model is ``Y = X β + ε``, and the ridge estimate
    minimizes

        ‖Y − X β‖² + λ ‖β‖²,

    with solution

        β̂_λ = (Xᵀ X + λ I)⁻¹ Xᵀ Y.

    This is Tikhonov regularization with ``Γ = √λ I``, so it is ordinary least
    squares on the augmented data

        X_aug = [X; √λ I],    Y_aug = [Y; 0],

    because ``X_augᵀ X_aug = Xᵀ X + λ I`` and ``X_augᵀ Y_aug = Xᵀ Y``.
    ``npl.lstsq`` computes that solution directly.

    The penalty is ``λ = ridge × s₀²``, where ``s₀`` is the largest singular
    value of ``X``. ``X_aug`` has singular values ``√(s_i² + λ)``, so it is
    well conditioned.

    https://en.wikipedia.org/wiki/Ridge_regression
    """
    lambda_ = ridge * npl.norm(X, 2) ** 2
    n_coef, n_cols = X.shape[1], Y.shape[1]
    X_aug = np.vstack([X, np.sqrt(lambda_) * np.eye(n_coef)])
    Y_aug = np.vstack([Y, np.zeros((n_coef, n_cols))])
    return npl.lstsq(X_aug, Y_aug)[0]


def fit(image, deltas=DELTAS, ridge=RIDGE):
    """Fit levels coarse to fine. Return the levels and the residual.

    The Gaussians of one width ``δ`` sit on a grid of spacing ``δ``, reaching
    one spacing past each edge. Each level is fitted to the residual that the
    coarser levels leave.

    An isotropic Gaussian factors into a row Gaussian times a column Gaussian.
    On a rectangular grid of centres, the 2-D basis is therefore the outer
    product of two 1-D bases: ``A_r`` (row Gaussians on the image rows) and
    ``A_c`` (column Gaussians on the image columns). The model at one width is

        residual ≈ A_r @ A @ A_c.T,

    with ``A`` the full matrix of amplitudes ``a_jk``. Vectorised, this is
    ``(A_c ⊗ A_r) vec(A) ≈ vec(residual)`` [2]. Two 1-D ridge solves fit it
    without forming the Kronecker product: ``A_r M ≈ residual``, then
    ``A_c A.T ≈ M.T``. That is a two-step ridge, not one ridge solve on the
    vectorised model.

    ``ridge`` is a relative Tikhonov penalty (see `ridge_solve`). Raising it
    shrinks the amplitudes and the cancellation between them, at the cost of
    fit accuracy.
    """
    residual = np.array(image, dtype=float)
    rows = np.arange(residual.shape[0], dtype=float)
    cols = np.arange(residual.shape[1], dtype=float)
    levels = []
    for delta in deltas:
        row_centres = grid_centres(len(rows), delta)
        col_centres = grid_centres(len(cols), delta)
        row_basis = basis(rows, row_centres, delta)
        col_basis = basis(cols, col_centres, delta)
        row_fit = ridge_solve(row_basis, residual, ridge)
        amplitudes = ridge_solve(col_basis, row_fit.T, ridge).T
        residual = residual - row_basis @ amplitudes @ col_basis.T
        levels.append(Level(delta, row_centres, col_centres, amplitudes))
    return levels, residual


def render(levels, shape, sigma=0.0, order=(0, 0)):
    """Mobahi's smoothed sum, or a derivative of that sum.

    The levels define Mobahi's Gaussian RBF sum ([1], §2.2, unnumbered),

        f(x) = Σ_j a_j exp(−‖x − x_j‖² / (2 δ_j²)).

    Its convolution with the normalised Gaussian ``k_σ`` ([1], §1,
    "Notation") is closed form ([1], §2.2, unnumbered; stated and proved as
    Lemma 1 in Appendix B):

        [f ⋆ k_σ](x) = Σ_j a_j (δ_j / √(δ_j² + σ²))^n
                           exp(−‖x − x_j‖² / (2 (δ_j² + σ²))).

    So each level of width ``δ`` and amplitudes ``a_j`` becomes width
    ``√(δ² + σ²)`` with amplitudes ``a_j (δ / √(δ² + σ²))^n``. At ``σ = 0``
    this is ``f`` itself.

    The derivatives are not in [1]. Each term factors into a row Gaussian
    times a column Gaussian, so the ``(p, q)`` derivative differentiates the
    row factor ``p`` times and the column factor ``q`` times (`basis`).
    ``order`` is ``(row derivative, column derivative)``. Values are at pixel
    centres.
    """
    rows = np.arange(shape[0], dtype=float)
    cols = np.arange(shape[1], dtype=float)
    out = np.zeros((len(rows), len(cols)))
    for level in levels:
        # [1], Lemma 1: width δ becomes √(δ² + σ²) ...
        delta_s = np.sqrt(level.delta**2 + sigma**2)
        # ... and amplitudes a_j are multiplied by (δ / √(δ² + σ²))^n.
        scale = (level.delta / delta_s) ** NDIM
        # One separable term per centre: (row factor) a_jk (column factor).
        row_basis = basis(rows, level.row_centres, delta_s, order[0])
        col_basis = basis(cols, level.col_centres, delta_s, order[1])
        out += scale * row_basis @ level.amplitudes @ col_basis.T
    return out


def hessian(levels, shape, sigma):
    """``[Hrr, Hrc, Hcc]`` of Mobahi's smoothed sum at scale ``sigma``."""
    return [
        render(levels, shape, sigma, order=(2, 0)),
        render(levels, shape, sigma, order=(1, 1)),
        render(levels, shape, sigma, order=(0, 2)),
    ]
