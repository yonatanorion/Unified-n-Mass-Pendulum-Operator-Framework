"""
Closed-form tridiagonal structure of the linearized n-mass pendulum.

With S_j = sum_{i>=j} m_i, L = diag(l), D = diag(m) and E the lower-triangular
matrix of ones, the mass matrix factorizes as M = L E^T D E L.  Because E^{-1}
is lower bidiagonal, M^{-1} = L^{-1} E^{-1} D^{-1} E^{-T} L^{-1} is tridiagonal
and so is the dynamic matrix Q = M^{-1} K that appears in the state-space
operator A = [[0, I], [-Q, 0]].  Every quantity below is computed in O(n)
operations from the masses and lengths, without forming or factorizing M.

Indexing is zero-based: ``diag[j]`` refers to joint j+1 of the paper, and
off-diagonal arrays have length n-1 with ``off[j-1]`` coupling joints j-1, j.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal

ArrayLike = Union[Sequence[float], NDArray[np.float64]]
Vector = NDArray[np.float64]


def _validate(masses: ArrayLike, lengths: ArrayLike, g: float = 9.81) -> Tuple[Vector, Vector]:
    m = np.asarray(masses, dtype=np.float64)
    l = np.asarray(lengths, dtype=np.float64)
    if m.ndim != 1 or l.ndim != 1:
        raise ValueError("masses and lengths must be one-dimensional")
    if m.size == 0 or m.size != l.size:
        raise ValueError("masses and lengths must have the same non-zero size")
    if not (np.all(np.isfinite(m)) and np.all(np.isfinite(l))):
        raise ValueError("masses and lengths must be finite")
    if np.any(m <= 0.0) or np.any(l <= 0.0):
        raise ValueError("masses and lengths must be strictly positive")
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError("g must be a finite positive number")
    return m, l


def suffix_masses(masses: ArrayLike) -> Vector:
    """Return S with S[j] = sum_{i >= j} masses[i] (O(n))."""
    m = np.asarray(masses, dtype=np.float64)
    return np.cumsum(m[::-1])[::-1]


def inverse_mass_tridiagonal(masses: ArrayLike, lengths: ArrayLike) -> Tuple[Vector, Vector]:
    """
    Closed-form inverse of the linearized mass matrix.

    Returns ``(diag, off)`` of the symmetric tridiagonal matrix M^{-1}:

        (M^{-1})_{jj}     = (1/m_j + 1/m_{j-1}) / l_j^2      (1/m_0 := 0)
        (M^{-1})_{j,j-1}  = -1 / (m_{j-1} l_{j-1} l_j)
    """
    m, l = _validate(masses, lengths)
    inv_m = 1.0 / m
    inv_prev = np.concatenate(([0.0], inv_m[:-1]))
    diag = (inv_m + inv_prev) / l**2
    off = -inv_m[:-1] / (l[:-1] * l[1:])
    return diag, off


def dynamic_matrix_tridiagonal(
    masses: ArrayLike, lengths: ArrayLike, g: float = 9.81
) -> Tuple[Vector, Vector, Vector]:
    """
    Closed-form tridiagonal dynamic matrix Q = M^{-1} K.

    Returns ``(diag, sub, sup)`` with

        Q_{jj}    =  g S_j (1/m_j + 1/m_{j-1}) / l_j
        Q_{j,j-1} = -g S_{j-1} / (m_{j-1} l_j)        (sub-diagonal)
        Q_{j-1,j} = -g S_j     / (m_{j-1} l_{j-1})    (super-diagonal)
    """
    m, l = _validate(masses, lengths, g)
    S = suffix_masses(m)
    inv_m = 1.0 / m
    inv_prev = np.concatenate(([0.0], inv_m[:-1]))
    diag = g * S * (inv_m + inv_prev) / l
    sub = -g * S[:-1] * inv_m[:-1] / l[1:]
    sup = -g * S[1:] * inv_m[:-1] / l[:-1]
    return diag, sub, sup


def dynamic_matrix_dense(masses: ArrayLike, lengths: ArrayLike, g: float = 9.81) -> NDArray[np.float64]:
    """Dense n x n array of Q = M^{-1} K built from the closed form (O(n^2) memory)."""
    d, sub, sup = dynamic_matrix_tridiagonal(masses, lengths, g)
    return np.diag(d) + np.diag(sub, -1) + np.diag(sup, 1)


def apply_dynamic_matrix(
    masses: ArrayLike, lengths: ArrayLike, theta: ArrayLike, g: float = 9.81
) -> Vector:
    """Return Q @ theta in O(n) without forming Q."""
    d, sub, sup = dynamic_matrix_tridiagonal(masses, lengths, g)
    x = np.asarray(theta, dtype=np.float64)
    y = d * x
    y[1:] += sub * x[:-1]
    y[:-1] += sup * x[1:]
    return y


def jacobi_matrix(masses: ArrayLike, lengths: ArrayLike, g: float = 9.81) -> Tuple[Vector, Vector]:
    """
    Symmetric Jacobi form J = K^{1/2} M^{-1} K^{1/2}, similar to Q.

    Returns ``(d, e)``; the off-diagonal entries e are strictly negative, so J is
    an irreducible Jacobi matrix and all squared frequencies are simple.
    """
    m, l = _validate(masses, lengths, g)
    S = suffix_masses(m)
    d, _, _ = dynamic_matrix_tridiagonal(m, l, g)
    e = -g * np.sqrt(S[:-1] * S[1:]) / (m[:-1] * np.sqrt(l[:-1] * l[1:]))
    return d, e


def natural_frequencies(
    masses: ArrayLike,
    lengths: ArrayLike,
    g: float = 9.81,
    select: str = "a",
    select_range: Optional[Tuple[float, float]] = None,
) -> Vector:
    """
    Natural angular frequencies omega_1 < ... < omega_n (rad/s) from the
    tridiagonal Jacobi form.  ``select``/``select_range`` are passed to
    :func:`scipy.linalg.eigh_tridiagonal` (e.g. ``select="i",
    select_range=(0, 9)`` for the ten lowest modes).
    """
    d, e = jacobi_matrix(masses, lengths, g)
    lam = eigh_tridiagonal(d, e, eigvals_only=True, select=select, select_range=select_range)
    return np.sqrt(lam)


def modal_matrix(
    masses: ArrayLike, lengths: ArrayLike, g: float = 9.81
) -> Tuple[Vector, NDArray[np.float64]]:
    """
    Natural frequencies and M-orthonormal mode shapes from the Jacobi form.

    If J w = lambda w with ||w|| = 1, then u = sqrt(lambda) K^{-1/2} w satisfies
    K u = lambda M u and u^T M u = 1.  Returns ``(omega, U)``.
    """
    m, l = _validate(masses, lengths, g)
    d, e = jacobi_matrix(m, l, g)
    lam, W = eigh_tridiagonal(d, e)
    k_diag = g * suffix_masses(m) * l
    U = (W / np.sqrt(k_diag)[:, None]) * np.sqrt(lam)[None, :]
    return np.sqrt(lam), U
