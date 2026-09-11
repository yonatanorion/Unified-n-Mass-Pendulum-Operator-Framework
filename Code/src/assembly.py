"""
Module A: Linear System Assembly via Cumulative Mass-Sum Rule
"""

from typing import Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray

ArrayLike = Union[Sequence[float], NDArray[np.float64]]


def assemble_system(
    masses: ArrayLike,
    lengths: ArrayLike,
    g: float = 9.81,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Construct linear state-space matrix A, mass matrix M, and stiffness matrix K
    for an arbitrary n-mass pendulum using the cumulative mass-sum rule.

    Parameters
    ----------
    masses : Sequence[float] or NDArray
        Point masses of the links [m_1, m_2, ..., m_n] (kg).
    lengths : Sequence[float] or NDArray
        Lengths of the links [l_1, l_2, ..., l_n] (m).
    g : float, optional
        Gravitational acceleration constant (m/s^2), default is 9.81.

    Returns
    -------
    A : NDArray[np.float64] of shape (2n, 2n)
        Linearized continuous-time state-space matrix.
    M : NDArray[np.float64] of shape (n, n)
        Symmetric positive-definite mass (inertia) matrix.
    K : NDArray[np.float64] of shape (n, n)
        Diagonal gravitational stiffness matrix.

    Raises
    ------
    ValueError
        If inputs have invalid shapes, non-positive values, or non-finite numbers.
    """
    masses_arr = np.asarray(masses, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.float64)

    if masses_arr.ndim != 1 or lengths_arr.ndim != 1:
        raise ValueError("masses and lengths must be one-dimensional")
    if masses_arr.size == 0 or masses_arr.size != lengths_arr.size:
        raise ValueError("masses and lengths must have the same non-zero size")
    if np.any(masses_arr <= 0.0) or np.any(lengths_arr <= 0.0):
        raise ValueError("masses and lengths must be strictly positive")
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError("g must be a finite positive number")

    n = masses_arr.size
    M = np.zeros((n, n), dtype=np.float64)
    K = np.zeros((n, n), dtype=np.float64)

    # S[j] = sum_{i=j}^{n-1} masses[i]. Computing these suffix sums once
    # is what makes the matrix assembly O(n^2), rather than O(n^3).
    S = np.cumsum(masses_arr[::-1])[::-1]

    for j in range(n):
        for k in range(n):
            # Cumulative mass-sum rule with dimensional lengths
            M[j, k] = S[max(j, k)] * lengths_arr[j] * lengths_arr[k]

        # Diagonal gravitational stiffness
        K[j, j] = S[j] * g * lengths_arr[j]

    # Solving M X = K is more stable than forming inv(M) explicitly.
    M_solve_K = np.linalg.solve(M, K)
    A = np.block([
        [np.zeros((n, n), dtype=np.float64), np.eye(n, dtype=np.float64)],
        [-M_solve_K, np.zeros((n, n), dtype=np.float64)],
    ])
    return A, M, K
