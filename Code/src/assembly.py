"""
Module A: Linear System Assembly via Cumulative Mass-Sum Rule
"""

import numpy as np


def assemble_system(masses, lengths, g=9.81):
    """
    Construct linear state-space matrix A for an n-mass pendulum.
    
    Returns
    -------
    A : ndarray (2n, 2n) - State matrix
    M : ndarray (n, n) - Mass matrix
    K : ndarray (n, n) - Stiffness matrix (diagonal)
    """
    masses = np.asarray(masses, dtype=float)
    lengths = np.asarray(lengths, dtype=float)

    if masses.ndim != 1 or lengths.ndim != 1:
        raise ValueError("masses and lengths must be one-dimensional")
    if masses.size == 0 or masses.size != lengths.size:
        raise ValueError("masses and lengths must have the same non-zero size")
    if np.any(masses <= 0.0) or np.any(lengths <= 0.0):
        raise ValueError("masses and lengths must be strictly positive")
    if not np.isfinite(g) or g <= 0.0:
        raise ValueError("g must be a finite positive number")

    n = masses.size
    M = np.zeros((n, n))
    K = np.zeros((n, n))

    # S[j] = sum_{i=j}^{n-1} masses[i].  Computing these suffix sums once
    # is what makes the matrix assembly O(n^2), rather than O(n^3).
    S = np.cumsum(masses[::-1])[::-1]

    for j in range(n):
        for k in range(n):
            # Cumulative mass-sum rule with dimensional lengths
            M[j, k] = S[max(j, k)] * lengths[j] * lengths[k]
        
        # Diagonal gravitational stiffness
        K[j, j] = S[j] * g * lengths[j]

    # Solving M X = K is more stable than forming inv(M) explicitly.
    M_solve_K = np.linalg.solve(M, K)
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-M_solve_K, np.zeros((n, n))]
    ])
    return A, M, K
