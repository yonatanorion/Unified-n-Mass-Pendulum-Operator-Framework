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
    n = len(masses)
    M = np.zeros((n, n))
    K = np.zeros((n, n))

    for j in range(n):
        for k in range(n):
            # Cumulative mass-sum rule with dimensional lengths
            M[j, k] = np.sum(masses[max(j, k):]) * lengths[j] * lengths[k]
        
        # Diagonal gravitational stiffness
        K[j, j] = np.sum(masses[j:]) * g * lengths[j]

    M_inv = np.linalg.inv(M)
    A = np.block([
        [np.zeros((n, n)), np.eye(n)],
        [-M_inv @ K, np.zeros((n, n))]
    ])
    return A, M, K