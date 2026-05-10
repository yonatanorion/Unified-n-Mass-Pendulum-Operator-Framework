"""
Module B: Operator-Based Linear Propagation
"""

import numpy as np
from math import factorial

def peano_baker_step(A, dt, N=20):
    """
    Compute Transition matrix Phi(dt) = sum_{k=0}^N (A*dt)^k/k!
    """
    dim = A.shape[0]
    Phi = np.eye(dim)
    A_pow = np.eye(dim)
    
    for k in range(1, N + 1):
        A_pow = A_pow @ A
        Phi += (dt**k / factorial(k)) * A_pow
        
    return Phi

def propagate_linear(Phi, R0, t):
    """
    Propagate linear system using precomputed transition matrix Phi.
    Efficient O(n^2) propagation reusing Phi.
    """
    R = np.zeros((len(t), len(R0)))
    R[0] = R0
    
    for i in range(1, len(t)):
        R[i] = Phi @ R[i-1]
        
    return R