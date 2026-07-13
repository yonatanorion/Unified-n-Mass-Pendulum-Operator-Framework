"""
Module B: Operator-Based Linear Propagation
"""

import numpy as np


def peano_baker_step(A, dt, N=20, scaling_threshold=0.5):
    """
    Approximate ``exp(A*dt)`` with a scaled Taylor/Peano--Baker series.

    The direct truncated series can lose accuracy when ``||A*dt||`` is
    large.  We first scale ``B = A*dt/2**s`` so that its infinity norm is
    at most ``scaling_threshold``, evaluate the order-``N`` series for
    ``exp(B)``, and then square the result ``s`` times.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if N < 0:
        raise ValueError("N must be non-negative")
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")
    if not np.isfinite(scaling_threshold) or scaling_threshold <= 0.0:
        raise ValueError("scaling_threshold must be finite and positive")

    dim = A.shape[0]
    B = A * dt
    norm_B = np.linalg.norm(B, ord=np.inf)
    scaling_steps = 0
    if norm_B > scaling_threshold:
        scaling_steps = int(np.ceil(np.log2(norm_B / scaling_threshold)))
        B = B / (2**scaling_steps)

    Phi = np.eye(dim)
    term = np.eye(dim)
    
    for k in range(1, N + 1):
        term = (term @ B) / k
        Phi += term

    for _ in range(scaling_steps):
        Phi = Phi @ Phi

    return Phi


def propagate_linear(Phi, R0, t):
    """
    Propagate linear system using precomputed transition matrix Phi.
    Efficient O(n^2) propagation reusing Phi.
    """
    Phi = np.asarray(Phi, dtype=float)
    R0 = np.asarray(R0, dtype=float)
    if Phi.ndim != 2 or Phi.shape[0] != Phi.shape[1]:
        raise ValueError("Phi must be a square matrix")
    if R0.ndim != 1 or Phi.shape[0] != R0.size:
        raise ValueError("R0 must be a vector compatible with Phi")
    if len(t) == 0:
        raise ValueError("t must contain at least one sample")

    R = np.zeros((len(t), R0.size))
    R[0] = R0
    
    for i in range(1, len(t)):
        R[i] = Phi @ R[i-1]
        
    return R
