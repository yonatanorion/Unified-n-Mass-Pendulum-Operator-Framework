"""Code listing of the paper's appendix, reproduced verbatim (see tests/test_paper.py)."""
import numpy as np
from scipy.linalg import solveh_banded

def dynamical_matrix(m, l, g=9.81):
    """Tridiagonal Q = M^-1 K of Eq. (Q): returns (diag, sub, sup)."""
    S = np.cumsum(m[::-1])[::-1]                 # S_k = mass below rod k
    inv_prev = np.concatenate(([0.0], 1.0 / m[:-1]))
    diag = g * S * (1.0 / m + inv_prev) / l
    sub = -g * S[:-1] / (m[:-1] * l[1:])         # Q[k, k-1]
    sup = -g * S[1:] / (m[:-1] * l[:-1])         # Q[k-1, k]
    return diag, sub, sup

def accelerations(th, om, m, l, g=9.81):
    """Exact nonlinear angular accelerations via rod tensions."""
    inv_m = 1.0 / m
    ab = np.zeros((2, m.size))
    ab[1] = inv_m + np.concatenate(([0.0], inv_m[:-1]))     # B_kk
    ab[0, 1:] = -np.cos(np.diff(th)) * inv_m[:-1]           # B_k,k+1
    r = l * om**2
    r[0] += g * np.cos(th[0])
    tau = solveh_banded(ab, r) if m.size > 1 else r / ab[1]  # tensions
    s = np.sin(np.diff(th))
    acc = np.zeros_like(th)
    acc[:-1] += tau[1:] * s * inv_m[:-1]
    acc[1:] -= tau[:-1] * s * inv_m[:-1]
    acc[0] -= g * np.sin(th[0])
    return acc / l
