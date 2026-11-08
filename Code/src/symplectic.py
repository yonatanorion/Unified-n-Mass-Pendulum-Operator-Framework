"""
Structure-preserving O(n) time stepping for the linearized n-mass pendulum.

The linear equations M theta'' + K theta = 0 are equivalent to theta'' = -Q theta
with the closed-form tridiagonal Q = M^{-1} K (src/structure.py).  The
Stormer--Verlet (leapfrog) scheme applied to this form costs O(n) per step,
needs no precomputation or dense storage, is symplectic and time-reversible,
and keeps the energy error bounded (no secular drift) for stable step sizes
dt < 2 / omega_max.

The energy itself is evaluated in O(n) through the factorization
M = L E^T D E L:  theta_dot^T M theta_dot = sum_i m_i (sum_{k<=i} l_k theta_dot_k)^2.
"""

from typing import Optional, Tuple, Union, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal

from .structure import _validate, dynamic_matrix_tridiagonal, jacobi_matrix, suffix_masses

ArrayLike = Union[Sequence[float], NDArray[np.float64]]
Vector = NDArray[np.float64]


def _tridiag_apply(d: Vector, sub: Vector, sup: Vector, x: Vector) -> Vector:
    y = d * x
    y[1:] += sub * x[:-1]
    y[:-1] += sup * x[1:]
    return y


def linear_energy(
    masses: ArrayLike, lengths: ArrayLike, theta: ArrayLike, omega: ArrayLike, g: float = 9.81
) -> float:
    """Quadratic energy 1/2 omega^T M omega + 1/2 theta^T K theta in O(n)."""
    m, l = _validate(masses, lengths, g)
    th = np.asarray(theta, dtype=np.float64)
    om = np.asarray(omega, dtype=np.float64)
    v = np.cumsum(l * om)                     # (E L omega)_i: linearized velocity of mass i
    kinetic = 0.5 * np.sum(m * v**2)
    potential = 0.5 * g * np.sum(suffix_masses(m) * l * th**2)
    return float(kinetic + potential)


def max_frequency(masses: ArrayLike, lengths: ArrayLike, g: float = 9.81) -> float:
    """Largest natural frequency omega_n, by bisection on the Jacobi matrix (O(n))."""
    d, e = jacobi_matrix(masses, lengths, g)
    n = d.size
    lam = eigh_tridiagonal(d, e, eigvals_only=True, select="i", select_range=(n - 1, n - 1))
    return float(np.sqrt(lam[0]))


def verlet_propagate(
    masses: ArrayLike,
    lengths: ArrayLike,
    theta0: ArrayLike,
    omega0: ArrayLike,
    dt: float,
    n_steps: int,
    g: float = 9.81,
    save_every: int = 1,
    check_stability: bool = True,
) -> Tuple[Vector, NDArray[np.float64], NDArray[np.float64]]:
    """
    Stormer--Verlet (velocity form) integration of theta'' = -Q theta.

    Returns ``(t, theta, omega)`` sampled every ``save_every`` steps
    (including t = 0).  Cost: O(n) per step, O(n) memory besides the output.

    Raises
    ------
    ValueError
        If ``check_stability`` and dt >= 2 / omega_max (linear instability).
    """
    m, l = _validate(masses, lengths, g)
    if dt <= 0 or n_steps < 0 or save_every < 1:
        raise ValueError("dt must be positive, n_steps >= 0 and save_every >= 1")
    if check_stability:
        w_max = max_frequency(m, l, g)
        if dt * w_max >= 2.0:
            raise ValueError(f"unstable step: dt*omega_max = {dt * w_max:.3f} >= 2")
    d, sub, sup = dynamic_matrix_tridiagonal(m, l, g)
    th = np.array(theta0, dtype=np.float64)
    om = np.array(omega0, dtype=np.float64)
    if th.shape != (m.size,) or om.shape != (m.size,):
        raise ValueError("theta0 and omega0 must have length n")

    n_out = n_steps // save_every + 1
    T = np.empty(n_out)
    TH = np.empty((n_out, m.size))
    OM = np.empty((n_out, m.size))
    T[0], TH[0], OM[0] = 0.0, th, om
    acc = -_tridiag_apply(d, sub, sup, th)
    half = 0.5 * dt
    k = 1
    for step in range(1, n_steps + 1):
        om += half * acc
        th += dt * om
        acc = -_tridiag_apply(d, sub, sup, th)
        om += half * acc
        if step % save_every == 0:
            T[k], TH[k], OM[k] = step * dt, th, om
            k += 1
    return T, TH, OM
