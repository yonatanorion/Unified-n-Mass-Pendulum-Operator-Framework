"""
Exact nonlinear dynamics of the planar n-mass pendulum in O(n) operations.

Newtonian formulation.  Link k joins mass k-1 to mass k (mass 0 is the fixed
pivot), with unit vector e_k = (sin th_k, -cos th_k), normal n_k = (cos th_k,
sin th_k) and rod tension T_k (T_{n+1} := 0).  Projecting the relative
accelerations of adjacent masses on e_k gives a symmetric positive-definite
tridiagonal system for the tensions,

    (1/m_k + 1/m_{k-1}) T_k - cos(th_{k+1}-th_k) T_{k+1}/m_k
                            - cos(th_k-th_{k-1}) T_{k-1}/m_{k-1}
        = l_k thdot_k^2 + g cos(th_1) delta_{k1},           (1/m_0 := 0)

whose matrix B(theta) reduces to (E^T D E)^{-1} of the linear theory at
theta = 0.  Projecting on n_k then gives the angular accelerations explicitly,

    l_k thddot_k = T_{k+1} sin(th_{k+1}-th_k)/m_k
                 - T_{k-1} sin(th_k-th_{k-1})/m_{k-1} - g sin(th_1) delta_{k1}.

Both steps cost O(n), versus O(n^3) for forming and factorizing M(theta).

The module also provides RATTLE, a second-order symplectic and time-reversible
integrator in Cartesian coordinates with the rod constraints enforced by
Lagrange multipliers (the tensions).  Every linear system it solves is
tridiagonal, so each step costs O(n) per Newton iteration.
"""

from typing import Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded, solveh_banded

from .structure import _validate

ArrayLike = Union[Sequence[float], NDArray[np.float64]]
Vector = NDArray[np.float64]


# ---------------------------------------------------------------- angles
def tension_system(theta, omega, masses, lengths, g=9.81):
    """Return (diag, off, rhs) of the tridiagonal tension system B(theta) T = r."""
    m, l = _validate(masses, lengths, g)
    th = np.asarray(theta, dtype=np.float64)
    om = np.asarray(omega, dtype=np.float64)
    inv_m = 1.0 / m
    inv_prev = np.concatenate(([0.0], inv_m[:-1]))
    diag = inv_m + inv_prev
    off = -np.cos(np.diff(th)) * inv_m[:-1]          # couples k and k+1
    rhs = l * om**2
    rhs[0] += g * np.cos(th[0])
    return diag, off, rhs


def tensions(theta, omega, masses, lengths, g=9.81) -> Vector:
    """Rod tensions T_1..T_n (N) for the state (theta, omega), in O(n)."""
    diag, off, rhs = tension_system(theta, omega, masses, lengths, g)
    if diag.size == 1:
        return rhs / diag
    ab = np.zeros((2, diag.size))
    ab[0, 1:] = off
    ab[1] = diag
    return solveh_banded(ab, rhs)


def angular_acceleration(theta, omega, masses, lengths, g=9.81) -> Vector:
    """Exact nonlinear angular accelerations in O(n)."""
    m, l = _validate(masses, lengths, g)
    th = np.asarray(theta, dtype=np.float64)
    T = tensions(th, omega, m, l, g)
    s = np.sin(np.diff(th))                           # sin(th_{k+1} - th_k)
    acc = np.zeros_like(th)
    acc[:-1] += T[1:] * s / m[:-1]
    acc[1:] -= T[:-1] * s / m[:-1]
    acc[0] -= g * np.sin(th[0])
    return acc / l


def state_derivative(_t, state, masses, lengths, g=9.81) -> Vector:
    """dR/dt = [omega, thddot] for R = [theta, omega], O(n) per call."""
    R = np.asarray(state, dtype=np.float64)
    n = R.size // 2
    return np.concatenate((R[n:], angular_acceleration(R[:n], R[n:], masses, lengths, g)))


def energy(theta, omega, masses, lengths, g=9.81) -> float:
    """Exact total energy T + U in O(n)."""
    m, l = _validate(masses, lengths, g)
    th = np.asarray(theta, dtype=np.float64)
    om = np.asarray(omega, dtype=np.float64)
    vx = np.cumsum(l * om * np.cos(th))
    vy = np.cumsum(l * om * np.sin(th))
    y = -np.cumsum(l * np.cos(th))
    return float(0.5 * np.sum(m * (vx**2 + vy**2)) + g * np.sum(m * y))


# ------------------------------------------------------------ Cartesian
def to_cartesian(theta, omega, lengths) -> Tuple[NDArray, NDArray]:
    """Positions q (n,2) and velocities v (n,2) of the masses."""
    l = np.asarray(lengths, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    om = np.asarray(omega, dtype=np.float64)
    q = np.cumsum(np.stack((l * np.sin(th), -l * np.cos(th)), axis=1), axis=0)
    v = np.cumsum(np.stack((l * om * np.cos(th), l * om * np.sin(th)), axis=1), axis=0)
    return q, v


def to_angles(q, v) -> Tuple[Vector, Vector]:
    """Inverse of :func:`to_cartesian` (angles and angular velocities)."""
    d = np.diff(np.vstack(([0.0, 0.0], q)), axis=0)
    dv = np.diff(np.vstack(([0.0, 0.0], v)), axis=0)
    th = np.arctan2(d[:, 0], -d[:, 1])
    l2 = np.sum(d**2, axis=1)
    om = (d[:, 0] * dv[:, 1] - d[:, 1] * dv[:, 0]) / l2
    return th, om


def cartesian_energy(q, v, masses, g=9.81) -> float:
    m = np.asarray(masses, dtype=np.float64)
    return float(0.5 * np.sum(m * np.sum(v**2, axis=1)) + g * np.sum(m * q[:, 1]))


def _links(q):
    return np.diff(np.vstack(([0.0, 0.0], q)), axis=0)       # d_k = q_k - q_{k-1}


def _constraint_force(d, lam, inv_m):
    """Acceleration -M^{-1} G^T lam for constraints g_k = (|d_k|^2 - l_k^2)/2."""
    f = -lam[:, None] * d
    f[:-1] += lam[1:, None] * d[1:]
    return f * inv_m[:, None]


def _gmg_bands(da, db, inv_m):
    """Tridiagonal G(a) M^{-1} G(b)^T as (super, diag, sub) for solve_banded."""
    inv_prev = np.concatenate(([0.0], inv_m[:-1]))
    diag = np.sum(da * db, axis=1) * (inv_m + inv_prev)
    sup = -np.sum(da[:-1] * db[1:], axis=1) * inv_m[:-1]      # (j, j+1)
    sub = -np.sum(da[1:] * db[:-1], axis=1) * inv_m[:-1]      # (j+1, j)
    ab = np.zeros((3, diag.size))
    ab[0, 1:] = sup
    ab[1] = diag
    ab[2, :-1] = sub
    return ab


def rattle_propagate(
    theta0, omega0, masses, lengths, dt, n_steps, g=9.81, save_every=1,
    tol=1e-13, max_iter=20,
):
    """
    RATTLE integration of the constrained Cartesian equations.

    Returns ``(t, theta, omega, newton_iterations)`` sampled every
    ``save_every`` steps.  Each step solves the position constraints by Newton's
    method with a tridiagonal Jacobian and the velocity constraints by one
    symmetric tridiagonal solve, so the cost is O(n) per iteration.
    """
    m, l = _validate(masses, lengths, g)
    if dt <= 0 or n_steps < 0 or save_every < 1:
        raise ValueError("dt must be positive, n_steps >= 0 and save_every >= 1")
    inv_m = 1.0 / m
    n = m.size
    q, v = to_cartesian(theta0, omega0, l)
    grav = np.zeros((n, 2)); grav[:, 1] = -g
    h, h2 = dt, 0.5 * dt * dt
    n_out = n_steps // save_every + 1
    T = np.empty(n_out); TH = np.empty((n_out, n)); OM = np.empty((n_out, n))
    T[0] = 0.0; TH[0], OM[0] = to_angles(q, v)
    iters = 0; k = 1
    for step in range(1, n_steps + 1):
        d0 = _links(q)
        q_free = q + h * v + h2 * grav
        lam = np.zeros(n)
        for it in range(max_iter):
            q_new = q_free + h2 * _constraint_force(d0, lam, inv_m)
            d1 = _links(q_new)
            res = 0.5 * (np.sum(d1**2, axis=1) - l**2)
            if np.max(np.abs(res) / l**2) < tol:
                break
            # d res / d lam = -h2 * G(q_new) M^{-1} G(q)^T
            J = -h2 * _gmg_bands(d1, d0, inv_m)
            lam -= solve_banded((1, 1), J, res) if n > 1 else res / J[1]
        else:
            raise RuntimeError("RATTLE position projection did not converge")
        iters += it
        v_half = (q_new - q) / h
        d1 = _links(q_new)
        w = v_half + 0.5 * h * grav
        dw = _links(w)
        rhs = np.sum(d1 * dw, axis=1)                              # G(q_new) w
        ab = 0.5 * h * _gmg_bands(d1, d1, inv_m)
        sym = np.zeros((2, n)); sym[0, 1:] = ab[0, 1:]; sym[1] = ab[1]
        mu = solveh_banded(sym, rhs) if n > 1 else rhs / sym[1]
        v = w + 0.5 * h * _constraint_force(d1, mu, inv_m)
        q = q_new
        if step % save_every == 0:
            T[k] = step * dt; TH[k], OM[k] = to_angles(q, v); k += 1
    return T, TH, OM, iters
