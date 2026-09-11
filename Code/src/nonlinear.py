"""
Nonlinear n-mass pendulum equations in absolute-angle coordinates.
"""

from typing import Sequence, Tuple, Union
import numpy as np
from numpy.typing import NDArray

ArrayLike = Union[Sequence[float], NDArray[np.float64]]


def _parameters(
    masses: ArrayLike,
    lengths: ArrayLike,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    masses_arr = np.asarray(masses, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.float64)
    if masses_arr.ndim != 1 or lengths_arr.ndim != 1:
        raise ValueError("masses and lengths must be one-dimensional")
    if masses_arr.size == 0 or masses_arr.shape != lengths_arr.shape:
        raise ValueError("masses and lengths must have the same non-zero size")
    if np.any(masses_arr <= 0.0) or np.any(lengths_arr <= 0.0):
        raise ValueError("masses and lengths must be strictly positive")
    suffix_masses = np.cumsum(masses_arr[::-1])[::-1]
    return masses_arr, lengths_arr, suffix_masses


def nonlinear_terms(
    theta: ArrayLike,
    angular_velocity: ArrayLike,
    masses: ArrayLike,
    lengths: ArrayLike,
    g: float = 9.81,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Return the exact nonlinear mass matrix M(theta), velocity vector c(theta, omega),
    and gravity vector g(theta).

    The velocity term is:
        c_d = sum_k S[max(d,k)] l_d l_k omega_k^2 sin(theta_d - theta_k)

    with no additional index-dependent sign. This convention matches:
        M(theta) \\ddot{theta} + c(theta, omega) + g(theta) = 0.

    Parameters
    ----------
    theta : Sequence[float] or NDArray of shape (n,)
        Joint angle vector (rad).
    angular_velocity : Sequence[float] or NDArray of shape (n,)
        Joint angular velocity vector (rad/s).
    masses : Sequence[float] or NDArray of shape (n,)
        Point masses (kg).
    lengths : Sequence[float] or NDArray of shape (n,)
        Link lengths (m).
    g : float, optional
        Gravitational acceleration (m/s^2), default is 9.81.

    Returns
    -------
    M : NDArray[np.float64] of shape (n, n)
        Nonlinear configuration-dependent mass matrix.
    c : NDArray[np.float64] of shape (n,)
        Coriolis / centrifugal torque vector.
    gravity : NDArray[np.float64] of shape (n,)
        Gravitational torque vector.
    """
    _, lengths_arr, suffix_masses = _parameters(masses, lengths)
    theta_arr = np.asarray(theta, dtype=np.float64)
    omega_arr = np.asarray(angular_velocity, dtype=np.float64)
    n = lengths_arr.size

    if theta_arr.shape != (n,) or omega_arr.shape != (n,):
        raise ValueError("theta and angular_velocity must contain n values")

    M = np.empty((n, n), dtype=np.float64)
    c = np.zeros(n, dtype=np.float64)
    gravity = suffix_masses * g * lengths_arr * np.sin(theta_arr)

    for d in range(n):
        for k in range(n):
            coefficient = suffix_masses[max(d, k)] * lengths_arr[d] * lengths_arr[k]
            angle_difference = theta_arr[d] - theta_arr[k]
            M[d, k] = coefficient * np.cos(angle_difference)
            c[d] += coefficient * (omega_arr[k] ** 2) * np.sin(angle_difference)

    return M, c, gravity


def state_derivative(
    _time: float,
    state: ArrayLike,
    masses: ArrayLike,
    lengths: ArrayLike,
    g: float = 9.81,
) -> NDArray[np.float64]:
    """
    Evaluate the exact nonlinear first-order state equation: dR/dt = [omega, -M^{-1}(c + g)].

    Parameters
    ----------
    _time : float
        Current simulation time (s), unused for autonomous system.
    state : Sequence[float] or NDArray of shape (2n,)
        Current state vector [theta, omega].
    masses : Sequence[float] or NDArray of shape (n,)
        Point masses (kg).
    lengths : Sequence[float] or NDArray of shape (n,)
        Link lengths (m).
    g : float, optional
        Gravitational acceleration (m/s^2), default is 9.81.

    Returns
    -------
    dstate_dt : NDArray[np.float64] of shape (2n,)
        Time derivative of the state vector.
    """
    state_arr = np.asarray(state, dtype=np.float64)
    masses_arr = np.asarray(masses, dtype=np.float64)
    n = masses_arr.size
    if state_arr.shape != (2 * n,):
        raise ValueError("state must contain n angles followed by n velocities")

    theta = state_arr[:n]
    angular_velocity = state_arr[n:]
    M, c, gravity = nonlinear_terms(
        theta, angular_velocity, masses_arr, lengths, g
    )
    angular_acceleration = np.linalg.solve(M, -(c + gravity))
    return np.concatenate((angular_velocity, angular_acceleration))


def total_energy(
    state: ArrayLike,
    masses: ArrayLike,
    lengths: ArrayLike,
    g: float = 9.81,
) -> float:
    """
    Return exact kinetic plus potential energy E = T + U for a state vector.

    Parameters
    ----------
    state : Sequence[float] or NDArray of shape (2n,)
        State vector [theta, omega].
    masses : Sequence[float] or NDArray of shape (n,)
        Point masses (kg).
    lengths : Sequence[float] or NDArray of shape (n,)
        Link lengths (m).
    g : float, optional
        Gravitational acceleration (m/s^2), default is 9.81.

    Returns
    -------
    energy : float
        Total mechanical energy (Joules).
    """
    state_arr = np.asarray(state, dtype=np.float64)
    _, lengths_arr, suffix_masses = _parameters(masses, lengths)
    n = lengths_arr.size
    if state_arr.shape != (2 * n,):
        raise ValueError("state must contain n angles followed by n velocities")

    theta = state_arr[:n]
    angular_velocity = state_arr[n:]
    M, _, _ = nonlinear_terms(theta, angular_velocity, masses, lengths_arr, g)
    kinetic = 0.5 * float(angular_velocity @ M @ angular_velocity)
    potential = -g * float(np.sum(suffix_masses * lengths_arr * np.cos(theta)))
    return kinetic + potential
