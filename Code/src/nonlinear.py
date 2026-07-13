"""Nonlinear n-mass pendulum equations in absolute-angle coordinates."""

import numpy as np


def _parameters(masses, lengths):
    masses = np.asarray(masses, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    if masses.ndim != 1 or lengths.ndim != 1:
        raise ValueError("masses and lengths must be one-dimensional")
    if masses.size == 0 or masses.shape != lengths.shape:
        raise ValueError("masses and lengths must have the same non-zero size")
    if np.any(masses <= 0.0) or np.any(lengths <= 0.0):
        raise ValueError("masses and lengths must be strictly positive")
    suffix_masses = np.cumsum(masses[::-1])[::-1]
    return masses, lengths, suffix_masses


def nonlinear_terms(theta, angular_velocity, masses, lengths, g=9.81):
    """Return the exact mass matrix, velocity vector, and gravity vector.

    The velocity term is

        c_d = sum_k S[max(d,k)] l_d l_k omega_k^2 sin(theta_d-theta_k),

    with no additional index-dependent sign.  This convention matches
    ``M(theta) theta_ddot + c(theta, omega) + g(theta) = 0``.
    """
    _, lengths, suffix_masses = _parameters(masses, lengths)
    theta = np.asarray(theta, dtype=float)
    angular_velocity = np.asarray(angular_velocity, dtype=float)
    n = lengths.size

    if theta.shape != (n,) or angular_velocity.shape != (n,):
        raise ValueError("theta and angular_velocity must contain n values")

    M = np.empty((n, n))
    c = np.zeros(n)
    gravity = suffix_masses * g * lengths * np.sin(theta)

    for d in range(n):
        for k in range(n):
            coefficient = (
                suffix_masses[max(d, k)] * lengths[d] * lengths[k]
            )
            angle_difference = theta[d] - theta[k]
            M[d, k] = coefficient * np.cos(angle_difference)
            c[d] += (
                coefficient
                * angular_velocity[k] ** 2
                * np.sin(angle_difference)
            )

    return M, c, gravity


def state_derivative(_time, state, masses, lengths, g=9.81):
    """Evaluate the exact nonlinear first-order state equation."""
    state = np.asarray(state, dtype=float)
    n = len(masses)
    if state.shape != (2 * n,):
        raise ValueError("state must contain n angles followed by n velocities")

    theta = state[:n]
    angular_velocity = state[n:]
    M, c, gravity = nonlinear_terms(
        theta, angular_velocity, masses, lengths, g
    )
    angular_acceleration = np.linalg.solve(M, -(c + gravity))
    return np.concatenate((angular_velocity, angular_acceleration))


def total_energy(state, masses, lengths, g=9.81):
    """Return exact kinetic plus potential energy for one state."""
    state = np.asarray(state, dtype=float)
    _, lengths, suffix_masses = _parameters(masses, lengths)
    n = lengths.size
    if state.shape != (2 * n,):
        raise ValueError("state must contain n angles followed by n velocities")

    theta = state[:n]
    angular_velocity = state[n:]
    M, _, _ = nonlinear_terms(theta, angular_velocity, masses, lengths, g)
    kinetic = 0.5 * angular_velocity @ M @ angular_velocity
    potential = -g * np.sum(suffix_masses * lengths * np.cos(theta))
    return kinetic + potential
