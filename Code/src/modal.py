"""Modal solution of the linearized n-mass pendulum."""

import numpy as np
from scipy.linalg import eigh


def solve_modal(M, K, R0, t):
    """Solve ``M theta_ddot + K theta = 0`` by modal decoupling.

    ``scipy.linalg.eigh(K, M)`` returns an M-orthonormal basis ``V`` such
    that ``V.T @ M @ V = I`` and ``V.T @ K @ V = diag(omega**2)``.
    The returned array follows the state ordering ``[theta, theta_dot]``.
    """
    M = np.asarray(M, dtype=float)
    K = np.asarray(K, dtype=float)
    R0 = np.asarray(R0, dtype=float)
    t = np.asarray(t, dtype=float)

    if M.ndim != 2 or M.shape[0] != M.shape[1] or K.shape != M.shape:
        raise ValueError("M and K must be square matrices of the same shape")

    n = M.shape[0]
    if R0.shape != (2 * n,):
        raise ValueError("R0 must contain n angles followed by n velocities")
    if t.ndim != 1:
        raise ValueError("t must be one-dimensional")

    omega_squared, modes = eigh(K, M)
    if np.any(omega_squared <= 0.0):
        raise ValueError("the generalized eigenvalues must be positive")
    omega = np.sqrt(omega_squared)

    theta0 = R0[:n]
    velocity0 = R0[n:]
    # Written as explicit contractions to keep the modal normalization clear:
    # q = V.T M theta.  This also avoids spurious BLAS floating-point warnings
    # emitted by some Accelerate/NumPy combinations after ``eigh``.
    M_theta0 = np.sum(M * theta0[None, :], axis=1)
    M_velocity0 = np.sum(M * velocity0[None, :], axis=1)
    q0 = np.sum(modes * M_theta0[:, None], axis=0)
    qdot0 = np.sum(modes * M_velocity0[:, None], axis=0)

    phase = np.outer(t, omega)
    q = np.cos(phase) * q0 + np.sin(phase) * (qdot0 / omega)
    qdot = -np.sin(phase) * (q0 * omega) + np.cos(phase) * qdot0

    theta = np.sum(q[:, None, :] * modes[None, :, :], axis=2)
    velocity = np.sum(qdot[:, None, :] * modes[None, :, :], axis=2)
    return np.hstack((theta, velocity))
