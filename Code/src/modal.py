"""
Modal solution of the linearized n-mass pendulum.
"""

from typing import Sequence, Union
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

ArrayLike = Union[Sequence[float], NDArray[np.float64]]


def solve_modal(
    M: NDArray[np.float64],
    K: NDArray[np.float64],
    R0: ArrayLike,
    t: ArrayLike,
) -> NDArray[np.float64]:
    """
    Solve the undamped multi-degree-of-freedom system ``M \\ddot{\\theta} + K \\theta = 0``
    via exact modal decoupling.

    Uses ``scipy.linalg.eigh(K, M)`` to compute the M-orthonormal modal matrix ``V``
    satisfying ``V.T @ M @ V = I`` and ``V.T @ K @ V = diag(\\omega^2)``.

    Parameters
    ----------
    M : NDArray[np.float64] of shape (n, n)
        Symmetric positive-definite mass matrix.
    K : NDArray[np.float64] of shape (n, n)
        Symmetric positive-definite stiffness matrix.
    R0 : Sequence[float] or NDArray of shape (2n,)
        Initial state vector [theta_0, dot{theta}_0].
    t : Sequence[float] or NDArray of shape (n_steps,)
        Time grid.

    Returns
    -------
    solution : NDArray[np.float64] of shape (n_steps, 2n)
        Array of state trajectories ordered as [theta(t), dot{theta}(t)].

    Raises
    ------
    ValueError
        If M and K shapes do not match, R0 has invalid dimension, or eigenvalues <= 0.
    """
    M_arr = np.asarray(M, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    R0_arr = np.asarray(R0, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)

    if M_arr.ndim != 2 or M_arr.shape[0] != M_arr.shape[1] or K_arr.shape != M_arr.shape:
        raise ValueError("M and K must be square matrices of the same shape")

    n = M_arr.shape[0]
    if R0_arr.shape != (2 * n,):
        raise ValueError("R0 must contain n angles followed by n velocities")
    if t_arr.ndim != 1:
        raise ValueError("t must be one-dimensional")

    omega_squared, modes = eigh(K_arr, M_arr)
    if np.any(omega_squared <= 0.0):
        raise ValueError("the generalized eigenvalues must be positive")
    omega = np.sqrt(omega_squared)

    theta0 = R0_arr[:n]
    velocity0 = R0_arr[n:]
    # Written as explicit contractions to keep the modal normalization clear:
    # q = V.T M theta. This also avoids spurious BLAS floating-point warnings
    # emitted by some Accelerate/NumPy combinations after ``eigh``.
    M_theta0 = np.sum(M_arr * theta0[None, :], axis=1)
    M_velocity0 = np.sum(M_arr * velocity0[None, :], axis=1)
    q0 = np.sum(modes * M_theta0[:, None], axis=0)
    qdot0 = np.sum(modes * M_velocity0[:, None], axis=0)

    phase = np.outer(t_arr, omega)
    q = np.cos(phase) * q0 + np.sin(phase) * (qdot0 / omega)
    qdot = -np.sin(phase) * (q0 * omega) + np.cos(phase) * qdot0

    theta = np.sum(q[:, None, :] * modes[None, :, :], axis=2)
    velocity = np.sum(qdot[:, None, :] * modes[None, :, :], axis=2)
    return np.hstack((theta, velocity))
