"""
Module B: Operator-Based Linear Propagation
"""

from typing import Sequence, Union
import numpy as np
from numpy.typing import NDArray

ArrayLike = Union[Sequence[float], NDArray[np.float64]]


def peano_baker_step(
    A: NDArray[np.float64],
    dt: float,
    N: int = 20,
    scaling_threshold: float = 0.5,
) -> NDArray[np.float64]:
    """
    Approximate the matrix exponential transition operator ``exp(A*dt)``
    using a scaled-and-squared Taylor / Peano--Baker series expansion.

    The direct truncated series can lose accuracy when ``||A*dt||`` is
    large. We first scale ``B = A*dt/2**s`` so that its infinity norm is
    at most ``scaling_threshold``, evaluate the order-``N`` series for
    ``exp(B)``, and then square the result ``s`` times.

    Parameters
    ----------
    A : NDArray[np.float64] of shape (2n, 2n)
        Continuous-time state-space matrix.
    dt : float
        Time step size (s).
    N : int, optional
        Order of Taylor series truncation, default is 20.
    scaling_threshold : float, optional
        Infinity norm threshold for scaling, default is 0.5.

    Returns
    -------
    Phi : NDArray[np.float64] of shape (2n, 2n)
        Discrete-time state transition matrix Phi(dt) = exp(A*dt).

    Raises
    ------
    ValueError
        If matrix A is not square, N is negative, or dt/threshold is invalid.
    """
    A_arr = np.asarray(A, dtype=np.float64)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError("A must be a square matrix")
    if N < 0:
        raise ValueError("N must be non-negative")
    if not np.isfinite(dt):
        raise ValueError("dt must be finite")
    if not np.isfinite(scaling_threshold) or scaling_threshold <= 0.0:
        raise ValueError("scaling_threshold must be finite and positive")

    dim = A_arr.shape[0]
    B = A_arr * dt
    norm_B = np.linalg.norm(B, ord=np.inf)
    scaling_steps = 0
    if norm_B > scaling_threshold:
        scaling_steps = int(np.ceil(np.log2(norm_B / scaling_threshold)))
        B = B / (2**scaling_steps)

    Phi = np.eye(dim, dtype=np.float64)
    term = np.eye(dim, dtype=np.float64)

    for k in range(1, N + 1):
        term = (term @ B) / k
        Phi += term

    for _ in range(scaling_steps):
        Phi = Phi @ Phi

    return Phi


def propagate_linear(
    Phi: NDArray[np.float64],
    R0: ArrayLike,
    t: ArrayLike,
) -> NDArray[np.float64]:
    """
    Propagate linear system states using the precomputed transition matrix Phi.

    Parameters
    ----------
    Phi : NDArray[np.float64] of shape (2n, 2n)
        Precomputed discrete state transition matrix.
    R0 : Sequence[float] or NDArray of shape (2n,)
        Initial state vector [theta_0, dot{theta}_0].
    t : Sequence[float] or NDArray of shape (n_steps,)
        Monotonically increasing time grid with uniform step dt.

    Returns
    -------
    R : NDArray[np.float64] of shape (n_steps, 2n)
        Simulated trajectory states over time.

    Raises
    ------
    ValueError
        If Phi is not square, R0 dimension is incompatible, or t is empty.
    """
    Phi_arr = np.asarray(Phi, dtype=np.float64)
    R0_arr = np.asarray(R0, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)

    if Phi_arr.ndim != 2 or Phi_arr.shape[0] != Phi_arr.shape[1]:
        raise ValueError("Phi must be a square matrix")
    if R0_arr.ndim != 1 or Phi_arr.shape[0] != R0_arr.size:
        raise ValueError("R0 must be a vector compatible with Phi")
    if t_arr.size == 0:
        raise ValueError("t must contain at least one sample")

    n_steps = t_arr.size
    R = np.zeros((n_steps, R0_arr.size), dtype=np.float64)
    R[0] = R0_arr

    for i in range(1, n_steps):
        R[i] = Phi_arr @ R[i - 1]

    return R
