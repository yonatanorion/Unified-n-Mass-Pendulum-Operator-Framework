import pytest
import numpy as np
from scipy.linalg import expm
from src.assembly import assemble_system
from src.propagator import peano_baker_step, propagate_linear


def test_peano_baker_step_accuracy():
    masses = [2.0, 1.5, 0.8]
    lengths = [1.2, 0.9, 0.6]
    A, _, _ = assemble_system(masses, lengths)

    dt = 0.01
    Phi_approx = peano_baker_step(A, dt, N=20)
    Phi_exact = expm(A * dt)

    np.testing.assert_allclose(Phi_approx, Phi_exact, rtol=1e-14, atol=1e-14)


def test_propagate_linear_trajectory():
    masses = [1.0, 1.0]
    lengths = [1.0, 1.0]
    A, _, _ = assemble_system(masses, lengths)
    dt = 0.01
    t = np.arange(0, 1.0 + dt, dt)
    Phi = peano_baker_step(A, dt, N=20)
    R0 = np.array([0.1, 0.05, 0.0, 0.0])

    trajectory = propagate_linear(Phi, R0, t)

    assert trajectory.shape == (len(t), 4)
    np.testing.assert_allclose(trajectory[0], R0)


def test_propagator_invalid_inputs():
    with pytest.raises(ValueError, match="square matrix"):
        peano_baker_step(np.zeros((2, 3)), 0.01)

    with pytest.raises(ValueError, match="non-negative"):
        peano_baker_step(np.eye(2), 0.01, N=-1)
