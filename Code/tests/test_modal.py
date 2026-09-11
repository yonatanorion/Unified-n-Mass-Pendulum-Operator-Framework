import pytest
import numpy as np
from scipy.linalg import expm
from src.assembly import assemble_system
from src.modal import solve_modal
from src.propagator import propagate_linear


def test_modal_vs_expm():
    masses = [1.0, 1.0, 1.0]
    lengths = [1.0, 1.0, 1.0]
    A, M, K = assemble_system(masses, lengths)

    dt = 0.01
    t = np.arange(0, 5.0 + dt, dt)
    R0 = np.array([0.1, 0.05, 0.02, 0.0, 0.0, 0.0])

    sol_modal = solve_modal(M, K, R0, t)

    Phi_exact = expm(A * dt)
    sol_expm = propagate_linear(Phi_exact, R0, t)

    np.testing.assert_allclose(sol_modal, sol_expm, rtol=1e-12, atol=1e-12)


def test_modal_invalid_inputs():
    with pytest.raises(ValueError, match="square matrices of the same shape"):
        solve_modal(np.zeros((2, 2)), np.zeros((3, 3)), np.zeros(4), np.zeros(2))
