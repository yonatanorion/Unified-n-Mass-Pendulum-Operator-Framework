import pytest
import numpy as np
from scipy.integrate import solve_ivp
from src.nonlinear import nonlinear_terms, state_derivative, total_energy


def test_nonlinear_terms_zero_angles():
    masses = [1.0, 1.0]
    lengths = [1.0, 1.0]
    theta = [0.0, 0.0]
    omega = [0.0, 0.0]

    M, c, gravity = nonlinear_terms(theta, omega, masses, lengths)

    # At theta = 0, omega = 0: c should be 0, gravity should be 0
    np.testing.assert_allclose(c, np.zeros(2))
    np.testing.assert_allclose(gravity, np.zeros(2))
    np.testing.assert_allclose(M, np.array([[2.0, 1.0], [1.0, 1.0]]))


def test_nonlinear_energy_conservation():
    masses = [1.0, 1.0]
    lengths = [1.0, 1.0]
    g = 9.81
    state0 = np.array([np.radians(5.0), 0.0, 0.0, 0.0])

    t_span = (0.0, 4.0)
    t_eval = np.linspace(0.0, 4.0, 200)

    sol = solve_ivp(
        state_derivative,
        t_span,
        state0,
        args=(masses, lengths, g),
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-14,
    )

    E0 = total_energy(state0, masses, lengths, g)
    energies = [total_energy(state, masses, lengths, g) for state in sol.y.T]
    max_drift = np.max(np.abs(np.array(energies) - E0))

    assert max_drift < 1e-12
