import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src import chain as C
from src import nonlinear as NL
from src.structure import dynamic_matrix_dense, suffix_masses

M3 = np.array([2.0, 1.5, 0.8])
L3 = np.array([1.2, 0.9, 0.6])


@pytest.mark.parametrize("n", [1, 2, 3, 8, 20])
def test_acceleration_matches_dense_mass_matrix(n):
    rng = np.random.default_rng(n)
    m, l = rng.uniform(0.2, 3, n), rng.uniform(0.2, 2, n)
    th, om = rng.uniform(-np.pi, np.pi, n), rng.normal(0, 2, n)
    ref = NL.state_derivative(0.0, np.r_[th, om], m, l)[n:]
    np.testing.assert_allclose(C.angular_acceleration(th, om, m, l), ref, rtol=1e-10, atol=1e-10)


def test_tensions_at_rest_equal_suspended_weight():
    T = C.tensions(np.zeros(3), np.zeros(3), M3, L3)
    np.testing.assert_allclose(T, 9.81 * suffix_masses(M3))


def test_single_pendulum():
    a = C.angular_acceleration([0.7], [1.3], [2.0], [1.5])
    assert a[0] == pytest.approx(-9.81 / 1.5 * np.sin(0.7))


def test_energy_matches_reference():
    rng = np.random.default_rng(1)
    th, om = rng.uniform(-2, 2, 6), rng.normal(0, 1, 6)
    m, l = rng.uniform(0.5, 2, 6), rng.uniform(0.5, 2, 6)
    assert C.energy(th, om, m, l) == pytest.approx(NL.total_energy(np.r_[th, om], m, l), rel=1e-12)


def test_linearization_recovers_tridiagonal_Q():
    rng = np.random.default_rng(2)
    m, l = rng.uniform(0.5, 2, 7), rng.uniform(0.5, 2, 7)
    h = 1e-6
    Jac = np.column_stack([
        (C.angular_acceleration(h * e, np.zeros(7), m, l) - C.angular_acceleration(-h * e, np.zeros(7), m, l)) / (2 * h)
        for e in np.eye(7)
    ])
    np.testing.assert_allclose(Jac, -dynamic_matrix_dense(m, l), rtol=1e-6, atol=1e-6)


def test_rattle_second_order_and_constraints():
    th0, om0 = np.full(3, np.pi / 2), np.zeros(3)
    ref = solve_ivp(C.state_derivative, (0, 1.0), np.r_[th0, om0], args=(M3, L3), method="DOP853",
                    rtol=1e-13, atol=1e-13).y[:3, -1]
    errs = []
    for dt in (2e-3, 1e-3):
        N = int(round(1.0 / dt))
        _, TH, OM, _ = C.rattle_propagate(th0, om0, M3, L3, dt, N, save_every=N)
        errs.append(np.abs(TH[-1] - ref).max())
        q, v = C.to_cartesian(TH[-1], OM[-1], L3)
        d = np.diff(np.vstack(([0, 0], q)), axis=0)
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), L3, rtol=1e-12)
    assert 3.5 < errs[0] / errs[1] < 4.5


def test_rattle_energy_bounded():
    th0, om0 = np.array([1.0, -0.5, 2.0]), np.zeros(3)
    E0 = C.energy(th0, om0, M3, L3)
    _, TH, OM, _ = C.rattle_propagate(th0, om0, M3, L3, 2e-3, 20000, save_every=20)
    E = np.abs([C.energy(a, b, M3, L3) - E0 for a, b in zip(TH, OM)]) / (M3.sum() * 9.81 * L3.sum())
    q = len(E) // 10
    assert E.max() < 1e-3
    assert E[-q:].max() < 3 * E[:q].max()


def test_rattle_time_reversible():
    th0, om0 = np.array([0.8, 0.3, -0.4]), np.array([0.1, 0.0, -0.2])
    _, TH, OM, _ = C.rattle_propagate(th0, om0, M3, L3, 1e-3, 2000, save_every=2000)
    _, TH2, OM2, _ = C.rattle_propagate(TH[-1], -OM[-1], M3, L3, 1e-3, 2000, save_every=2000)
    np.testing.assert_allclose(TH2[-1], th0, atol=1e-9)
    np.testing.assert_allclose(-OM2[-1], om0, atol=1e-9)
