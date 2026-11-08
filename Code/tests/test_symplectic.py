import numpy as np
import pytest

from src.assembly import assemble_system
from src.modal import solve_modal
from src.symplectic import linear_energy, max_frequency, verlet_propagate

M3 = np.array([2.0, 1.5, 0.8])
L3 = np.array([1.2, 0.9, 0.6])
TH0 = np.array([0.1, 0.05, 0.02])


def test_linear_energy_matches_dense_quadratic_form():
    rng = np.random.default_rng(5)
    m, l = rng.uniform(0.2, 3.0, 12), rng.uniform(0.3, 2.0, 12)
    _, M, K = assemble_system(m, l)
    th, om = rng.standard_normal(12), rng.standard_normal(12)
    ref = 0.5 * om @ M @ om + 0.5 * th @ K @ th
    assert linear_energy(m, l, th, om) == pytest.approx(ref, rel=1e-12)


def test_max_frequency_matches_dense():
    _, M, K = assemble_system(M3, L3)
    from scipy.linalg import eigh
    assert max_frequency(M3, L3) == pytest.approx(np.sqrt(eigh(K, M, eigvals_only=True)[-1]), rel=1e-12)


def test_second_order_convergence():
    _, M, K = assemble_system(M3, L3)
    ref = solve_modal(M, K, np.r_[TH0, np.zeros(3)], [0.0, 4.0])[-1, :3]
    errs = []
    for dt in (0.01, 0.005):
        N = int(round(4.0 / dt))
        _, TH, _ = verlet_propagate(M3, L3, TH0, np.zeros(3), dt, N, save_every=N)
        errs.append(np.abs(TH[-1] - ref).max())
    assert 3.5 < errs[0] / errs[1] < 4.5


def test_energy_error_bounded_without_drift():
    E0 = linear_energy(M3, L3, TH0, np.zeros(3))
    _, TH, OM = verlet_propagate(M3, L3, TH0, np.zeros(3), 0.01, 200000, save_every=100)
    E = np.array([abs(linear_energy(M3, L3, a, b) - E0) / E0 for a, b in zip(TH, OM)])
    q = len(E) // 10
    assert E.max() < 5e-4
    assert E[-q:].max() < 1.5 * E[:q].max()  # no secular growth


def test_time_reversibility():
    _, TH, OM = verlet_propagate(M3, L3, TH0, np.zeros(3), 0.01, 5000, save_every=5000)
    _, TH2, OM2 = verlet_propagate(M3, L3, TH[-1], -OM[-1], 0.01, 5000, save_every=5000)
    np.testing.assert_allclose(TH2[-1], TH0, atol=1e-11)


def test_unstable_step_rejected():
    dt = 2.5 / max_frequency(M3, L3)
    with pytest.raises(ValueError, match="unstable step"):
        verlet_propagate(M3, L3, TH0, np.zeros(3), dt, 10)
