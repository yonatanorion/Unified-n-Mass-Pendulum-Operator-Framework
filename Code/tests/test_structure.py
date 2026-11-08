import numpy as np
import pytest
from scipy.linalg import eigh

from src.assembly import assemble_system
from src.structure import (
    apply_dynamic_matrix,
    dynamic_matrix_dense,
    inverse_mass_tridiagonal,
    jacobi_matrix,
    modal_matrix,
    natural_frequencies,
)


def _random_chain(rng, n):
    return rng.uniform(0.2, 3.0, n), rng.uniform(0.3, 2.0, n)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 15])
def test_inverse_mass_closed_form(n):
    rng = np.random.default_rng(n)
    m, l = _random_chain(rng, n)
    _, M, _ = assemble_system(m, l)
    d, off = inverse_mass_tridiagonal(m, l)
    Minv = np.diag(d) + np.diag(off, 1) + np.diag(off, -1)
    np.testing.assert_allclose(Minv @ M, np.eye(n), atol=1e-11)


def test_inverse_mass_double_pendulum_by_hand():
    # M = [[2, 1], [1, 1]] for unit masses and lengths, so M^{-1} = [[1, -1], [-1, 2]].
    d, off = inverse_mass_tridiagonal([1.0, 1.0], [1.0, 1.0])
    np.testing.assert_allclose(d, [1.0, 2.0])
    np.testing.assert_allclose(off, [-1.0])


@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_dynamic_matrix_matches_solve(n):
    rng = np.random.default_rng(100 + n)
    m, l = _random_chain(rng, n)
    _, M, K = assemble_system(m, l)
    Q = dynamic_matrix_dense(m, l)
    np.testing.assert_allclose(Q, np.linalg.solve(M, K), rtol=1e-10, atol=1e-10)
    theta = rng.standard_normal(n)
    np.testing.assert_allclose(apply_dynamic_matrix(m, l, theta), Q @ theta, rtol=1e-12, atol=1e-12)


def test_jacobi_offdiagonal_negative_and_frequencies():
    rng = np.random.default_rng(7)
    m, l = _random_chain(rng, 20)
    _, M, K = assemble_system(m, l)
    _, e = jacobi_matrix(m, l)
    assert np.all(e < 0.0)
    omega = natural_frequencies(m, l)
    omega_ref = np.sqrt(eigh(K, M, eigvals_only=True))
    np.testing.assert_allclose(omega, omega_ref, rtol=1e-10)
    assert np.all(np.diff(omega) > 0.0)  # simple spectrum


def test_modal_matrix_is_m_orthonormal():
    rng = np.random.default_rng(3)
    m, l = _random_chain(rng, 8)
    _, M, K = assemble_system(m, l)
    omega, U = modal_matrix(m, l)
    np.testing.assert_allclose(U.T @ M @ U, np.eye(8), atol=1e-10)
    np.testing.assert_allclose(U.T @ K @ U, np.diag(omega**2), atol=1e-8 * omega.max() ** 2)


def test_lowest_mode_selection():
    m = np.ones(200)
    l = np.full(200, 0.01)
    low = natural_frequencies(m, l, select="i", select_range=(0, 4))
    np.testing.assert_allclose(low, natural_frequencies(m, l)[:5], rtol=1e-10)
