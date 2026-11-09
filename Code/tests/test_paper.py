"""Checks of statements made in the paper that are not covered by the module tests."""
import os
import re

import numpy as np
import pytest
from scipy.special import jn_zeros

from src import chain as C
from src import minimal
from src.assembly import assemble_system
from src.structure import dynamic_matrix_tridiagonal, modal_matrix, natural_frequencies

PAPER = os.path.join(os.path.dirname(__file__), "..", "..", "Paper", "Main.tex")


def _sign_changes(v):
    v = v[np.abs(v) > 1e-9 * np.abs(v).max()]
    return int(np.sum(np.diff(np.sign(v)) != 0))


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_mode_k_has_k_minus_1_nodes_in_angles_and_displacements(n):
    rng = np.random.default_rng(10 + n)
    m, l = rng.uniform(0.5, 2.0, n), rng.uniform(0.5, 2.0, n)
    _, U = modal_matrix(m, l)
    for k in range(n):
        assert _sign_changes(U[:, k]) == k                 # angle mode shape
        assert _sign_changes(np.cumsum(l * U[:, k])) == k  # horizontal displacements x = E L theta


def test_hanging_chain_limit_one_quarter_over_n():
    n, L = 1024, 1.0
    omega = natural_frequencies(np.full(n, 1.0 / n), np.full(n, L / n), select="i", select_range=(0, 2))
    exact = jn_zeros(0, 3) / 2 * np.sqrt(9.81 / L)
    rel = (omega - exact) / exact
    assert np.all(rel < 0)                                 # discrete frequencies are lower
    np.testing.assert_allclose(-n * rel, 0.25, rtol=0.01)


def test_zero_angle_tension_matrix_is_rescaled_inverse_mass_matrix():
    rng = np.random.default_rng(4)
    m, l = rng.uniform(0.5, 2.0, 6), rng.uniform(0.5, 2.0, 6)
    _, M, _ = assemble_system(m, l)
    d, off, _ = C.tension_system(np.zeros(6), np.zeros(6), m, l)
    B0 = np.diag(d) + np.diag(off, 1) + np.diag(off, -1)
    np.testing.assert_allclose(B0, np.diag(l) @ np.linalg.inv(M) @ np.diag(l), atol=1e-12)


def test_double_pendulum_textbook_frequencies():
    d, sub, sup = minimal.dynamical_matrix(np.ones(2), np.ones(2))
    Q = np.diag(d) + np.diag(sub, -1) + np.diag(sup, 1)
    np.testing.assert_allclose(Q / 9.81, [[2, -1], [-2, 2]])
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(Q).real) / 9.81, [2 - np.sqrt(2), 2 + np.sqrt(2)])


@pytest.mark.parametrize("n", [1, 2, 3, 10, 40])
def test_appendix_code_matches_modules(n):
    rng = np.random.default_rng(n)
    m, l = rng.uniform(0.3, 2.0, n), rng.uniform(0.3, 2.0, n)
    th, om = rng.uniform(-3, 3, n), rng.normal(0, 2, n)
    np.testing.assert_allclose(minimal.accelerations(th, om, m, l), C.angular_acceleration(th, om, m, l),
                               rtol=1e-11, atol=1e-11)
    for a, b in zip(minimal.dynamical_matrix(m, l), dynamic_matrix_tridiagonal(m, l)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not os.path.exists(PAPER), reason="paper source not available")
def test_appendix_listing_is_verbatim():
    listing = re.search(r"\\begin\{lstlisting\}\n(.*?)\\end\{lstlisting\}", open(PAPER).read(), re.S).group(1)
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "minimal.py")).read()
    assert listing in src
