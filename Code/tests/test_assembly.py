import pytest
import numpy as np
from src.assembly import assemble_system


def test_assemble_system_n2():
    masses = [1.0, 1.0]
    lengths = [1.0, 1.0]
    A, M, K = assemble_system(masses, lengths, g=9.81)

    assert M.shape == (2, 2)
    assert K.shape == (2, 2)
    assert A.shape == (4, 4)

    # Symmetry of M
    np.testing.assert_allclose(M, M.T)

    # Positive definiteness of M and K
    assert np.all(np.linalg.eigvals(M) > 0)
    assert np.all(np.linalg.eigvals(K) > 0)

    # Check theoretical values
    # M_11 = (m1+m2)*l1^2 = 2.0, M_12 = m2*l1*l2 = 1.0
    # M_21 = 1.0, M_22 = m2*l2^2 = 1.0
    expected_M = np.array([[2.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(M, expected_M)

    # K_11 = (m1+m2)*g*l1 = 2*9.81 = 19.62, K_22 = m2*g*l2 = 9.81
    expected_K = np.array([[19.62, 0.0], [0.0, 9.81]])
    np.testing.assert_allclose(K, expected_K)


def test_assemble_system_invalid_inputs():
    with pytest.raises(ValueError, match="one-dimensional"):
        assemble_system([[1.0]], [1.0])

    with pytest.raises(ValueError, match="same non-zero size"):
        assemble_system([1.0], [1.0, 2.0])

    with pytest.raises(ValueError, match="strictly positive"):
        assemble_system([-1.0, 1.0], [1.0, 1.0])

    with pytest.raises(ValueError, match="strictly positive"):
        assemble_system([1.0, 1.0], [1.0, 0.0])

    with pytest.raises(ValueError, match="finite positive number"):
        assemble_system([1.0], [1.0], g=-9.81)
