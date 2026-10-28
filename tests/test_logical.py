import numpy as np
import pytest

from qldpc_bposd_jax.logical import is_in_row_space, logical_success

H_X = np.array([[1, 1, 0, 0]], dtype=np.uint8)
H_Z = np.array([[0, 0, 1, 1]], dtype=np.uint8)


def test_row_space_membership():
    zero = np.zeros(4, dtype=np.uint8)
    stabilizer = np.array([1, 1, 0, 0], dtype=np.uint8)
    logical_X = np.array([1, 0, 0, 0], dtype=np.uint8)

    assert is_in_row_space(H_X, zero)
    assert is_in_row_space(H_X, stabilizer)
    assert not is_in_row_space(H_X, logical_X)

    # the logical X operator has zero Z-check syndrome
    assert np.array_equal((H_Z @ logical_X) % 2, np.zeros(1))


def test_logical_success_with_x_stabilizer_residual():
    e_X = np.array([1, 0, 0, 0], dtype=np.uint8)
    estimate_X = np.array([0, 1, 0, 0], dtype=np.uint8)

    zero = np.zeros(4, dtype=np.uint8)

    assert not np.array_equal(e_X, estimate_X)
    assert logical_success(H_X, H_Z, e_X, zero, estimate_X, zero)


def test_logical_success_with_z_stabilizer_residual():
    e_Z = np.array([0, 0, 1, 0], dtype=np.uint8)
    estimate_Z = np.array([0, 0, 0, 1], dtype=np.uint8)
    zero = np.zeros(4, dtype=np.uint8)

    assert not np.array_equal(e_Z, estimate_Z)
    assert logical_success(H_X, H_Z, zero, e_Z, zero, estimate_Z)


def test_logical_failure_with_zero_syndrome():
    zero = np.zeros(4, dtype=np.uint8)
    logical_X = np.array([1, 0, 0, 0], dtype=np.uint8)

    assert np.array_equal((H_Z @ logical_X) % 2, np.zeros(1))
    assert not logical_success(H_X, H_Z, zero, zero, logical_X, zero)


def test_logical_success_rejects_nonbinary_error():
    zero = np.zeros(4, dtype=np.uint8)
    invalid_estimate = np.array([2, 0, 0, 0], dtype=np.uint8)

    with pytest.raises(ValueError, match="must be binary"):
        logical_success(H_X, H_Z, zero, zero, invalid_estimate, zero)
