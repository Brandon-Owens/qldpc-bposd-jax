import numpy as np

from qldpc_bposd_jax.codes import bb72_check_matrices
from qldpc_bposd_jax.syndrome import binary_syndrome


def test_single_x_error_syndrome():
    # dont need H_X, only need H_Z
    _, H_Z = bb72_check_matrices()

    # generate X-error on qubit 0
    e_X = np.zeros(72, dtype=np.uint8)
    e_X[0] = 1

    # verify that binary_syndrome(H_Z,e_X) is the same
    # as the first column of H_Z
    assert np.array_equal(binary_syndrome(H_Z, e_X), H_Z[:, 0])


def test_bb72_single_x_syndromes_distinct():
    _, H_Z = bb72_check_matrices()

    # verify that the 72 columns of H_Z are distinct.
    # calling np.unique(H_Z, axis=1) returns a matrix consisting of the
    # unique columns (indicated by axis=1 arg) of H_Z
    assert np.unique(H_Z, axis=1).shape[1] == H_Z.shape[1]


def test_single_z_error_syndromes():
    H_X, _ = bb72_check_matrices()

    for i in range(72):
        # construct the Z error on qubit i
        e_Z = np.zeros(72, dtype=np.uint8)
        e_Z[i] = 1

        # verify that its syndrome equals column i of H_X
        assert np.array_equal(binary_syndrome(H_X, e_Z), H_X[:, i])


def test_bb72_single_z_syndromes_distinct():
    H_X, _ = bb72_check_matrices()

    # verify that the 72 columns of H_X are distinct.
    # calling np.unique(H_X, axis=1) returns a matrix consisting of the
    # unique columns (indicated by axis=1 arg) of H_X
    assert np.unique(H_X, axis=1).shape[1] == H_X.shape[1]


def test_single_y_error_syndromes():
    H_X, H_Z = bb72_check_matrices()

    for i in range(72):
        # construct the X and Z components of Y_i
        e_X = np.zeros(72, dtype=np.uint8)
        e_X[i] = 1
        e_Z = np.zeros(72, dtype=np.uint8)
        e_Z[i] = 1

        # verify the X-check syndrome
        assert np.array_equal(binary_syndrome(H_X, e_Z), H_X[:, i])

        # verify the Z-check syndrome
        assert np.array_equal(binary_syndrome(H_Z, e_X), H_Z[:, i])
