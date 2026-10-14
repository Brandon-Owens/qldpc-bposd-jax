import galois
import numpy as np

from qldpc_bposd_jax.codes import bb72_check_matrices


def test_bb72_shapes():
    H_X, H_Z = bb72_check_matrices()

    assert H_X.shape == (36, 72)
    assert H_Z.shape == (36, 72)


def test_bb72_css_commutation():
    H_X, H_Z = bb72_check_matrices()

    # verify H_X @ H_Z.T = 0 over F_2
    assert np.array_equal((H_X @ H_Z.T) % 2, np.zeros((36, 36), dtype=np.uint8))


def test_bb72_binary_entries():
    H_X, H_Z = bb72_check_matrices()

    assert np.all(np.isin(H_X, [0, 1]))
    assert np.all(np.isin(H_Z, [0, 1]))


def test_bb72_weights():
    H_X, H_Z = bb72_check_matrices()

    # verify Hamming weight of every column of H_X and H_Z is equal to 3
    assert np.all(np.sum(H_X, axis=0) == 3)
    assert np.all(np.sum(H_Z, axis=0) == 3)

    # verify the Hamming weight of every row of H_X and H_Z is equal to 6
    assert np.all(np.sum(H_X, axis=1) == 6)
    assert np.all(np.sum(H_Z, axis=1) == 6)


def test_bb72_ranks():
    H_X, H_Z = bb72_check_matrices()

    # load in F_2
    GF2 = galois.GF(2)

    # GF2(...) interprets the check matrix as a matrix over F_2
    # then .rows_pace() returns a matrix whose rows form a basis for the row space
    # .shape[0] counts those basis vectors, giving the rank.
    rank_X = GF2(H_X).row_space().shape[0]
    rank_Z = GF2(H_Z).row_space().shape[0]

    # verify that each matrix has rank 30
    assert rank_X == 30
    assert rank_Z == 30

    # verify the equation for logical qubit count
    k = H_X.shape[1] - rank_X - rank_Z
    assert k == 12
