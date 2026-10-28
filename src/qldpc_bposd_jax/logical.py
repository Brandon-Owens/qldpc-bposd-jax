import galois
import numpy as np
from numpy.typing import ArrayLike

GF2 = galois.GF(2)


def is_in_row_space(H: ArrayLike, vector: ArrayLike) -> bool:
    """Check whether a binary vector belongs to the row space of H over GF(2)."""
    H = np.asarray(H)
    vector = np.asarray(vector)

    if H.ndim != 2 or vector.shape != (H.shape[1],):
        raise ValueError("Incompatible matrix and vector shapes.")

    if not np.all((H == 0) | (H == 1)):
        raise ValueError("Matrix entries must be binary.")

    if not np.all((vector == 0) | (vector == 1)):
        raise ValueError("Vector entries must be binary.")

    H = GF2(H.astype(np.uint8))
    vector = GF2(vector.astype(np.uint8))

    rank_H = np.linalg.matrix_rank(H)
    augmented = np.vstack([H, vector])

    return bool(np.linalg.matrix_rank(augmented) == rank_H)


def logical_success(
    H_X: ArrayLike,
    H_Z: ArrayLike,
    e_X: ArrayLike,
    e_Z: ArrayLike,
    estimate_X: ArrayLike,
    estimate_Z: ArrayLike,
) -> bool:
    """Check whether the residual Pauli error is a CSS stabilizer."""
    e_X = np.asarray(e_X)
    e_Z = np.asarray(e_Z)
    estimate_X = np.asarray(estimate_X)
    estimate_Z = np.asarray(estimate_Z)

    H_X = np.asarray(H_X)
    H_Z = np.asarray(H_Z)

    if H_X.ndim != 2 or H_Z.ndim != 2 or H_X.shape[1] != H_Z.shape[1]:
        raise ValueError("Incompatible CSS parity-check matrix shapes.")

    n = H_X.shape[1]
    vectors = (e_X, e_Z, estimate_X, estimate_Z)

    if any(v.shape != (n,) for v in vectors):
        raise ValueError("Incompatible error-vector shapes.")

    if any(not np.all((v == 0) | (v == 1)) for v in vectors):
        raise ValueError("Error-vector entries must be binary.")

    r_X = (e_X + estimate_X) % 2
    r_Z = (e_Z + estimate_Z) % 2

    return is_in_row_space(H_X, r_X) and is_in_row_space(H_Z, r_Z)
