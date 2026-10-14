import numpy as np


def cyclic_shift(n: int) -> np.ndarray:
    """Return the n x n binary cyclic shift matrix"""
    array = []

    for i in range(n):
        target = (i + 1) % n
        row = np.zeros(n, dtype=np.uint8)
        row[target] = 1
        array.append(row)

    return np.array(array, dtype=np.uint8)


def bb72_check_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Construct CSS check matrices for the [[72,12,6]] BB code.

    Uses the construction of Bravyi et al. (2024), with
    ell = m = 6 and polynomials

        A = x^3 + y + y^2
        B = y^3 + x + x^2

    Returns:
        H_X, H_Z: Binary NumPy arrays of shape (36, 72).

    Reference:
        Bravyi et al., Nature 627, 778–782 (2024),
        Extended Data Table 1.
    """
    # construct the shift matrices x and y
    x = np.kron(cyclic_shift(6), np.eye(6, dtype=np.uint8))
    y = np.kron(np.eye(6, dtype=np.uint8), cyclic_shift(6))

    # construct A and B over F_2
    A = (np.linalg.matrix_power(x, 3) + y + np.linalg.matrix_power(y, 2)) % 2
    B = (np.linalg.matrix_power(y, 3) + x + np.linalg.matrix_power(x, 2)) % 2

    # construct H_X and H_Z
    H_X = np.hstack((A, B))
    H_Z = np.hstack((B.T, A.T))

    return H_X, H_Z
