import numpy as np

from qldpc_bposd_jax.syndrome import binary_syndrome


def sample_pauli_error(
    n: int,
    p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an n-qubit Pauli error under depolarizing noise."""

    # sample one integer from {0,1,2,3} (corresponding to I,X,Y,Z, resp.) for each of the n-qubits,
    # using probabilities [1-p, p/3, p/3, p/3]
    samples = rng.choice(4, size=n, p=[1 - p, p / 3, p / 3, p / 3])

    # convert the n-entry array into two n-entry binary arrays based on the following rule:
    # an entry of e_X equals 1 if it lies in {1,2} (X or Y) and equals 0 otherwise, and
    # an entry of e_Z equals 1 if it lies in {2,3} (Y or Z) and equals 0 otherwise.
    e_X = np.isin(samples, [1, 2]).astype(dtype=np.uint8)
    e_Z = np.isin(samples, [2, 3]).astype(dtype=np.uint8)

    return e_X, e_Z


def sample_error_and_syndrome(
    H_X: np.ndarray,
    H_Z: np.ndarray,
    p: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a Pauli error and compute its CSS syndrome."""

    # get the number of physical qubits for our input CSS code
    n = H_X.shape[1]

    # generates the binary representation of a randomly sampled
    # n-qubit Pauli error under our depolarizing noise model
    e_X, e_Z = sample_pauli_error(n, p, rng)

    # compute both syndrome components
    s_X = binary_syndrome(H_X, e_Z)
    s_Z = binary_syndrome(H_Z, e_X)

    return e_X, e_Z, s_X, s_Z
