import numpy as np

from qldpc_bposd_jax.codes import bb72_check_matrices
from qldpc_bposd_jax.noise import sample_error_and_syndrome, sample_pauli_error

# We chose 137 for our random seed because of its relation to the (inverse) fine-structure constant, which seems
# fitting for a quantum computing project.


def test_zero_noise():
    rng = np.random.default_rng(137)
    # no noise means no qubit experiences an error
    e_X, e_Z = sample_pauli_error(n=72, p=0.0, rng=rng)

    # so the error vectors should both be zero
    assert np.array_equal(e_X, np.zeros(72, dtype=np.uint8))
    assert np.array_equal(e_Z, np.zeros(72, dtype=np.uint8))


def test_full_noise():
    rng = np.random.default_rng(137)

    # when p=1, every qubit experiences a Pauli error with the specific Pauli determined at random
    e_X, e_Z = sample_pauli_error(n=72, p=1.0, rng=rng)

    # e_X == 1 (similarly for e_Z == 1) returns a 72-entry Boolean array
    # whose i-th entry is True exactly when e_X[i] == 1.
    # Since e_X and e_Z are binary arrays, e_X | e_Z computes the
    # elementwise bitwise OR. The assertion passes if, for every qubit i,
    # at least one of e_X[i] and e_Z[i] equals 1.
    # Equivalently, no qubit has the binary representation (0, 0).
    assert np.all((e_X | e_Z) == 1)


def test_intermediate_noise():
    rng = np.random.default_rng(137)

    # we use 10,000 independent qubit samples (rather than 72) to reduce
    # statistical fluctuations in the observed error frequencies.
    # For p = 0.3, the standard deviation of the observed error frequency
    # is sqrt(p * (1 - p) / n), approximately 0.0046 for n = 10,000.
    # A tolerance of 0.02 (two percentage points) is therefore generous
    # enough to avoid failures from ordinary random fluctuations while
    # still detecting substantial errors in the sampled distribution.

    n = 10_000
    p = 0.3
    tol = 0.02

    e_X, e_Z = sample_pauli_error(n=n, p=p, rng=rng)

    # calculate the fraction of qubits with a nonidentity error (observed error frequency)
    # obs_err_freq = (number of qubits with errors)/n
    obs_err_freq = np.mean(e_X | e_Z)

    # fraction of qubits experiencing an X error
    obs_x_freq = np.mean((e_X == 1) & (e_Z == 0))

    # Fraction of qubits experiencing a Y error
    obs_y_freq = np.mean((e_X == 1) & (e_Z == 1))

    # Fraction of qubits experiencing a Z error
    obs_z_freq = np.mean((e_X == 0) & (e_Z == 1))

    # check that the observed fraction is close to p
    assert abs(obs_err_freq - p) < tol

    # check that the observed frequencies of the different Paulis are close to p/3
    assert abs(obs_x_freq - p / 3) < tol
    assert abs(obs_y_freq - p / 3) < tol
    assert abs(obs_z_freq - p / 3) < tol


def test_sample_error_and_syndrome():
    rng = np.random.default_rng(137)
    H_X, H_Z = bb72_check_matrices()

    e_X, e_Z, s_X, s_Z = sample_error_and_syndrome(H_X, H_Z, p=0.03, rng=rng)

    # verify the shapes of all four returned vectors
    assert e_X.shape == (72,)
    assert e_Z.shape == (72,)
    assert s_X.shape == (36,)
    assert s_Z.shape == (36,)

    # verify both syndrome equations over F_2
    assert np.array_equal(s_X, (H_X @ e_Z) % 2)
    assert np.array_equal(s_Z, (H_Z @ e_X) % 2)
