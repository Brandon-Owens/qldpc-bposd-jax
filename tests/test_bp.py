"""Tests for binary sum-product belief propagation and syndrome decoding."""

from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qldpc_bposd_jax.bp import (
    binary_prior_llr,
    bp_iteration,
    check_to_data_update,
    data_to_check_update,
    decode_binary,
    hard_decision,
    posterior_llrs,
    syndrome_matches,
)

# Small, cycle-free Tanner graph used throughout the BP tests.
H_demo = np.array(
    [[1, 1, 0, 0], [0, 1, 1, 1]],
    dtype=np.uint8,
)
s_demo = jnp.array([1, 0], dtype=jnp.uint8)
check_idx, data_idx = (jnp.asarray(indices) for indices in np.nonzero(H_demo))


# --- Prior LLRs and message updates ---


def test_binary_prior_llr():
    p = 0.06
    q = 2 * p / 3
    expected = np.log((1 - q) / q)

    assert np.isclose(binary_prior_llr(p), expected, rtol=1e-6, atol=1e-6)


def test_check_to_data_update():
    incoming = jnp.array([0.4, -0.7, 1.2, -0.2, 0.9])
    result = check_to_data_update(incoming, check_idx, s_demo)

    # Degree-two check with syndrome bit 1.
    assert jnp.allclose(result[:2], jnp.array([0.7, -0.4]))

    # Degree-three check: the receiving edge must be excluded.
    expected = 2 * jnp.arctanh(jnp.tanh(incoming[3] / 2) * jnp.tanh(incoming[4] / 2))
    assert jnp.allclose(result[2], expected)


def test_check_to_data_syndrome_sign():
    incoming = jnp.array([0.4, -0.7, 1.2, -0.2, 0.9])
    flipped_syndrome = jnp.array([1, 1], dtype=jnp.uint8)

    result = check_to_data_update(incoming, check_idx, s_demo)
    flipped = check_to_data_update(incoming, check_idx, flipped_syndrome)

    assert jnp.allclose(flipped[:2], result[:2])
    assert jnp.allclose(flipped[2:], -result[2:])


def test_check_to_data_clipping():
    large_messages = jnp.full(check_idx.shape, 100.0)
    result = check_to_data_update(large_messages, check_idx, s_demo)

    assert jnp.all(jnp.isfinite(result))


def test_check_to_data_jit():
    incoming = jnp.array([0.4, -0.7, 1.2, -0.2, 0.9])
    expected = check_to_data_update(incoming, check_idx, s_demo)
    actual = jax.jit(check_to_data_update)(incoming, check_idx, s_demo)

    assert jnp.allclose(actual, expected)


def test_data_to_check_update():
    prior_llrs = jnp.array([0.3, -0.2, 1.0, 0.4])
    check_to_data = jnp.array([0.6, 1.1, -0.7, 0.2, -0.9])

    result = data_to_check_update(prior_llrs, check_to_data, data_idx)

    # both edge exclusion at d_1 and degree-one data nodes are covered
    assert jnp.allclose(result, jnp.array([0.3, -0.9, 0.9, 1.0, 0.4]))


def test_posterior_llrs():
    prior_llrs = jnp.array([0.3, -0.2, 1.0, 0.4])
    check_to_data = jnp.array([0.6, 1.1, -0.7, 0.2, -0.9])

    result = posterior_llrs(prior_llrs, check_to_data, data_idx)

    # unlike data-to-check updates, posterior LLRs include every incoming message
    assert jnp.allclose(result, jnp.array([0.9, 0.2, 1.2, -0.5]))


# --- Hard decisions and syndrome consistency ---


def test_hard_decision():
    result = hard_decision(jnp.array([2.5, -1.3, 0.0, -0.7]))
    expected = jnp.array([0, 1, 0, 1], dtype=jnp.uint8)

    assert jnp.array_equal(result, expected)  # zero LLR breaks ties toward 0


def test_syndrome_matches():
    consistent_error = jnp.array([1, 0, 0, 0], dtype=jnp.uint8)
    inconsistent_error = jnp.array([0, 1, 0, 0], dtype=jnp.uint8)

    assert syndrome_matches(H_demo, consistent_error, s_demo)
    assert not syndrome_matches(H_demo, inconsistent_error, s_demo)

    # ordinary integer multiplication yields (2, 2), which is (0, 0) mod 2
    even_parity_error = jnp.array([1, 1, 1, 0], dtype=jnp.uint8)
    assert syndrome_matches(H_demo, even_parity_error, jnp.zeros(2, dtype=jnp.uint8))


# --- Full BP iteration and independent numerical reference ---


def test_bp_iteration():
    prior_llrs = jnp.array([0.3, -0.2, 1.0, 0.4])
    initial_data_to_check = prior_llrs[data_idx]

    outgoing_data, outgoing_check, posterior, estimate = bp_iteration(
        prior_llrs, initial_data_to_check, check_idx, data_idx, s_demo
    )

    expected_check = check_to_data_update(initial_data_to_check, check_idx, s_demo)
    expected_data = data_to_check_update(prior_llrs, expected_check, data_idx)
    expected_posterior = posterior_llrs(prior_llrs, expected_check, data_idx)
    expected_estimate = hard_decision(expected_posterior)

    assert jnp.allclose(outgoing_check, expected_check)
    assert jnp.allclose(outgoing_data, expected_data)
    assert jnp.allclose(posterior, expected_posterior)
    assert jnp.array_equal(estimate, expected_estimate)


def test_bp_posteriors_match_exact_marginals():
    """on this cycle-free graph, two BP rounds give exact posterior marginals"""
    p = 0.06
    q = 2 * p / 3
    n = H_demo.shape[1]

    errors = np.array(list(product((0, 1), repeat=n)), dtype=np.uint8)
    matches = np.all((errors @ H_demo.T) % 2 == np.asarray(s_demo), axis=1)
    consistent_errors = errors[matches]

    weights = consistent_errors.sum(axis=1)
    prior_probabilities = q**weights * (1 - q) ** (n - weights)
    conditional_probabilities = prior_probabilities / prior_probabilities.sum()
    exact_marginals = conditional_probabilities @ consistent_errors
    exact_llrs = np.log1p(-exact_marginals) - np.log(exact_marginals)

    prior_llrs = jnp.full(n, np.log((1 - q) / q))
    data_to_check = prior_llrs[data_idx]
    for _ in range(2):
        data_to_check, _, bp_llrs, _ = bp_iteration(
            prior_llrs, data_to_check, check_idx, data_idx, s_demo
        )

    # Float32 tanh/arctanh updates introduce small rounding errors.
    np.testing.assert_allclose(np.asarray(bp_llrs), exact_llrs, rtol=0, atol=2e-5)


# --- Complete decoder: stopping behavior ---


def test_decode_binary_early_stopping():
    syndrome = jnp.zeros(H_demo.shape[0], dtype=jnp.uint8)
    estimate, consistent, iterations = decode_binary(
        H_demo, syndrome, p=0.06, max_iter=10
    )

    assert jnp.array_equal(estimate, jnp.zeros(H_demo.shape[1], dtype=jnp.uint8))
    assert consistent
    assert iterations == 1


def test_decode_binary_noiseless():
    syndrome = jnp.zeros(H_demo.shape[0], dtype=jnp.uint8)
    estimate, consistent, iterations = decode_binary(H_demo, syndrome, p=0)

    assert jnp.array_equal(estimate, jnp.zeros(H_demo.shape[1], dtype=jnp.uint8))
    assert consistent
    assert iterations == 0


def test_decode_binary_noiseless_nonzero_syndrome():
    with pytest.raises(ValueError, match="Nonzero syndrome"):
        decode_binary(H_demo, s_demo, p=0)


def test_decode_binary_iteration_limit():
    # a solution exists, but symmetric BP beliefs cannot select exactly one bit
    H = jnp.array([[1, 1]], dtype=jnp.uint8)
    syndrome = jnp.array([1], dtype=jnp.uint8)
    max_iter = 3

    estimate, consistent, iterations = decode_binary(
        H, syndrome, p=0.06, max_iter=max_iter
    )

    assert not consistent
    assert iterations == max_iter
    assert not jnp.array_equal((H @ estimate) % 2, syndrome)


# --- Complete decoder: input validation ---


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_decode_binary_invalid_probability(p):
    with pytest.raises(ValueError, match="0 <= p <= 1"):
        decode_binary(H_demo, s_demo, p=p)


@pytest.mark.parametrize("p,max_iter", [(0.06, 0), (0.06, -1), (0, -1)])
def test_decode_binary_invalid_max_iter(p, max_iter):
    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        decode_binary(H_demo, s_demo, p=p, max_iter=max_iter)


def test_decode_binary_invalid_shapes():
    with pytest.raises(ValueError, match="Incompatible"):
        decode_binary(H_demo, jnp.array([1], dtype=jnp.uint8), p=0.06)

    with pytest.raises(ValueError, match="Incompatible"):
        decode_binary(jnp.array([1, 1, 0, 0], dtype=jnp.uint8), s_demo, p=0.06)


@pytest.mark.parametrize("invalid_entry", [2.0, 0.5])
def test_decode_binary_nonbinary_matrix(invalid_entry):
    H = jnp.asarray(H_demo, dtype=jnp.float32).at[0, 0].set(invalid_entry)
    with pytest.raises(ValueError, match="must be binary"):
        decode_binary(H, s_demo, p=0.06)


@pytest.mark.parametrize("invalid_entry", [2.0, 0.5])
def test_decode_binary_nonbinary_syndrome(invalid_entry):
    syndrome = jnp.array([1.0, invalid_entry], dtype=jnp.float32)
    with pytest.raises(ValueError, match="must be binary"):
        decode_binary(H_demo, syndrome, p=0.06)
