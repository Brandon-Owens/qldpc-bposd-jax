"""Binary sum-product belief propagation for CSS-syndrome decoding

Notation:
    n = number of data nodes (physical error bits)
    m = number of check nodes
    E = number of Tanner graph edges
    H = binary parity-check matrix of shape = (m,n)
"""

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike


def binary_prior_llr(p: float) -> Array:
    """Compute the prior binary LLR under depolarizing noise for 0 < p <= 1."""
    if not 0.0 < p <= 1.0:
        raise ValueError("Require 0 < p <= 1 for a finite prior LLR.")

    q = 2 * p / 3
    # jnp.log1p(-q) computes ln(1-q) with better numerical accuracy when q is small
    return jnp.log1p(-q) - jnp.log(q)


def check_to_data_update(
    data_to_check: Array, check_idx: Array, syndrome: Array
) -> Array:
    """Compute one check-to-data LLR update per Tanner graph edge.

    Arguments:
        data_to_check: Incoming data-to-check LLRs, shape (E,)
        check_idx: Check-node index for each edge, shape (E,)
        syndrome: Observed binary syndrome, shape (m,)

    Returns:
        Updated check-to-data LLRs, shape (E,)
    """
    num_edges = check_idx.shape[0]

    # for each outgoing edge, select other edges incident to the same check node
    # and store the info as a boolean mask
    same_check = check_idx[:, None] == check_idx[None, :]
    # exclude incoming updates from the same edge
    exclude_self = ~jnp.eye(num_edges, dtype=bool)
    mask = same_check & exclude_self

    # convert incoming LLRs to probability biases
    incoming_biases = jnp.tanh(data_to_check / 2)

    # compute the parity bias for each outgoing edge
    # note: excluded edges contribute the mult. identity 1.0
    selected_biases = jnp.where(mask, incoming_biases[None, :], 1.0)
    parity_biases = jnp.prod(selected_biases, axis=1)

    # incorporate the observed syndrome
    syndrome_signs = jnp.where(syndrome[check_idx] == 0, 1.0, -1.0)
    outgoing_biases = syndrome_signs * parity_biases

    # clip to avoid infinite LLRs from finite-precision rounding to +1 or -1
    eps = jnp.finfo(outgoing_biases.dtype).eps
    outgoing_biases = jnp.clip(outgoing_biases, -1 + eps, 1 - eps)

    return 2 * jnp.arctanh(outgoing_biases)


def data_to_check_update(
    prior_llrs: Array, check_to_data: Array, data_idx: Array
) -> Array:
    """Compute outgoing data-to-check LLR updates along all Tanner graph edges.

    Arguments:
        prior_llrs: Prior LLR for each data node, shape (n,)
        check_to_data: Incoming LLRs, shape (E,)
        data_idx: Data-node index for each edge, shape (E,)

    Returns:
        Data-to-check LLRs, shape (E,)
    """

    # accumulate incoming messages at each data node
    incoming_sums = jnp.zeros_like(prior_llrs).at[data_idx].add(check_to_data)

    # gather node totals by edge and exclude the receiving edge
    return prior_llrs[data_idx] + incoming_sums[data_idx] - check_to_data


def posterior_llrs(prior_llrs: Array, check_to_data: Array, data_idx: Array) -> Array:
    """Compute one posterior LLR estimate per data node, shape (n,)"""
    incoming_sums = jnp.zeros_like(prior_llrs).at[data_idx].add(check_to_data)
    return prior_llrs + incoming_sums


def hard_decision(posterior: Array) -> Array:
    """Convert posterior LLRs into a binary error-vector estimate, breaking ties toward zero."""
    return (posterior < 0).astype(jnp.uint8)


def syndrome_matches(H: Array, error_estimate: Array, syndrome: Array) -> Array:
    """Check whether an estimated error reproduces the observed syndrome."""
    return jnp.array_equal(syndrome, (H @ error_estimate) % 2)


def bp_iteration(
    prior_llrs: Array,
    data_to_check: Array,
    check_idx: Array,
    data_idx: Array,
    syndrome: Array,
) -> tuple[Array, Array, Array, Array]:
    """Perform one flooding-schedule BP iteration.

    Returns:
    tuple containing:
        - Data-to-check LLRs, shape (E,)
        - Check-to-data LLRs, shape (E,)
        - Posterior LLRs, shape (n,)
        - Binary error estimate, shape (n,)
    """
    # compute the new check-to-data messages
    new_check_to_data = check_to_data_update(data_to_check, check_idx, syndrome)

    # compute the new data-to-check messages
    new_data_to_check = data_to_check_update(prior_llrs, new_check_to_data, data_idx)

    # compute the posterior LLR estimates
    posterior = posterior_llrs(prior_llrs, new_check_to_data, data_idx)

    # convert posterior LLRs into a binary error estimate
    estimate = hard_decision(posterior)

    return new_data_to_check, new_check_to_data, posterior, estimate


def bp_decode(
    H: Array,
    syndrome: Array,
    prior_llrs: Array,
    check_idx: Array,
    data_idx: Array,
    max_iter: int = 50,
) -> tuple[Array, bool, int]:
    """Decode a binary syndrome using iterative belief propagation.

        In short, we run BP until the hard decision matches the syndrome or iterations run out.

    Arguments:
        H: Binary parity-check matrix, shape (m, n)
        syndrome: Observed syndrome, shape (m,)
        prior_llrs: Prior LLRs, shape (n,)
        check_idx: Check-node index for each edge, shape (E,)
        data_idx: Data-node index for each edge, shape (E,)
        max_iter: Maximum number of iterations; must be at least 1

    Returns:
        (estimate, syndrome_consistent, iterations).
        Note: Syndrome consistency does not imply logical decoding success.
    """
    if max_iter < 1:
        raise ValueError("max_iter must be greater than or equal to 1.")

    # initialize data-to-check messages
    data_to_check = prior_llrs[data_idx]

    for t in range(1, max_iter + 1):
        # perform one BP iteration
        # we never use new_check_to_data or posterior so they get underscored
        # also, note that we update the data_to_check variable with the outgoing data-to-check messages each pass
        data_to_check, _, _, estimate = bp_iteration(
            prior_llrs, data_to_check, check_idx, data_idx, syndrome
        )

        # check whether the estimated error matches the syndrome
        if bool(syndrome_matches(H, estimate, syndrome)):
            return estimate, True, t

    return estimate, False, max_iter


def decode_binary(
    H: ArrayLike, syndrome: ArrayLike, p: float, max_iter: int = 50
) -> tuple[Array, bool, int]:
    """Decode one binary error component using sum-product BP.

    Assumes independent depolarizing noise across qubits and perfect syndrome
    measurements. The two CSS error components are decoded separately.

    Arguments:
        H: binary parity-check matrix, shape (m, n)
        syndrome: observed binary syndrome, shape (m,)
        p: physical depolarizing error probability, 0 <= p <= 1
        max_iter: maximum BP iterations; must be at least 1

    Returns:
        (estimate, syndrome_consistent, iterations). For p=0 and a zero
        syndrome, returns the zero error vector without running BP.

    Raises:
        ValueError: If p, max_iter, array shapes, or binary entries are invalid,
            or if p=0 with a nonzero syndrome under perfect measurements.
    """
    if not 0 <= p <= 1:
        raise ValueError("Require 0 <= p <= 1.")

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")

    H = jnp.asarray(H)
    syndrome = jnp.asarray(syndrome)

    if H.ndim != 2 or syndrome.shape != (H.shape[0],):
        raise ValueError("Incompatible parity-check matrix and syndrome shapes.")

    if not bool(jnp.all((H == 0) | (H == 1))):
        raise ValueError("Parity-check matrix entries must be binary.")

    if not bool(jnp.all((syndrome == 0) | (syndrome == 1))):
        raise ValueError("Syndrome entries must be binary.")

    H = H.astype(jnp.uint8)
    syndrome = syndrome.astype(jnp.uint8)

    # under perfect syndrome measurements p=0 implies zero error
    if p == 0:
        if bool(jnp.any(syndrome != 0)):
            raise ValueError(
                "Nonzero syndrome is incompatible with p=0 "
                "and perfect syndrome measurements."
            )
        return jnp.zeros(H.shape[1], dtype=jnp.uint8), True, 0

    check_idx, data_idx = np.nonzero(np.asarray(H))
    check_idx = jnp.asarray(check_idx)
    data_idx = jnp.asarray(data_idx)

    prior_llrs = jnp.full(H.shape[1], binary_prior_llr(p))

    return bp_decode(H, syndrome, prior_llrs, check_idx, data_idx, max_iter)
