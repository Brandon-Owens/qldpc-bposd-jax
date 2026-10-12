# qldpc-bposd-jax

## BP+OSD Decoding for Quantum LDPC Codes in JAX

This project develops a from-scratch implementation of belief propagation with ordered-statistics decoding (BP+OSD) in JAX for CSS quantum low-density parity-check (qLDPC) codes.

The initial target is the \(\lbrack\lbrack72,12,6\rbrack\rbrack\) bivariate bicycle code.

### Project Goals

- Construct and verify the CSS parity-check matrices.
- Simulate Pauli errors and compute their syndromes under a code-capacity noise model.
- Implement belief propagation (BP) decoding in JAX.
- Implement ordered-statistics decoding (OSD) as a post-processing step.
- Benchmark logical failure rate as a function of physical error rate.

### Project Structure

- `src/qldpc_bposd_jax/` — Code construction, noise simulation, decoding, and benchmarking.
- `tests/` — Automated tests for mathematical correctness and implementation.
- `notebooks/` — Mathematical derivations, demonstrations, and numerical experiments.

### Development

The project uses Python 3.12, JAX, NumPy, pytest, Ruff, and `uv` for dependency management.

Install the project dependencies with:

```bash
uv sync
```

### Status

Under development as part of the Erdos Institute Quantum Computing Bootcamp.
