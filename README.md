# qldpc-bposd-jax

## BP+OSD Decoding for a Bivariate Bicycle Code in JAX

This project aims to implement belief propagation with ordered-statistics decoding (BP+OSD) from scratch in JAX for the $[[72,12,6]]$ bivariate bicycle (BB) code introduced by Bravyi et al. in [High-threshold and low-overhead fault-tolerant quantum memory](https://arxiv.org/abs/2308.07915).

This CSS quantum low-density parity-check (qLDPC) code encodes 12 logical qubits in 72 physical qubits and has distance 6.

### Project Goals

- Construct and verify the CSS parity-check matrices.
- Simulate Pauli errors and compute their syndromes under a code-capacity noise model with ideal syndrome measurements.
- Implement belief propagation (BP) decoding in JAX.
- Implement ordered-statistics decoding (OSD) in JAX as a post-processing step for BP.
- Benchmark logical failure rate as a function of physical error rate.

### Planned Project Structure

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