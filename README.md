# qldpc-bposd-jax

## BP+OSD Decoding for a Bivariate Bicycle Code in JAX

This project develops a from-scratch implementation of belief propagation with ordered-statistics decoding (BP+OSD) in JAX for the $[[72,12,6]]$ bivariate bicycle (BB) code introduced by Bravyi et al. in [High-threshold and low-overhead fault-tolerant quantum memory](https://arxiv.org/abs/2308.07915).

This CSS quantum low-density parity-check (qLDPC) code encodes 12 logical qubits in 72 physical qubits and has distance 6. Decoding is studied under a code-capacity depolarizing Pauli noise model with ideal syndrome measurements.

### Implementation Status

**Completed**

- Constructed and verified the BB code's CSS parity-check matrices.
- Implemented Pauli error sampling and syndrome computation.
- Implemented binary sum-product belief propagation (BP) in JAX, including syndrome-based early stopping.
- Validated BP against exact posterior marginals on a small cycle-free Tanner graph.
- Implemented checks for syndrome consistency and logical decoding success using stabilizer equivalence.
- Tested BP decoding on the BB code, with automated tests for mathematical correctness and edge cases.

**Remaining**

- Implement ordered-statistics decoding (OSD) in JAX as a post-processing step for BP.
- Benchmark logical failure rate as a function of physical error rate, including a comparison of BP and BP+OSD.

### Project Structure

- `src/qldpc_bposd_jax/` — Code construction, noise simulation, decoding, and logical-success checks.
- `tests/` — Automated tests for mathematical correctness and implementation.
- `notebooks/` — Mathematical derivations, demonstrations, and numerical experiments.

### Development

The project uses Python 3.12, JAX, NumPy, `galois`, pytest, Ruff, and `uv` for dependency management.

Install the project dependencies with:

```bash
uv sync
```

Run the automated tests and code-quality checks with:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Developed as part of the Erdős Institute Quantum Computing Bootcamp.