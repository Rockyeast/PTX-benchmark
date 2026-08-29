# Author-only negative fixtures

These PTX files are task-author regression inputs. Harbor does not expose the
task `tests/` directory during the Agent phase; it uploads `/tests` only for the
trusted verifier phase.

- `missing-gate-multiply.ptx`: computes `sigmoid(gate) * up` instead of
  `silu(gate) * up`.
- `partial-output.ptx`: writes only indices `0..511` of each 4096-element row.
- `slow-correct.ptx`: produces correct output but uses only thread 0 in each
  block and serializes all 4096 elements; it calibrates continuous performance
  scoring rather than Candidate rejection.

The first two fixtures must assemble successfully for `sm_100`, then be rejected by
`verify_and_benchmark.py --correctness-only` with exit status `1`,
`status=candidate_rejected`, and zero reward.

The slow-correct fixture is the exception: it must pass correctness and receive
a materially lower performance score than the fixed vectorized Reference.
