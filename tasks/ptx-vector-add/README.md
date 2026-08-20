# ptx-bench/ptx-vector-add

First performance-measurement calibration task for PTX-Bench.

Current status: implemented and runtime-validated on `catalyst-b200` with NVIDIA
B200 and CUDA 13.2. Oracle correctness, missing-entry rejection, output-canary
rejection, slow-correct scoring, raw timing logs, Harbor multi-reward propagation,
and a real Codex Agent run have all passed.

## Evaluation boundary

- `instruction.md` defines one `vector_add` operation contract.
- `solution/solve.sh` writes a known-good PTX submission for Harbor's Oracle.
- `tests/verify_and_benchmark.py` independently computes CPU `float32` expected
  values, checks input preservation and output guards, and only then benchmarks.
- `tests/reference.cu` is compiled by NVIDIA `nvcc` into a hidden performance
  baseline. It does not decide correctness.
- `tests/fixtures/` contains reproducible missing-entry, output-overrun, and
  deliberately-slow submissions used only for task-author validation.
- different vector lengths are verifier cases inside this one task, not separate
  Harbor tasks.

On success, the verifier writes numeric `reward`, `correctness`, `performance`,
and `speedup_vs_reference` fields to `/logs/verifier/reward.json`. Detailed raw
timings and provenance go to `/logs/verifier/metrics.json`.

The first three Oracle calibration rewards were `0.990683`, `0.997697`, and
`0.987482`. The deliberately slow correct fixture received `0.267804`; both
negative fixtures received `0`.

Codex CLI 0.148.0 with `gpt-5.6-sol` directly wrote PTX, checked it with
`ptxas`, passed all correctness cases, and obtained raw
`speedup_vs_reference=1.000609` (reward capped at `1`). This confirms the task
pipeline, but also confirms vector add is too easy to serve as a discriminating
final benchmark task.

The design contract is documented in
[`../../docs/tasks/ptx-vector-add-spec.md`](../../docs/tasks/ptx-vector-add-spec.md).
