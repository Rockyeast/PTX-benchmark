# ptx-bench/ptx-rope-forward-bf16-d128

Direct-PTX B200 pilot for out-of-place BF16 NeoX-style RoPE with fixed head
dimension 128, 32 query heads, and 8 KV heads.

## Validation status

- Lifecycle: `feasibility-validated`; not yet verifier-validated, agent-calibrated,
  or frozen.
- Target: NVIDIA B200 (`sm_100`).
- Oracle: handwritten PTX, checked against an independent CPU float32/BF16 truth.
- Correctness cases: `num_tokens = 0, 1, 2, 3, 7, 31, 257`.
- Performance cases: `num_tokens = 256, 4096, 32768`.
- Reward: correctness hard gate followed by capped geometric-mean speedup against
  the task-owned, same-ABI PTX reference.

Full Job IDs, task digests, measurements, and negative-test evidence are kept in
`docs/tasks/ptx-rope-forward-bf16-d128-validation.md`, outside this task directory
so recording a new run does not change the task digest being reported.

The environment policy is checked again during grading by a checker and trusted
documentation manifest uploaded from `tests/` after the Agent phase. The
verifier therefore does not trust the Agent-visible checker installed under
`/usr/local/bin` or the runtime manifest under `/opt/ptx-docs`.

That Job used a temporary Oracle-only copy whose Agent network policy inherited
`public`, because the current Harbor Singularity backend cannot switch from a
public setup phase to a no-network Agent phase after the container has started.
No untrusted Agent ran in that Job. The canonical task remains `no-network` for
the Agent and must not be described as release-ready on Singularity until this
backend constraint is resolved.

Two author-side negative checks also passed on B200: a PTX file with the required
entry renamed was rejected before module loading, and a valid PTX file with the
sine term forced to zero was rejected by the numerical check at `num_tokens=2`.
Both retained the fail-closed reward `{reward: 0, correctness: 0}`. The remaining
negative matrix and repeated timing runs are still required before upgrading the
task to `validated`.
