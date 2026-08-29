# ptx-rmsnorm-forward-bf16-n4096 validation

Feasibility implementation of a BF16 RMSNorm plain-forward PTX task specialized
for `N=4096` on NVIDIA B200.

Current status: feasibility-validated on an NVIDIA B200. The first Harbor
Oracle run passed six correctness cases and three performance shapes in 18
seconds. The Oracle and hidden Reference are intentionally the same simple CUDA
algorithm, precompiled and pinned as PTX during task authoring. They produced
raw speedup `1.000141864` as expected. The evaluated environment no longer
contains `nvcc` or NVRTC.

This result validates the raw PTX ABI, independent correctness checks, and
timing pipeline. It does not establish model discrimination or parity with
INT21's optimized B200 implementation. Those require an INT21-backed external
comparison, negative fixtures, and real Agent runs.

Design contract:
[`ptx-rmsnorm-forward-bf16-n4096-spec.md`](ptx-rmsnorm-forward-bf16-n4096-spec.md).
