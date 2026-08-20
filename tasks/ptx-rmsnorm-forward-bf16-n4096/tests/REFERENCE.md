# Feasibility reference provenance

`reference.cu` is a small task-owned CUDA implementation of the mathematical
RMSNorm contract. `reference.ptx` was compiled once during task authoring and
uses exactly the Candidate ABI and launch configuration. The verifier loads
the pinned PTX and does not contain or invoke `nvcc`.

Build provenance:

- source: `reference.cu`, SHA-256
  `ee6e9befd557831dbd04740d431cfcdf80193fa268f19489d1541c4bc15d3c90`;
- compiler: NVIDIA `nvcc` 13.2.78, NVVM 22.0.0;
- command: `nvcc --ptx --optimize=3 --gpu-architecture=compute_100`;
- artifact: `reference.ptx`, SHA-256
  `fe79babf4681310cef875b51628e75d800a4ba5326579b3f0f7a29cae24c6402`.

It was written for the initial interface smoke test. It is not copied from and
must not be reported as the performance of INT21's optimized implementation.

The operation, tested shape, public tolerance, and the choice of 128 threads
for the BF16 `N=4096` plain-forward specialization were checked against:

- <https://github.com/Int21-AI/RMSNorm-B200>
- revision `6b0a8beeaa306b1a55c0dc5c5e03dafeda914c5f`
- upstream license: MIT

Before this task becomes a benchmark release, the task-owned Reference must be
reviewed against the INT21 fast path and an independently measured INT21
operator result must be recorded. Correctness never comes from either
performance implementation; the verifier computes the formula independently.
