# Paged sparse attention backward validation

This task combines a paged BF16 KV cache, per-query-block CSR sparsity, causal
masking, GQA head sharing, forward `output/LSE` inputs, and full FP32
`dQ/dK/dV`. It is deliberately correctness-only until Agent calibration shows
that the ABI and hidden cases are stable.

Status: implementation in progress; not release-ready.

## Provenance and trusted artifacts

The paged NHD cache, GQA head mapping, output/LSE contract, and block-sparse
layout are grounded in FlashInfer revision
`103fcf862b902e100775ac781a0de36d067de3a4`, Apache-2.0. FlashInfer is an
inference library and does not provide this exact backward task. The backward
calculus, two-entry launch ABI, CPU truth, and CUDA feasibility implementation
are task-owned; this is not presented as an upstream FlashInfer kernel.

```text
reference.cu          b2c19e07f9de7e431b087047c66f0e3b1b464092f92676cf24ca0d9bc3120a06
reference.ptx         8b024f5e7a1b8c18eff75ada88eb04becc5406835c3adbcff6ba9d49a07a416a
paged_sparse_bwd_truth.c 7956d6feeb86849ef0ec82313b1f81b84fdc4ff1c9e716174cb83c5911a6edd5
oracle.ptx             8b024f5e7a1b8c18eff75ada88eb04becc5406835c3adbcff6ba9d49a07a416a
```

`reference.ptx` and `oracle.ptx` were author-built with CUDA 13.2.1:

```bash
nvcc -ptx -O3 --use_fast_math -arch=compute_100 -code=compute_100 \
  -o reference.ptx reference.cu
```
