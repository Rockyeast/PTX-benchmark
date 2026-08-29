# Implement BF16 GEMM plus rowwise FP8 E4M3 quantization in direct PTX

Create `/app/kernel.ptx` with visible entry
`gemm_rowwise_fp8e4m3_m128_n128_k128`. Write PTX directly. Do not compile CUDA
C++, Triton, Python, or another high-level language into the submission.
`ptxas` and `cuobjdump` are allowed; `nvcc`, NVRTC, NVVM, and Triton are absent.
Online official PTX documentation is permitted.

Logical parameter order, all as 64-bit pointers:

```text
(const bf16* a, const bf16* b, uint8* output_e4m3, float* row_scale)
```

## Fixed operation

`A` and `B` are row-major BF16 `[128,128]` matrices. Compute an FP32-accumulated
matrix product with B interpreted transposed:

```text
C[m,n] = sum_k float(A[m,k]) * float(B[n,k])
```

For every row independently:

```text
amax         = max_n abs(C[m,n])
row_scale[m] = amax / 448, if amax > 0; otherwise 1
output[m,n]  = FP8 E4M3 round-to-nearest(C[m,n] / row_scale[m])
```

Use NVIDIA finite E4M3 encoding with maximum finite magnitude 448. Inputs are
finite. Preserve A/B, write all 16384 output bytes and all 128 FP32 scales, and
do not depend on previous output contents.

## Fixed launch ABI

```text
grid dimensions:       (128, 1, 1)
block dimensions:      (128, 1, 1)
dynamic shared memory: 0 bytes
```

Block `m` owns output row `m`; thread `n` may own column `n`. Static shared
storage and any PTX instructions supported by B200 are allowed.

## Scoring

Correctness checks row scales and decodes every FP8 byte before comparing with
independent FP32 GEMM truth. It also checks input immutability, memory guards,
module loading, zero rows, sign/range behavior, and CUDA errors. Quantization
tolerance reflects E4M3 precision; incorrect scaling or non-E4M3 bytes fail.

Correctness is a hard gate. Performance reward is
`min(1, reference_time/candidate_time)` against fixed same-ABI PTX on the same
B200. Test a complete Candidate early and optimize within 5400 seconds.
