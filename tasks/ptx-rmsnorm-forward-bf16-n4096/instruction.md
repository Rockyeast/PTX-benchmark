# Implement a BF16 RMSNorm forward PTX kernel for N=4096

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. It must define a
visible entry named `rmsnorm_fwd_bf16_n4096` with this logical signature:

Write or edit the PTX directly. Do not generate the submission by compiling
CUDA C++, Triton, Python, or another higher-level language. `ptxas` may be used
to validate PTX syntax, but `nvcc` and NVRTC are not available.

```text
rmsnorm_fwd_bf16_n4096(
    const bfloat16* x,
    const float* weight,
    bfloat16* output,
    uint32_t M,
    float eps)
```

The tensors are contiguous row-major arrays:

```text
x      : [M, 4096], bfloat16
weight : [4096],    float32
output : [M, 4096], bfloat16
```

For each row `m`, compute in float32:

```text
mean_sq = sum(float32(x[m, j]) ** 2 for j in 0..4095) / 4096
rstd = rsqrt(mean_sq + eps)
output[m, j] = bfloat16(float32(x[m, j]) * rstd * weight[j])
```

The verifier uses this fixed launch configuration:

```text
block dimensions: (128, 1, 1)
grid dimensions:  (max(1, M), 1, 1)
dynamic shared memory: 1024 bytes
```

Each block corresponds to one row. The kernel must handle `M=0`, must not
modify either input, and must not access memory outside any tensor. Pointer
parameters use 64-bit addresses; `M` is uint32 and `eps` is float32.

The target device is NVIDIA B200; targeting `sm_100` is recommended.
Correctness is required before performance is measured. Only
`/app/kernel.ptx` is graded.
