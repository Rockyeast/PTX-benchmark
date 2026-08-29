# Implement BF16 Flash Attention backward in direct PTX

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. Only this file is
graded. Write or edit PTX directly; do not generate the submission by compiling
CUDA C++, Triton, Python, or another higher-level language. `ptxas` and
`cuobjdump` may be used to validate PTX. `nvcc`, NVRTC, NVVM, and Triton are not
available. Online access to official PTX documentation is permitted.

The file must define a visible entry named
`flash_attn_bwd_bf16_s128_d128` with this logical parameter order:

```text
flash_attn_bwd_bf16_s128_d128(
    const bf16* q,
    const bf16* k,
    const bf16* v,
    const bf16* grad_output,
    float* grad_q,
    float* grad_k,
    float* grad_v)
```

All seven device pointers are passed as 64-bit values.

## Tensor layout and math

Every tensor is a contiguous row-major matrix with fixed shape `[128, 128]`.
`q`, `k`, `v`, and `grad_output` contain BF16 values. `grad_q`, `grad_k`, and
`grad_v` contain FP32 values.

The forward operation whose gradient is required is non-causal scaled
dot-product attention:

```text
scale       = 1 / sqrt(128)
score[i,j]  = scale * sum_d(float(q[i,d]) * float(k[j,d]))
p[i,:]      = softmax(score[i,:])
output[i,d] = sum_j(p[i,j] * float(v[j,d]))
```

For the supplied `grad_output = dL/doutput`, compute all three gradients:

```text
grad_v[j,d] = sum_i p[i,j] * grad_output[i,d]
dP[i,j]     = sum_d grad_output[i,d] * v[j,d]
delta[i]    = sum_j p[i,j] * dP[i,j]
dS[i,j]     = p[i,j] * (dP[i,j] - delta[i])
grad_q[i,d] = scale * sum_j dS[i,j] * k[j,d]
grad_k[j,d] = scale * sum_i dS[i,j] * q[i,d]
```

Use at least FP32 accumulation. Inputs are finite and do not alias outputs.
Do not modify the four inputs. The output buffers are not pre-zeroed as part of
the contract: the kernel must write every output element and must not depend on
their previous contents.

## Fixed launch ABI

The verifier launches exactly one block:

```text
grid dimensions:       (1, 1, 1)
block dimensions:      (128, 1, 1)
dynamic shared memory: 0 bytes
```

Static shared storage is allowed. No additional launch metadata is submitted.

## Correctness and performance

Correctness uses multiple hidden input families, input immutability checks,
memory guards, full FP32 gradient comparison, module-load checks, and CUDA error
checks. Each finite output must satisfy approximately
`abs(got-want) <= 2e-2 + 3e-2*abs(want)`.

Correctness is a hard gate. After it passes, Candidate and fixed same-ABI
Reference PTX are timed on the same NVIDIA B200. The raw speedup is
`reference_time / candidate_time`; the task reward is
`min(1, raw_speedup)`. A correct but slower implementation therefore receives a
continuous score rather than zero.

Iterate with the allowed PTX-level tools, test the first complete Candidate
early, and keep the fastest version that remains correct. You may finish before
the 6000-second budget when further useful improvement is unlikely.
