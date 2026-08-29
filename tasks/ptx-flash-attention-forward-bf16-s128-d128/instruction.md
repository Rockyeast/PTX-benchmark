# Implement causal BF16 Flash Attention in direct PTX

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. Only this file is
graded. Write or edit PTX directly; do not generate the submission by compiling
CUDA C++, Triton, Python, or another higher-level language. `ptxas` and
`cuobjdump` may be used to validate PTX. `nvcc`, NVRTC, NVVM, and Triton are not
available.

The file must define a visible entry named
`flash_attn_fwd_bf16_s128_d128` with this logical parameter order:

```text
flash_attn_fwd_bf16_s128_d128(
    const bf16* q,
    const bf16* k,
    const bf16* v,
    bf16* output,
    uint32_t num_heads)
```

The four device pointers are passed as 64-bit values and `num_heads` as a
32-bit unsigned integer.

## Tensor layout and math

All tensors are contiguous NHD arrays:

```text
q, k, v, output : [128, num_heads, 128], bfloat16
linear index    : ((token * num_heads + head) * 128 + feature)
```

For every head `h`, query position `i`, and output feature `d`, compute causal
scaled dot-product attention:

```text
score[j] = sum_f(float(q[i,h,f]) * float(k[j,h,f])) / sqrt(128), j <= i
m        = max_j(score[j])
p[j]     = exp(score[j] - m) / sum_t(exp(score[t] - m))
acc[d]   = sum_j(p[j] * float(v[j,h,d]))
output[i,h,d] = bf16(acc[d])
```

Keys with `j > i` must not contribute. QK, Softmax, and PV accumulation use at
least FP32 precision. Inputs are finite. Approximate exponential instructions
are permitted if the decoded BF16 output satisfies
`abs(got-want) <= 2e-2 + 2e-2*abs(want)`.

The tensors do not alias. Do not modify Q, K, or V. Write every output element
and do not access outside any allocation.

## Fixed launch ABI

The verifier launches the kernel with:

```text
block dimensions:      (128, 1, 1)
grid dimensions:       (max(1, num_heads * 128), 1, 1)
dynamic shared memory: 0 bytes
```

Each block represents one `(query_position, head)` pair. A useful mapping is:

```text
query_position = blockIdx.x / num_heads
head           = blockIdx.x % num_heads
```

When `num_heads == 0`, the verifier still launches one block. Return before
division or any global-memory access. Static `.shared` storage is allowed, but
no dynamic shared memory is provided.

Correctness, input immutability, memory guards, module loading, and CUDA errors
are hard gates. Performance is compared with a fixed same-ABI Reference for
`num_heads` 1, 8, and 32 on NVIDIA B200. Correctness must pass before any
performance score is awarded.
