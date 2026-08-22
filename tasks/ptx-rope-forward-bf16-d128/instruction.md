# Implement a BF16 RoPE forward PTX kernel for head dimension 128

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. Write or edit the PTX
directly. Do not generate the submission by compiling CUDA C++, Triton, Python,
or another higher-level language. `ptxas` may be used to validate PTX syntax,
but `nvcc` and NVRTC are not available.

The file must define a visible entry named `rope_forward_bf16_d128` with this
logical signature and parameter order:

```text
rope_forward_bf16_d128(
    const bfloat16* q,
    const bfloat16* k,
    const int32_t* positions,
    const float* cos_sin_cache,
    bfloat16* q_out,
    bfloat16* k_out,
    uint32_t num_tokens)
```

The C-style declaration describes the logical ABI. The submission itself must
use a valid PTX `.entry` and `.param` declarations. Device pointers are passed
as 64-bit values, and `num_tokens` is passed as a 32-bit unsigned integer.

All tensors are contiguous row-major arrays:

```text
q             : [num_tokens, 32, 128], bfloat16
k             : [num_tokens,  8, 128], bfloat16
positions     : [num_tokens],          int32
cos_sin_cache : [8192, 128],           float32
q_out         : [num_tokens, 32, 128], bfloat16
k_out         : [num_tokens,  8, 128], bfloat16
```

Inputs and outputs do not alias. Each `positions[t]` is in `[0, 8191]`.
Positions may repeat or appear in any order and are not necessarily equal to
the token index.

Apply NeoX-style split-half rotary position embedding independently to every Q
and K head. For every token `t`, every head, and `d` in `[0, 63]`, compute with
float32 arithmetic:

```text
p = positions[t]
c = cos_sin_cache[p, d]
s = cos_sin_cache[p, 64 + d]

out[..., d]      = bfloat16(float32(x[..., d])      * c
                          - float32(x[..., 64 + d]) * s)
out[..., 64 + d] = bfloat16(float32(x[..., 64 + d]) * c
                          + float32(x[..., d])      * s)
```

Use this formula for both `q -> q_out` and `k -> k_out`. This is split-half
layout: dimension `d` is paired with dimension `64 + d`. It is not the
interleaved even/odd layout. Use the supplied cache values; do not assume that
the position equals `t` or reconstruct the cache inside the kernel.

The verifier uses this fixed launch configuration:

```text
block dimensions: (256, 1, 1)
grid dimensions:  (max(1, num_tokens), 1, 1)
dynamic shared memory: 512 bytes
```

Each block corresponds to one token. The 512-byte dynamic shared-memory region
is available for use but does not have to be used. When `num_tokens` is zero,
one block is still launched and the kernel must return without accessing any
tensor.

The kernel must write every element of both outputs, must not modify any input,
and must not access memory outside any tensor. The target device is NVIDIA B200;
targeting `sm_100` is recommended.

Correctness is required before performance is measured. Avoid unnecessary work
and global-memory traffic. Only `/app/kernel.ptx` is graded; do not submit CUDA
source, a cubin, a host program, or a separate launch configuration.
