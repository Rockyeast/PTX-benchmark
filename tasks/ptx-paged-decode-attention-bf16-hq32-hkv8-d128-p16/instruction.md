# Implement paged GQA decode attention in direct PTX

Create `/app/kernel.ptx`. Write the PTX directly; do not compile CUDA C++,
Triton, Python, or another higher-level language into the submission. You may
use `ptxas` and `cuobjdump`; `nvcc`, NVRTC, NVVM, and Triton are unavailable.

Define the visible entry `paged_decode_fwd_bf16_hq32_hkv8_d128_p16` with this
logical parameter order:

```text
(const bf16* q, const bf16* k_cache, const bf16* v_cache,
 const uint32_t* page_table, bf16* output, uint32_t kv_len)
```

Pointers are 64-bit and `kv_len` is u32 in `[0,4096]`.

## Fixed data contract

```text
q          : [32,128], bf16
k_cache    : [320,16,8,128], bf16
v_cache    : [320,16,8,128], bf16
page_table : [256], uint32
output     : [32,128], bf16
```

For logical KV token `t`:

```text
physical_page = page_table[t / 16]
page_offset   = t % 16
cache_index   = ((((physical_page*16)+page_offset)*8+kv_head)*128+feature)
```

Physical page IDs are shuffled and are not logical page numbers. Do not read
tokens at or beyond `kv_len`, including unused slots in the final page.

For query head `hq`, use `kv_head = hq / 4` and compute stable scaled
dot-product attention over tokens `0 <= t < kv_len`:

```text
score[t] = sum_d(float(q[hq,d]) * float(K[t,kv_head,d])) / sqrt(128)
p[t]     = exp(score[t]-max(score)) / sum_j(exp(score[j]-max(score)))
output[hq,d] = bf16(sum_t(p[t] * float(V[t,kv_head,d])))
```

Use at least FP32 for QK, Softmax, and PV. When `kv_len==0`, write BF16 `+0`
to all 4096 output elements without reading `page_table`, K, or V. Do not
modify inputs or access outside any allocation.

## Fixed launch

```text
grid:                  (32,1,1)
block:                 (128,1,1)
dynamic shared memory: 0
```

Each block is one query head (`blockIdx.x`). Static shared memory is allowed;
4096 FP32 scores require 16384 bytes.

Correctness, guards, input immutability, assembly/module/entry, and CUDA errors
are hard gates. Decoded BF16 output is checked with
`abs(got-want) <= 2e-2 + 2e-2*abs(want)`. Performance is compared with a fixed
same-ABI Reference at `kv_len=128,1024,4096` on NVIDIA B200.
