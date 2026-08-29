# Implement paged sparse GQA backward in direct PTX

Create `/app/kernel.ptx` containing both visible entries below. Write or edit
PTX directly. Do not compile CUDA C++, Triton, Python, or another high-level
language into the submission. `ptxas` and `cuobjdump` are allowed; `nvcc`,
NVRTC, NVVM, and Triton are unavailable. Online official PTX documentation is
permitted.

This is one logical attention-backward operation implemented by two launches:

```text
paged_sparse_attn_bwd_zero_bf16_q128_hq8_hkv2_d128_p16(
    float* grad_q, float* grad_k_cache, float* grad_v_cache)

paged_sparse_attn_bwd_main_bf16_q128_hq8_hkv2_d128_p16(
    const bf16* q, const bf16* k_cache, const bf16* v_cache,
    const bf16* output, const bf16* grad_output, const float* lse,
    const int32* page_table, const int32* row_offsets,
    const int32* page_indices,
    float* grad_q, float* grad_k_cache, float* grad_v_cache)
```

All pointers are 64-bit kernel parameters.

## Fixed tensors and metadata

```text
Q_LEN=128, Hq=8, Hkv=2, D=128
PAGE_SIZE=16, LOGICAL_PAGES=32, PHYSICAL_PAGES=32

q/output/grad_output : [128,8,128], NHD BF16
k_cache/v_cache      : [32,16,2,128], page-token-head-dim BF16
lse                   : [128,8], FP32 natural-log softmax normalizer
page_table            : [32], int32 logical-page -> physical-page permutation
row_offsets           : [9], int32 CSR offsets for 8 query blocks
page_indices          : [row_offsets[8]], int32 logical page IDs
grad_q                : [128,8,128], FP32
grad_k_cache/v_cache  : [32,16,2,128], FP32 physical-page layout
```

Query block size is 16. CSR row `qi/16` contains 2 through 8 unique logical
page IDs in arbitrary order. The 128 queries are global positions 384 through
511. A query may attend only to tokens in its listed logical pages whose
logical token position is at most `384+qi`. Query head `hq` maps to KV head
`hq/4`.

The supplied `output` and `lse` are the result of this sparse causal forward:

```text
score(qh,t) = dot(q[qh], k[t]) / sqrt(128)
p(qh,t)     = exp(score(qh,t) - lse[qh])
output[qh]  = sum_t p(qh,t) * v[t]
```

For `grad_output=dL/doutput`, compute:

```text
delta[qh]       = dot(grad_output[qh], output[qh])
dP[qh,t]        = dot(grad_output[qh], v[t])
dS[qh,t]        = p[qh,t] * (dP[qh,t] - delta[qh])
grad_q[qh,d]    = scale * sum_t dS[qh,t] * k[t,d]
grad_k[t,d]    += scale * dS[qh,t] * q[qh,d]
grad_v[t,d]    += p[qh,t] * grad_output[qh,d]
```

Only selected causal tokens contribute. Gradients for unreferenced physical
cache locations must be exactly zero. Use at least FP32 accumulation, preserve
all inputs, and do not assume logical and physical page order match.

## Fixed launch ABI

The verifier always launches and synchronizes `zero` before `main`:

```text
zero: grid=(3072,1,1), block=(128,1,1), dynamic shared memory=0
main: grid=(1024,1,1), block=(128,1,1), dynamic shared memory=0
```

For `zero`, the 393216 FP32 output elements form the concatenation
`grad_q || grad_k_cache || grad_v_cache`; block/thread linear index owns one
element. For `main`, block `x` owns `(query=x/8, query_head=x%8)`. Threads may
cooperate arbitrarily. Static shared memory and FP32 global atomics are allowed.

## Scoring

Hidden cases vary page permutations, CSR rows, values, and dominant logits.
The verifier checks both entries, all three gradients, zero gradients on
unselected pages, input immutability, guards, module loading, and CUDA errors.
Finite outputs must satisfy `abs(got-want) <= 3e-2 + 5e-2*abs(want)`.

Correctness is the only score in this upper-probe version: all cases pass for
reward 1, otherwise reward 0. Test a complete implementation early and iterate
within the 7200-second budget.
