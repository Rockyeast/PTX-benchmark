# Implement paged prefill BF16 attention in direct PTX

Create `/app/kernel.ptx` with visible entry
`paged_prefill_attn_bf16_q128_hq8_hkv2_d128_p16`. Write PTX directly; do not
compile CUDA C++, Triton, Python, or another high-level language into the
submission. `ptxas` and `cuobjdump` are allowed. `nvcc`, NVRTC, NVVM, and Triton
are unavailable. Online official PTX documentation is permitted.

Logical parameter order:

```text
(const bf16* q,
 const bf16* k_cache,
 const bf16* v_cache,
 const int32* page_table,
 bf16* output,
 uint32 context_length)
```

Device pointers are 64-bit; `context_length` is 32-bit.

## Fixed shapes and layout

```text
Q_LEN = 128, Hq = 8, Hkv = 2, D = 128, PAGE_SIZE = 16
q/output : [128, 8, 128], NHD BF16
k/v cache: [64 physical pages, 16 tokens, 2 KV heads, 128], BF16
page_table: [64], int32; logical page -> physical page
context_length: one of 128..1024, divisible by 16
```

The 128 queries are the final 128 tokens of the context. Query `qi` therefore
has global position `context_length - 128 + qi` and attends causally to logical
KV tokens `0..global_position`. Query head `hq` maps to KV head `hq/4`.

For each allowed logical key token, use `page_table[token/16]` and `token%16`
to locate K/V. Compute standard scaled dot-product attention with scale
`1/sqrt(128)`, stable Softmax, FP32 accumulation, and BF16 output. Preserve all
inputs, write every output element, and do not assume that physical pages are
ordered like logical pages.

## Fixed launch ABI

```text
grid dimensions:       (1024, 1, 1)
block dimensions:      (128, 1, 1)
dynamic shared memory: 0 bytes
```

Block `x` owns `query=x/8`, `query_head=x%8`; thread `x` may own output feature
`x`. Static shared storage is allowed.

## Scoring

Hidden cases vary `context_length`, page permutations, values, and grouped-query
head patterns. The verifier checks full BF16 output, input immutability, memory
guards, module loading, and CUDA errors with tolerance
`abs(got-want) <= 2e-2 + 2e-2*abs(want)`.

Correctness is a hard gate. Performance reward is
`min(1, geometric_mean(reference_time/candidate_time))` over fixed context
lengths on the same B200. Test a complete Candidate early, then optimize within
the 5400-second budget.
