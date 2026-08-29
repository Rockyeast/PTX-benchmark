# Implement block-sparse BF16 attention forward in direct PTX

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. Only this file is
graded. Write PTX directly; do not compile CUDA C++, Triton, Python, or another
higher-level language into the submission. `ptxas` and `cuobjdump` are allowed;
`nvcc`, NVRTC, NVVM, and Triton are unavailable. Online official PTX
documentation is permitted.

Define the visible entry `block_sparse_attn_fwd_bf16_s1024_d128_b16`:

```text
(const bf16* q,
 const bf16* k,
 const bf16* v,
 const int32* row_offsets,
 const int32* block_indices,
 bf16* output)
```

All pointers are 64-bit kernel parameters.

## Fixed tensors and sparse layout

```text
q, k, v, output : [1024, 128], row-major BF16
row_offsets      : [65], int32
block_indices    : [row_offsets[64]], int32
query block size : 16 tokens
key block size   : 16 tokens
scale            : 1 / sqrt(128)
```

There are 64 query blocks. For query token `i`, let `r=i/16`. Its permitted key
tokens are every token in the key blocks listed by
`block_indices[row_offsets[r]:row_offsets[r+1]]`. Each row contains 1 through 8
unique block IDs in `[0,63]`; their order is arbitrary. The complete index list
contains at most 512 entries.

For each query token, perform scaled dot-product attention only over those
selected keys:

```text
score[j] = sum_d(float(q[i,d]) * float(k[j,d])) / sqrt(128)
p        = softmax(score over selected j only)
output[i,d] = sum_selected_j p[j] * float(v[j,d])
```

Use at least FP32 accumulation. Write every output element, preserve all inputs,
and do not read outside the described arrays. Different query rows may use
different sparse block lists; do not assume causal, sorted, or identical rows.

## Fixed launch ABI

```text
grid dimensions:       (1024, 1, 1)
block dimensions:      (128, 1, 1)
dynamic shared memory: 0 bytes
```

Block `x` owns query token `x`. Static shared storage is allowed.

## Scoring

Hidden correctness cases vary the CSR block layout and tensor values and check
all output values, input immutability, allocation guards, and CUDA errors. BF16
outputs must satisfy `abs(got-want) <= 2e-2 + 2e-2*abs(want)`.

Correctness is a hard gate. A correct Candidate receives continuous performance
reward `min(1, reference_time / candidate_time)` against fixed same-ABI PTX on
the same B200. Test a complete Candidate early, then use the remaining
4800-second budget for PTX-level optimization.
