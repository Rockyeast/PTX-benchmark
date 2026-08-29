# Implement BF16 SiLU-gate forward in direct PTX

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. Only this file is
graded.

Write or edit the PTX directly. Do not generate the submission by compiling
CUDA C++, Triton, Python, or another higher-level language. `ptxas` and
`cuobjdump` may be used to validate and inspect PTX, but `nvcc`, NVRTC, and
NVVM are not available.

A task-pinned PTX ISA 9.2 source record is available under `/opt/ptx-docs`.
The development environment does not redistribute the full NVIDIA PDF and does
not permit downloading replacement documentation or tools during the task.

The submission must define a visible kernel entry named `silu_gate_fwd_bf16_n4096`.

## Logical contract

Use this logical signature and parameter order:

```text
silu_gate_fwd_bf16_n4096(
    const bfloat16* input,
    bfloat16* output,
    uint32_t num_tokens)
```

This C-style declaration describes the logical ABI. The PTX submission must use
a valid `.entry` and `.param` declarations. Device pointers are passed as
64-bit values, and `num_tokens` is passed as a 32-bit unsigned integer.

Both tensors are contiguous row-major arrays:

```text
input  : [num_tokens, 8192], bfloat16
output : [num_tokens, 4096], bfloat16
```

Each input row contains two contiguous halves:

```text
input[t, 0:4096]    = gate
input[t, 4096:8192] = up
```

For every token `t` and element `j` in `[0, 4095]`, compute:

```text
g = float32(input[t, j])
u = float32(input[t, 4096 + j])

silu(g) = g / (1 + exp(-g))
        = g * sigmoid(g)

output[t, j] = bfloat16(silu(g) * u)
```

Compute intermediates with at least float32 precision and round the final result
to BF16 using round-to-nearest-even semantics. PTX approximate exponential
instructions are allowed if the final result meets the stated tolerance.

`input` and `output` do not alias. Their base addresses and row starts are at
least 16-byte aligned. The kernel must not modify `input`, and it must write
every output element without accessing memory outside either tensor.

## Fixed launch ABI

The verifier uses an ordinary CUDA Driver API launch with:

```text
block dimensions:         (512, 1, 1)
grid dimensions:          (max(1, num_tokens), 1, 1)
dynamic shared memory:    0 bytes
programmatic dependency:  disabled
```

Each block corresponds to one token. When `num_tokens` is zero, one block is
still launched and the kernel must return before reading or writing any tensor.
Do not depend on Programmatic Dependent Launch attributes.

## Correctness and performance

Correctness is checked against an independent CPU implementation of the formula
above after BF16 rounding. Finite outputs are compared with:

```text
atol = 3e-2
rtol = 1e-2
```

NaN or infinity in the output is rejected. PTX assembly, module loading, entry
lookup, launch, synchronization, numerical correctness, input immutability, and
memory-safety checks must all pass before performance receives any credit.

Performance is initially scored at:

```text
num_tokens = 32, 256, 4096
```

`num_tokens=1` latency is measured as a diagnostic but is not included in the
aggregate score. Candidate kernel time is compared with a fixed same-ABI,
same-launch Reference on the same GPU and inputs. The score uses the geometric
mean of per-size Reference-time/Candidate-time ratios and is capped at `1`.
Effective memory bandwidth is also reported using the minimum explicit traffic
of 6 bytes per output element.

The target device is NVIDIA L40. Targeting `sm_89` is recommended.
Correctness must pass before any performance score is awarded.
