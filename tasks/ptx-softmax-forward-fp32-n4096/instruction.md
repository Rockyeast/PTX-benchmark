# Implement FP32 row-wise Softmax in direct PTX

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. Only this file is
graded.

Write or edit the PTX directly. Do not generate the submission by compiling
CUDA C++, Triton, Python, or another higher-level language. `ptxas` and
`cuobjdump` may be used to validate and inspect PTX, but `nvcc`, NVRTC, and
NVVM are not available.

A task-pinned PTX ISA 9.2 source record is available under `/opt/ptx-docs`.
The development task does not redistribute the NVIDIA PDF and does not permit
downloading replacement documentation or tools during the task.

The submission must define a visible kernel entry named `softmax_fwd_fp32_n4096`.

## Logical contract

Use this logical signature and parameter order:

```text
softmax_fwd_fp32_n4096(
    const float* input,
    float* output,
    uint32_t num_rows)
```

This C-style declaration describes the logical ABI. The PTX submission must use
a valid `.entry` and `.param` declarations. Device pointers are passed as
64-bit values, and `num_rows` is passed as a 32-bit unsigned integer.

Both tensors are contiguous row-major arrays:

```text
input  : [num_rows, 4096], float32
output : [num_rows, 4096], float32
```

For each row `r`, compute a numerically stable softmax over all 4096 columns:

```text
x_j = input[r, j]
m   = max_j(x_j)
s   = sum_j(exp(x_j - m))

output[r, j] = exp(x_j - m) / s
```

All operations use at least FP32 precision. Inputs are finite; NaN and infinity
are not part of the contract. Approximate exponential instructions are allowed
if the final output meets the stated tolerance.

`input` and `output` do not alias. Their base addresses and row starts are at
least 16-byte aligned. The kernel must not modify `input`, and it must write
every output element without accessing memory outside either tensor.

## Fixed launch ABI

The verifier uses an ordinary CUDA Driver API launch with:

```text
block dimensions:         (1024, 1, 1)
grid dimensions:          (max(1, num_rows), 1, 1)
dynamic shared memory:    0 bytes
programmatic dependency:  disabled
```

Each block corresponds to one row. When `num_rows` is zero, one block is still
launched and the kernel must return before reading or writing any tensor. Static
PTX `.shared` storage may be used, but the verifier provides no dynamic shared
memory. Do not depend on Programmatic Dependent Launch attributes.

## Correctness and performance

Correctness is checked against an independent stable FP32 softmax. Finite,
non-negative outputs are compared elementwise with:

```text
atol = 2e-6
rtol = 2e-3
```

Each output row must also sum to `1` within absolute tolerance `2e-4`. PTX
assembly, module loading, entry lookup, launch, synchronization, numerical
correctness, input immutability, and memory-safety checks must all pass before
performance receives any credit.

Performance is initially scored at:

```text
num_rows = 1, 32, 256
```

Candidate kernel time is compared with a fixed same-ABI, same-launch Reference
on the same GPU and inputs. The score uses the geometric mean of per-size
Reference-time/Candidate-time ratios and is capped at `1`. Logical memory
bandwidth is also reported using the minimum interface traffic of 8 bytes per
element.

The target device is NVIDIA B200. Targeting `sm_100` is recommended.
Correctness must pass before any performance score is awarded.
