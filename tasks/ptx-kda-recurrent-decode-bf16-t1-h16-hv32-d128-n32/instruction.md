# Implement fixed-shape KDA recurrent decode in direct PTX

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. Only this file is
graded.

Write or edit the PTX directly. Do not generate the submission by compiling
CUDA C++, Triton, Python, or another higher-level language. `ptxas` and
`cuobjdump` may be used to validate and inspect PTX, but `nvcc`, NVRTC, and
NVVM are not available.

A task-pinned PTX ISA reference is available under `/opt/ptx-docs`. Use the
local snapshot rather than downloading replacement documentation or tools.

The submission must define a visible kernel entry named
`kda_recurrent_decode_t1_h16_hv32_d128_n32`.

## Fixed tensors and layout

All tensors are contiguous with the rightmost dimension contiguous. Every
element is IEEE bfloat16 unless noted otherwise.

```text
q, k       : [32, 1, 16, 128]
v          : [32, 1, 32, 128]
log_decay  : [32, 1, 32, 128]
beta       : [32, 1, 32]
state      : [32, 32, 128, 128], updated in place
output     : [32, 1, 32, 128]
scale      : scalar f32, fixed by the verifier to 1/sqrt(128)
```

`q`, `k`, `v`, `log_decay`, and `beta` are read-only and do not alias any
other allocation. `state` is the only in-place input/output. The entire
`output` allocation must be written.

For example, the linear state index is:

```text
(((n * 32 + hv) * 128 + dv) * 128 + dk)
```

## Operation

For each sequence `n` and value head `hv`, use query/key head `qh = hv / 2`.
All normalization, exponential, products, reductions, and state-update
arithmetic are evaluated in at least f32 before the explicitly stated bf16
rounding.

```text
eps = 1e-6

normalize(x)[d] =
    f32(x[d]) / sqrt(sum_j(f32(x[j]) * f32(x[j])) + eps)

qn[dk]    = normalize(q[n,0,qh,:])[dk] * scale
kn[dk]    = normalize(k[n,0,qh,:])[dk]
decay[dk] = exp(f32(log_decay[n,0,hv,dk]))

decayed[dv,dk] = f32(state[n,hv,dv,dk]) * decay[dk]

prediction[dv] = sum_dk(decayed[dv,dk] * kn[dk])
residual[dv]   = f32(v[n,0,hv,dv]) - prediction[dv]

updated[dv,dk] =
    decayed[dv,dk]
    + f32(beta[n,0,hv]) * residual[dv] * kn[dk]

state[n,hv,dv,dk] = bf16(updated[dv,dk])

output[n,0,hv,dv] =
    bf16(sum_dk(f32(state[n,hv,dv,dk]) * qn[dk]))
```

The output projection must read the newly updated state **after that state has
been rounded to bf16**. `log_decay` is already a log-space decay. Do not apply
sigmoid, softplus, an A-log transform, or a lower-bound gate.

## Fixed parameter and launch ABI

The visible entry has these PTX parameters in this exact order:

```text
.param .u64 q
.param .u64 k
.param .u64 v
.param .u64 log_decay
.param .u64 beta
.param .u64 state
.param .u64 output
.param .f32 scale
```

The verifier always launches the entry as:

```text
grid                    = (256, 32, 1)
block                   = (32, 1, 1)
dynamic shared memory   = 0 bytes
programmatic dependency = disabled
```

There is no runtime `N` parameter and no `launch.json`. Static shared memory is
allowed.

## Correctness and performance

The verifier compares both the updated state and output against an independent
implementation of the formula above, checks read-only inputs and guarded
allocations, and rejects module, launch, synchronization, or memory errors.
The initial elementwise tolerance is:

```text
abs(got - expected) <= 1e-2 + 1e-2 * abs(expected)
```

Correctness is a hard gate. Only a correct submission is timed against a fixed
same-contract Reference on the same NVIDIA B200. The final performance score
is capped at 1.0; an incorrect submission receives reward 0 regardless of
speed.

The target is NVIDIA B200 with the SM100a architectural boundary. The PTX must
load and execute on that target; architecture-specific PTX may be used when it
is valid for the pinned toolchain.
