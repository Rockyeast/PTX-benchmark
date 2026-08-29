# Implement T6 speculative KDA decode in direct PTX

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. Only this file is
graded. It must define the visible entry
`flash_kda_decode_sm100a_t6_split1`.

Write or edit PTX directly. Do not generate the submission by compiling CUDA
C++, Triton, Python, or another higher-level language. `ptxas` and `cuobjdump`
may be used to validate and inspect PTX; `nvcc`, NVRTC, NVVM, and Triton are not
available. A pinned PTX ISA source record is under `/opt/ptx-docs`.

## Fixed specialization

```text
T=6 speculative tokens per sequence
N=8 independent sequences
H=16 query/key heads
HV=32 value/state heads
D=128
state slots=54
dtype=IEEE bfloat16 except scale and intermediate arithmetic
target=NVIDIA B200 / sm_100a
```

The pointer parameters are 64-bit and appear in exactly this order:

```text
.param .u64 q
.param .u64 k
.param .u64 v
.param .u64 log_decay
.param .u64 beta
.param .u64 state
.param .u64 output
.param .u64 ssm_state_indices
.param .u64 num_accepted_tokens
.param .f32 scale
```

Contiguous row-major layouts are:

```text
q, k                 : [48, 16, 128], bf16
v, log_decay         : [48, 32, 128], bf16
beta                  : [48, 32],      bf16
state                 : [54, 32, 128, 128], bf16, input/output pool
output                : [48, 32, 128], bf16
ssm_state_indices     : [8, 6], int32
num_accepted_tokens   : [8],    int32
scale                 : 1/sqrt(128), f32
```

`q`, `k`, `v`, `log_decay`, `beta`, `ssm_state_indices`, and
`num_accepted_tokens` are read-only. `state` is an in-place checkpoint pool and
`output` must be fully written. The state indices are valid non-negative slots
and are unique within a call, but their order is not fixed.

## Speculative checkpoint semantics

For sequence `n`, first select the state from which all six speculative tokens
start:

```text
accepted_index = clamp(num_accepted_tokens[n] - 1, 0, 5)
initial_slot   = ssm_state_indices[n, accepted_index]
work_state    = float32(state[initial_slot, :, :, :])
```

Then process local tokens `t=0..5` in order. The packed token index is
`p = 6*n + t`. For each value head `hv`, use query/key head `qh = hv/2` and:

```text
qn[d]    = float(q[p,qh,d]) / sqrt(sum_j float(q[p,qh,j])^2 + 1e-6) * scale
kn[d]    = float(k[p,qh,d]) / sqrt(sum_j float(k[p,qh,j])^2 + 1e-6)
decay[d] = exp(float(log_decay[p,hv,d]))

decayed[dv,dk] = work_state[hv,dv,dk] * decay[dk]
prediction[dv] = sum_dk(decayed[dv,dk] * kn[dk])
residual[dv]   = float(v[p,hv,dv]) - prediction[dv]
updated[dv,dk] = decayed[dv,dk]
                 + float(beta[p,hv]) * residual[dv] * kn[dk]
rounded[dv,dk] = bf16(updated[dv,dk])

slot = ssm_state_indices[n,t]
state[slot,hv,dv,dk] = rounded[dv,dk]
work_state[hv,dv,dk] = float(rounded[dv,dk])
output[p,hv,dv] = bf16(sum_dk(work_state[hv,dv,dk] * qn[dk]))
```

Every token therefore writes a complete checkpoint, and token `t+1` consumes
the BF16-rounded work state produced by token `t`. Reading only slot 0,
ignoring `num_accepted_tokens`, writing only the final checkpoint, or delaying
BF16 rounding is incorrect.

## Fixed launch ABI

The verifier launches exactly:

```text
grid                  = (32, 8, 1)
block                 = (256, 1, 1)
dynamic shared memory = 50,560 bytes
```

The kernel may use the supplied dynamic shared memory and architecture-specific
PTX valid for `sm_100a`. It must not require a second entry or a second launch.

## Correctness and reward

Candidate and a fixed same-ABI Reference are independently checked against a
CPU implementation. The initial elementwise tolerance is:

```text
abs(got - expected) <= 1e-2 + 1e-2 * abs(expected)
```

The verifier checks the complete state pool, all outputs, read-only inputs, and
allocation guards. Correctness is a hard gate. Only a correct Candidate is
timed against the fixed Reference on the same B200. The uncapped speed ratio is
reported; the performance reward is `clamp(reference_time/candidate_time,0,1)`.

## Iterative optimization

Correctness is the first milestone, not the end of the task. Once a Candidate
assembles and launches, use `ptxas`, `cuobjdump`, and the public command below
to measure and improve it:

```bash
direct-ptx-benchmark-candidate /app/kernel.ptx
```

The command assembles and times only your Candidate on one deterministic public
development input. It reports absolute Candidate latency, timing dispersion,
and logical throughput. It does not contain or expose the fixed Reference,
hidden cases, expected outputs, or final reward, and a successful public run
does not prove hidden correctness. Keep the best Candidate that remains correct
under your own checks as `/app/kernel.ptx`. You may finish early when further
useful improvement is unlikely; you are not required to consume the full time
budget.
