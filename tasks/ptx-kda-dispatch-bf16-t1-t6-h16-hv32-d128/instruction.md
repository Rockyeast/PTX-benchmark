# Implement a two-entry KDA dispatch module in direct PTX

Create one file, `/app/kernel.ptx`, containing NVIDIA PTX assembly. Only this
file is graded. The same PTX module must implement both visible entries below:

```text
kda_recurrent_decode_t1_h16_hv32_d128_n32
flash_kda_decode_sm100a_t6_split1
```

Write or edit PTX directly. Do not generate either entry by compiling CUDA C++,
Triton, Python, or another high-level kernel language. `ptxas` and `cuobjdump`
are available. You may inspect online PTX documentation, but the environment
does not provide or permit installing NVCC, NVRTC, NVVM, or Triton.

Both entries must coexist in one valid `sm_100a` module. Implementing only one
entry, changing an entry name, submitting two separate PTX files, or requiring
an extra launch is incorrect.

## Entry 1: fixed T1 recurrent decode

All data tensors are contiguous IEEE bfloat16. `scale` is f32.

```text
q, k       : [32, 1, 16, 128]
v          : [32, 1, 32, 128]
log_decay  : [32, 1, 32, 128]
beta       : [32, 1, 32]
state      : [32, 32, 128, 128], updated in place
output     : [32, 1, 32, 128]
scale      : 1/sqrt(128)
```

For sequence `n`, value head `hv`, and query/key head `qh = hv/2`:

```text
qn[dk]    = normalize(q[n,0,qh,:])[dk] * scale
kn[dk]    = normalize(k[n,0,qh,:])[dk]
decay[dk] = exp(f32(log_decay[n,0,hv,dk]))

decayed[dv,dk] = f32(state[n,hv,dv,dk]) * decay[dk]
prediction[dv] = sum_dk(decayed[dv,dk] * kn[dk])
residual[dv]   = f32(v[n,0,hv,dv]) - prediction[dv]
updated[dv,dk] = decayed[dv,dk]
                 + f32(beta[n,0,hv]) * residual[dv] * kn[dk]
state[n,hv,dv,dk] = bf16(updated[dv,dk])
output[n,0,hv,dv] = bf16(sum_dk(f32(state[n,hv,dv,dk]) * qn[dk]))
```

`normalize(x)[d] = f32(x[d]) / sqrt(sum_j(f32(x[j])^2) + 1e-6)`. The output
must read the newly updated state after BF16 rounding.

The exact parameters and launch are:

```text
.param .u64 q
.param .u64 k
.param .u64 v
.param .u64 log_decay
.param .u64 beta
.param .u64 state
.param .u64 output
.param .f32 scale

grid=(256,32,1), block=(32,1,1), dynamic shared memory=0
```

## Entry 2: fixed T6 speculative decode

```text
T=6, N=8, H=16, HV=32, D=128, state slots=54
q, k                 : [48, 16, 128], bf16
v, log_decay         : [48, 32, 128], bf16
beta                  : [48, 32], bf16
state                 : [54, 32, 128, 128], bf16, input/output pool
output                : [48, 32, 128], bf16
ssm_state_indices     : [8, 6], int32
num_accepted_tokens   : [8], int32
scale                 : 1/sqrt(128), f32
```

For sequence `n`, select the initial checkpoint:

```text
accepted_index = clamp(num_accepted_tokens[n] - 1, 0, 5)
initial_slot   = ssm_state_indices[n, accepted_index]
work_state    = float32(state[initial_slot, :, :, :])
```

Then process `t=0..5` sequentially with packed index `p=6*n+t`. Apply the same
normalized KDA recurrence as Entry 1, but after each token:

```text
rounded[dv,dk] = bf16(updated[dv,dk])
slot = ssm_state_indices[n,t]
state[slot,hv,dv,dk] = rounded[dv,dk]
work_state[hv,dv,dk] = f32(rounded[dv,dk])
output[p,hv,dv] = bf16(sum_dk(work_state[hv,dv,dk] * qn[dk]))
```

Every token writes a complete checkpoint, and the next token consumes the
BF16-rounded state. The exact parameters and launch are:

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

grid=(32,8,1), block=(256,1,1), dynamic shared memory=50,560 bytes
```

## Correctness and reward

The verifier independently tests both entries against CPU truth, including the
complete updated state, all outputs, read-only inputs, metadata permutations,
BF16 rounding, and allocation guards. The initial elementwise tolerance is:

```text
abs(got - expected) <= 1e-2 + 1e-2 * abs(expected)
```

Correctness is a hard gate across both entries. If either entry fails, the task
reward is zero. When both pass, each entry is timed against its own fixed
same-ABI Reference on one B200. Let `r1` and `r6` be the uncapped T1 and T6
speedups. The reported raw speedup is `sqrt(r1*r6)` and the final performance
reward is `min(1, sqrt(r1*r6))`.

Use the time budget to assemble, test, and improve both paths. Optimizing only
one entry while leaving the other missing or incorrect cannot receive credit.
