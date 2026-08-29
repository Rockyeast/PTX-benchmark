# Implement the complete seven-route B200 KDA decode family in direct PTX

Create one file, `/app/kernel.ptx`, containing NVIDIA PTX assembly. Only this
file is graded. The same module must define all seven visible entries:

```text
flash_kda_decode_sm100a_t1_split16
flash_kda_decode_sm100a_t1_split8
flash_kda_decode_sm100a_t2_split4
flash_kda_decode_sm100a_t3_split4
flash_kda_decode_sm100a_t4_split2
flash_kda_decode_sm100a_t5_split1
flash_kda_decode_sm100a_t6_split1
```

Write PTX directly. Do not compile CUDA C++, Triton, Python, or another
higher-level language into the submission. `ptxas` and `cuobjdump` are
available, and online NVIDIA PTX documentation is permitted. NVCC, NVRTC,
NVVM, Triton, and installation of another compiler are prohibited.

The module implements the complete native `D=128, H=16` recurrent-decode
dispatch family below. Missing one entry, changing an ABI, or requiring an
extra kernel launch makes the whole task incorrect.

## Supported shape matrix

```text
T=1, HV=32, N in {8,16,32,64,128}
T=2, HV=32, N in {8,16,32,64,128}
T=3, HV=16, N in {1,2,4,8,16}
T=4, HV=32, N in {8,16,32,64,128}
T=5, HV=32, N in {8,16,32,64,128}
T=6, HV=32, N in {8,16,32,64,128}
```

For T=1, the verifier selects `split16` when `N<=16` and `split8` otherwise.
For T=2..6, the entry is selected by T. N is encoded by `gridDim.y`, not by a
kernel parameter.

## Common recurrence

All tensor data except the T3 gate parameters is contiguous IEEE bfloat16.
`D=128`, `H=16`, and `scale=1/sqrt(128)`. Let `HV=16` for T3 and `HV=32` for
all other routes. For value head `hv`, use query/key head
`qh = hv / (HV/H)`.

For one token:

```text
qn = normalize(q[qh,:]) * scale
kn = normalize(k[qh,:])
decay[dk] = exp(log_decay[hv,dk])

decayed[dv,dk] = f32(work_state[hv,dv,dk]) * decay[dk]
prediction[dv] = sum_dk(decayed[dv,dk] * kn[dk])
residual[dv] = f32(v[hv,dv]) - prediction[dv]
updated[dv,dk] = decayed[dv,dk]
                 + f32(beta[hv]) * residual[dv] * kn[dk]
rounded[dv,dk] = bf16(updated[dv,dk])
output[hv,dv] = bf16(sum_dk(f32(rounded[dv,dk]) * qn[dk]))
```

```text
normalize(x)[d] = f32(x[d]) / sqrt(sum_j(f32(x[j])^2) + 1e-6)
```

The BF16-rounded state, not the pre-round FP32 value, is consumed by the next
token and by the output projection.

For T1, `g` directly contains BF16 `log_decay`, and state has shape
`[N,HV,128,128]`. Each sequence updates its own state slot in place.

For T2/T4/T5/T6, tensors use packed token-major shapes:

```text
q,k          [N*T,H,128]
v,g          [N*T,HV,128]
beta         [N*T,HV]
state        [slots,HV,128,128]
output       [N*T,HV,128]
indices      [N,T] int32
accepted     [N] int32
```

For sequence `n`, clamp `accepted[n]-1` to `[0,T-1]`, load the initial state
from `indices[n,clamped]`, process all T tokens sequentially, and after token t
write the complete rounded checkpoint to `state[indices[n,t]]` and the output
to packed token `n*T+t`.

T3 uses the same checkpoint rules but has `HV=16`. Its `g` is a BF16
pre-activation and log decay must be computed in-kernel:

```text
gate_a = exp(f32(A_log[qh]))
bias = f32(dt_bias[qh*128 + dk])
log_decay[hv,dk] = lower_bound
                         * sigmoid(gate_a * (f32(g[hv,dk]) + bias))
```

`A_log[16]` and `dt_bias[16*128]` are FP32; `lower_bound=-5.0` in the graded
matrix.

## Exact entry ABIs and launches

T1 entries both use:

```text
(q, k, v, g, beta, state, output, scale)
seven .u64 pointers followed by one .f32

t1_split16: grid=(512,N,1), block=(32,1,1), dynamic shared=0
t1_split8:  grid=(256,N,1), block=(32,1,1), dynamic shared=0
```

T2/T4/T5/T6 entries use:

```text
(q, k, v, g, beta, state, output, indices, accepted, scale)
nine .u64 pointers followed by one .f32

T2: grid=(128,N,1), block=(64,1,1),  dynamic shared=14720
T4: grid=(64,N,1),  block=(128,1,1), dynamic shared=25856
T5: grid=(32,N,1),  block=(256,1,1), dynamic shared=49024
T6: grid=(32,N,1),  block=(256,1,1), dynamic shared=50560
```

T3 uses:

```text
(q, k, v, g, beta, A_log, dt_bias, state, output, indices, accepted,
 scale, lower_bound)
eleven .u64 pointers followed by two .f32

grid=(64,N,1), block=(96,1,1), dynamic shared=15872
```

## Correctness reward

The hidden verifier loads every entry and selects cases across all seven
routes. It checks independent CPU truth for complete outputs and checkpoint
state, read-only inputs, BF16 per-token rounding, accepted-token selection,
permuted checkpoint slots, T3 lower-bound gating, allocation guards, and CUDA
errors. Finite values use:

```text
abs(got-expected) <= 1e-2 + 1e-2*abs(expected)
```

This calibration task is correctness-only. All required routes and all hidden
cases must pass for reward `1`; any missing or incorrect route receives `0`.
There is no performance score and no benefit to omitting work or optimizing
only a subset of shapes. Assemble a complete module early and verify every
entry before finishing.
