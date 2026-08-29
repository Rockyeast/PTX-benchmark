# `ptx-kda-speculative-decode-bf16-t6-n8-h16-hv32-d128` 任务规格

**状态：B200 feasibility（2026-08-22）；Oracle 和定向负例已验证，真实 Agent
校准进行中，`release_ready=false`。**

## 1. 选题边界

这道题直接选择 INT21 KDA-B200 公开实现中语义最完整的固定 speculative decode
路径，而不是继续从 T1 逐级增加难度：

```text
T=6 speculative tokens per sequence
N=8 independent sequences
H=16 query/key heads
HV=32 value/state heads
D=128
BF16 input/state/output, FP32 recurrence arithmetic
NVIDIA B200 / sm_100a
upstream route: d128_t6_cached_gram_split1
```

固定来源：

```text
repository: https://github.com/Int21-AI/KDA-B200
commit:     4896cd2da4ca421829907ab9fd3550e3f84a141b
source:     csrc/decode_sm100a/d128_t6_cached_gram_split1.cu
license:    MIT
```

T6 同时引入六步顺序 recurrence、六个 checkpoint、accepted-token 选择、非连续
state slot metadata、GQA 2:1 head mapping、BF16 逐步舍入和 50,560-byte 动态共享
内存。N=8 已让 hidden cases 覆盖八类 metadata/数值模式；继续增加 N 主要增加显存和
重复 block，不增加 Agent 需要实现的代码语义。

## 2. Agent 交付和固定 ABI

Agent 只创建 `/app/kernel.ptx`，entry 必须是：

```text
flash_kda_decode_sm100a_t6_split1(
    bf16* q, bf16* k, bf16* v, bf16* log_decay, bf16* beta,
    bf16* state, bf16* output,
    int32* ssm_state_indices, int32* num_accepted_tokens,
    float scale)
```

参数在 PTX 中依次为九个 `.u64` 和一个 `.f32`。固定 contiguous shape：

```text
q, k:                  [48,16,128]
v, log_decay, output: [48,32,128]
beta:                  [48,32]
state:                 [54,32,128,128]  (in-place checkpoint pool)
ssm_state_indices:     [8,6]
num_accepted_tokens:   [8]
scale:                 1/sqrt(128)
```

`q/k/v/log_decay/beta/indices/accepted` 只读；`state` 原地写入六个 checkpoint，
`output` 必须完整写入。

## 3. 数学语义

对 sequence `n`，先计算：

```text
a = clamp(num_accepted_tokens[n] - 1, 0, 5)
initial_slot = ssm_state_indices[n,a]
S = bf16 state[initial_slot]
```

随后按 token `t=0..5` 严格顺序执行。对每个 `hv`，query head 为 `qh=hv/2`；Q/K
先作 L2 normalization，gate 为 `exp(log_decay)`。每一步执行 state decay、KDA
prediction、`beta * residual * K` rank-1 update，把更新结果**先舍入为 BF16**，再：

1. 写入 `state[ssm_state_indices[n,t],hv,:,:]`；
2. 用这份已舍入 state 与 normalized Q 计算 `output[n,t,hv,:]`。

后一步 recurrence 使用前一步已舍入 state。只写最终 checkpoint、忽略 accepted
metadata、把 `hv` 映射成错误 query head，均不符合本题。

## 4. 固定 launch

```text
grid:                  (32,8,1)
block:                 (256,1,1)
dynamic shared memory: 50560 bytes
```

launch 由 verifier 固定，不使用 `launch.json`。Candidate PTX 与 Reference PTX 都先由
固定 CUDA 13.2.1 `ptxas -arch=sm_100a` 汇编为 cubin，再通过 CUDA Driver API 加载；
这避免驱动 PTX JIT 与正式离线汇编路径不一致。

## 5. Correctness、Reference 与 reward

`tests/kda_t6_truth.c` 独立重写 recurrence，不调用 Agent、Oracle 或 INT21 GPU
kernel。八个 hidden cases 覆盖随机 accepted、checkpoint permutation、zero-state、
zero-beta、head mapping、decay extremes、逐步 BF16 rounding 和 accepted boundaries。
每个 case 比较完整 state pool 与 output，并检查只读输入和 allocation guards。

逐元素门槛：

```text
abs(got-expected) <= 1e-2 + 1e-2*abs(expected)
```

Candidate 和固定 Reference 都必须独立通过 CPU truth。任一 correctness 失败：

```text
correctness = 0
reward = 0
```

通过后，在四组固定输入上交替测 Candidate/Reference：

```text
r_i = reference_median_ms_i / candidate_median_ms_i
speedup = geometric_mean(r_i)
performance = clamp(speedup, 0, 1)
reward = correctness * performance
```

Reference/Oracle 是固定上游 revision 编译出的同 ABI PTX，只负责 feasibility 和性能
锚点，不是 correctness 真值。正式性能排名还需要评估 Reference 强度和计时方差。

## 6. 发布边界

- [x] 固定来源、revision、ABI、shape、launch 和公式；
- [x] 独立 CPU truth、8 cases、内存安全检查；
- [x] Reference/Oracle 与三个可汇编语义错误 fixture；
- [x] B200 直接 verifier 与 Harbor Oracle；
- [ ] 多次强模型与开放权重模型校准；
- [ ] 用 3600 秒候选预算完成 Oracle 和真实 Agent 校准后，冻结正式时间预算；
- [ ] 通过 release checker 并进入冻结 manifest。

完成前保持 `release_ready=false`；一次 Agent 结果只能说明 feasibility，不能冻结通过率。
