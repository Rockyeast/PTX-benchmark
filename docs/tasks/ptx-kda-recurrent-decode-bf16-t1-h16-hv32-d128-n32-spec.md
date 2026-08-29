# `ptx-kda-recurrent-decode-bf16-t1-h16-hv32-d128-n32` 任务规格

**状态：Agent-feasibility draft v0.1（2026-08-22）。本题只抽取官方 KDA-B200 的一个
固定 recurrent decode 路径；Oracle、负例和一次 Codex/high B200 feasibility 已完成，
尚未完成多模型校准或 release。**

## 1. 为什么先做这个固定版本

INT21 的 KDA-B200 recurrent decode 覆盖 30 个 shape，并由七条 GPU kernel 路径和
host dispatch 共同完成。把整个系统塞进一道 Harbor single-entry task，会同时改变
token 数、sequence 数、value-head 数、gate 语义、checkpoint metadata 和 launch，无法
判断 Agent 究竟在哪一项失败。

第一题只固定官方标准 decode 中的一条真实路径：

```text
T=1 token per sequence
N=32 independent sequences
H=16 query/key heads
HV=32 value/state heads
D=128 key/value dimension
BF16 inputs, state and output
precomputed log-decay gate
NVIDIA B200 / sm_100a
official route: d128_t1_cached_direct_split8
```

它仍然包含 Q/K normalization、GQA-style 2:1 head mapping、32 MiB recurrent state 的
衰减、预测、rank-1 更新、BF16 state rounding 和输出投影，不是 add-one/Softmax 一类
基础题。其他 T/N/gate 路径以后拆成独立 task 或同 ABI case，不能暗中混入本题。

## 2. 上游依据与 provenance

固定来源：

```text
repository: https://github.com/Int21-AI/KDA-B200
commit:     4896cd2da4ca421829907ab9fd3550e3f84a141b
license:    MIT
```

本题语义和路径依据：

- `tests/torch_decode_ref.py`：独立 PyTorch recurrence、BF16 state update 和 head mapping；
- `csrc/flash_kda_decode_sm100a.cpp`：`N > 16` 选择 T1 split8；
- `csrc/decode_sm100a/d128_t1_cached_direct_split8.cu`：参数顺序以及
  `grid=(256,N,1), block=(32,1,1)`；
- `BENCHMARK_GB200.md`：B200/CUDA 13.2 环境、30-shape CUPTI 方法和外部性能数据。

官方在 `T=1,N=32` 报告 `15.32483 us`，公开 FlashInfer 对照为 `16.096 us`，即
`1.050322x`。这些是 GB200 上的 CUPTI kernel-activity 外部证据，不直接充当本地
verifier 的时间，也不决定 correctness。

## 3. Agent 交付与 ABI

Agent 只创建 `/app/kernel.ptx`，其中包含 visible entry：

```text
kda_recurrent_decode_t1_h16_hv32_d128_n32(
    const bf16* q,
    const bf16* k,
    const bf16* v,
    const bf16* log_decay,
    const bf16* beta,
    bf16* state,
    bf16* output,
    float scale)
```

PTX parameter 顺序与类型固定为：

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

固定 tensor shape 和 contiguous row-major layout：

```text
q, k       : [32, 1, 16, 128], bf16
v          : [32, 1, 32, 128], bf16
log_decay  : [32, 1, 32, 128], bf16
beta       : [32, 1, 32],      bf16
state      : [32, 32, 128, 128], bf16, input/output in-place
output     : [32, 1, 32, 128], bf16
scale      : FP32, verifier 固定为 1/sqrt(128)
```

`q/k/v/log_decay/beta` 只读且不得别名；`output` 必须完整写入；`state` 是唯一允许原地
更新的张量。所有地址 64-bit，BF16 使用 IEEE bfloat16 位模式。

线性索引以最右维连续。例如：

```text
state[n,hv,dv,dk] -> (((n*32 + hv)*128 + dv)*128 + dk)
```

## 4. 数学语义

令：

```text
qh = hv / 2
eps = 1e-6
normalize(x) = float(x) / sqrt(sum_d(float(x[d])^2) + eps)
qn[d] = normalize(q[n,0,qh,:])[d] * scale
kn[d] = normalize(k[n,0,qh,:])[d]
decay[d] = exp(float(log_decay[n,0,hv,d]))
```

对每个 `n in [0,32)`、`hv in [0,32)`：

```text
decayed[dv,dk] = float(state[n,hv,dv,dk]) * decay[dk]

prediction[dv] = sum_dk(decayed[dv,dk] * kn[dk])
residual[dv]   = float(v[n,0,hv,dv]) - prediction[dv]

updated_fp32[dv,dk] =
    decayed[dv,dk]
    + float(beta[n,0,hv]) * residual[dv] * kn[dk]

state[n,hv,dv,dk] = bf16(updated_fp32[dv,dk])

output[n,0,hv,dv] =
    bf16(sum_dk(float(state[n,hv,dv,dk]) * qn[dk]))
```

关键点是 output 使用**已经舍入为 BF16 的新 state**；如果只在最后统一舍入，结果不符合
本题。`log_decay` 已经是 log-space 衰减，不再做 sigmoid/softplus，也不读取 A_log、
dt_bias 或 lower-bound gate 参数。

## 5. 固定 launch

为保持与官方 `d128_t1_cached_direct_split8` 路径一致：

```text
grid:                    (256,32,1)
block:                   (32,1,1)
dynamic shared memory:   0
programmatic dependency: disabled
```

launch 由 verifier 固定，Agent 不提交 `launch.json`。允许静态 shared memory。Task 固定
N=32，因此 ABI 中没有 runtime `N`；改变 N 并触发 split16/split8 dispatch 时应另行定义
case 或 task。

## 6. Correctness 设计

CPU truth 独立重写上述公式，不调用 INT21 kernel，也不比较 PTX 文本。Candidate 和固定
Reference 分别与 CPU truth 比较；Reference 自己也必须先通过 truth，避免错误 baseline
污染成绩。

初始 hidden cases：

| case | 目的 |
|---|---|
| random-standard | 正常随机输入、负 log-decay 和非零 state |
| zero-state | 分离 rank-1 update 与 output 投影 |
| zero-beta | 只验证 decay + state projection |
| structured-head-map | 捕获错误的 `hv -> qh=hv/2` 映射 |
| decay-extremes | 覆盖接近 0 和明显负的 log-decay |
| state-rounding | 放大“更新后先 BF16 舍入”语义差异 |

初始逐元素门槛沿用上游 reference test 的量级：

```text
abs(got-want) <= 1e-2 + 1e-2*abs(want)
```

同时记录 output/state 的 max-abs、max-rel 和 mismatch 数；校准后才能决定是否收紧。
verifier 还检查只读输入未修改、state/output allocation guards、完整 output 写入、合法
module/entry、launch/sync 和 CUDA error。任一 correctness case 失败时 reward 为 0。

## 7. Reference、Oracle 与隔离

- correctness 真值：task-owned CPU/FP32 公式；
- 性能 Reference：task-owned CUDA 13.2.1 实现，保持上述固定 revision 的 split8
  参数语义与 launch，但不是 INT21 官方二进制；源码、编译参数和哈希均保留；
- Oracle：可以先使用已验证的作者侧 PTX 作为可行性答案，但它不是 correctness 真值；
- Agent 只能看到 instruction、离线 PTX ISA 文档、`ptxas` 和 `cuobjdump`；CPU truth、
  Reference、Oracle 及 hidden cases 均不放进 `/app`。

若 Reference 无法做到相同参数语义、launch 和同步边界，本题只能报告 correctness，不能
发布性能排名。

## 8. Performance 与 reward

因为 shape 固定，性能 case 只更换四组合法输入，不改变 ABI 或 launch。Reference 与
Candidate 在同一 GPU/context/stream 中交替运行；每组先 warmup，再取多批次 median，
并在两者之间使用相同 cache-control 策略。

```text
r_i = reference_median_time_i / candidate_median_time_i
speedup = geometric_mean(r_0, r_1, r_2, r_3)
performance = clamp(speedup, 0, 1)
reward = correctness * performance
```

另外报告原始微秒、speedup、state bytes 和逻辑 FLOPs。正式 verifier 是否采用 CUDA
event 或 CUPTI 必须固定并在 validation 文档声明；不同计时方法不能和官方 15.32483 us
直接混算。

## 9. 难度和验收门槛

暂标为 **E / D5 candidate**，这里只表示结构与状态更新明显复杂于现有 P9/P10，不代表
已经得到 30%～50% Agent 通过率。

- [x] 固定上游 revision、许可证、公式、ABI、shape 和 launch；
- [x] 生成并静态检查 fail-closed scaffold；
- [x] 实现独立 CPU truth 与 hidden verifier；
- [x] 生成同 ABI Reference 和 Oracle，记录 provenance/hash；
- [x] 加入至少三个有针对性的错误 fixture 和一个慢正确 fixture；
- [x] B200 Oracle、负例、重复性能验证；
- [x] Codex/high 单次 feasibility（artifact 通过，Agent timeout）；
- [ ] 开放权重模型、严格网络隔离和预算校准。

完成前保持 `release_ready=false`，不加入冻结数据集。
