# `ptx-kda-recurrent-decode-bf16-t1-h16-hv32-d128-n32` 验证记录

**状态：B200 Oracle、负例和一次 Codex/high Agent feasibility 已完成；
`release_ready=false`。日期：2026-08-22。**

## 1. 固定任务

```text
operation:       KDA standard recurrent decode
shape:           T=1, N=32, H=16, HV=32, D=128
dtype:           BF16 input/state/output; FP32 recurrence arithmetic
state:           [32,32,128,128], in-place
entry:           kda_recurrent_decode_t1_h16_hv32_d128_n32
launch:          grid=(256,32,1), block=(32,1,1), dynamic smem=0
hardware:        NVIDIA B200 / sm_100a boundary
setting:         Direct-PTX
Agent budget:    600 seconds
```

语义和 launch 来源固定为 INT21 KDA-B200：

```text
repository: https://github.com/Int21-AI/KDA-B200
commit:     4896cd2da4ca421829907ab9fd3550e3f84a141b
license:    MIT
route:      d128_t1_cached_direct_split8
```

## 2. Reference 和 correctness 独立性

`tests/kda_truth.c` 独立生成并计算六类输入：

```text
random-standard
zero-state
zero-beta
structured-head-map
decay-extremes
state-rounding
```

它不调用 Agent、Oracle、INT21 GPU kernel 或性能 Reference。每个 case 都检查 32 MiB
更新后 state、完整 output、只读输入不变和 allocation guards。

性能 Reference 是 task-owned CUDA 代码，不是 correctness 真值：

```text
compiler image: CUDA 13.2.1 / nvcc 13.2
command: nvcc -ptx -O3 --use_fast_math -arch=compute_100

reference.cu:  9b4175a6a8a2d658b0171cd00278b372db3d888a4e408c4cf4731ae109fd1266
reference.ptx: 968aa30820f2ec8d52cd93dd0cb2c11644b6640c7c036a2192c30d8f5f28511f
oracle.ptx:    968aa30820f2ec8d52cd93dd0cb2c11644b6640c7c036a2192c30d8f5f28511f
```

## 3. 直接 B200 verifier 和负例

在 `catalyst-b200` GPU 1、Direct-PTX CUDA 13.2.1 SIF 中运行。Oracle 通过全部六个
case：

```text
correctness:          1
reward:               1
raw speedup:          1.000368x
Reference latency:    about 22.83 us
timing dispersion:    below 0.2%
```

针对性 fixtures：

| fixture | correctness | reward | 结果 |
|---|---:|---:|---|
| wrong head mapping | 0 | 0 | 被 state comparison 拒绝 |
| ignore beta | 0 | 0 | 被 state comparison 拒绝 |
| output before update | 0 | 0 | 被 output comparison 拒绝 |
| output before BF16 rounding | 1 | 1 | 误差仍在公开 `1e-2` 容差内，只保留为诊断，不算负例 |
| slow correct | 1 | 0.072128 | 正确性通过并得到连续性能分 |

“output before BF16 rounding”没有被拒绝，不能声称 verifier 能在当前容差下区分该实现；
正式冻结前应结合更多模型误差决定是收紧门槛还是接受数值等价范围。

## 4. Harbor Oracle

最终 Codex SIF 对照 Job：

```text
jobs/ptx-kda-b200-oracle-codex-image-r2-20260822/
trial:         ptx-kda-recurrent-decode-bf16-t1__SyGoGkv
task checksum: 86e0366f951894b20b3665d784612fbe0ac771524c9194a722d639e620293f40
correctness:   1
reward:        1
raw speedup:   1.000448926660286x
runtime:       about 20 seconds
```

它与下一节真实 Agent 使用相同 task checksum 和 Codex SIF。

早期还完成一个 Oracle feasibility Job和三个并行请求。三个并行请求中两个成功、一个
因共享服务器当时没有第三张可租用健康 GPU 而得到 `GpuAllocationError`；这是资源分配
异常，不是 kernel 或 verifier correctness 失败。连同首次 Job，共取得三个成功的
Reference-vs-Reference 样本：

```text
1.0005337411x
1.0003344961x
0.9997043033x
```

## 5. 真实 Agent feasibility

固定配置：

```text
Agent:              Codex CLI 0.148.0
model:              gpt-5.6-sol
reasoning_effort:   high
Agent timeout:      600 seconds
hardware:           NVIDIA B200, allocated physical GPU 1
network:            allow only chatgpt.com and *.chatgpt.com
task checksum:      86e0366f951894b20b3665d784612fbe0ac771524c9194a722d639e620293f40
```

Job：

```text
jobs/ptx-kda-b200-codex-gpt56-high-feasibility-r2-20260822/
trial: ptx-kda-recurrent-decode-bf16-t1__cbqvwVK
```

必须同时报告以下两项：

1. Agent 在 600 秒边界被 Harbor 取消，`AgentTimeoutError`，没有正常自然语言退出；
2. 截止时 `/app/kernel.ptx` 已存在，隐藏 verifier 判定全部六个 correctness case
   通过，`correctness=1`、`reward=1`。

最终结果：

```text
artifact SHA-256:       6618e65a4191f1b71a9b00f1c95e1d33e3df58842216bff6c4898c7e42089447
artifact size:          13,867 bytes
raw speedup:            1.733134845814182x
candidate median:       13.165--13.169 us
Reference median:       22.817--22.826 us
candidate logical rate: about 7.64 TFLOPS under the task-owned logical formula
input tokens:           1,617,635
cached input tokens:    1,503,232
output tokens:          27,953
Harbor estimated cost:  USD 1.6179648 (not an external billing receipt)
```

四个性能输入池：

| case | Candidate median | Reference median | speedup |
|---|---:|---:|---:|
| random-standard | 13.169 us | 22.817 us | 1.732589x |
| structured-head-map | 13.167 us | 22.820 us | 1.733056x |
| decay-extremes | 13.169 us | 22.823 us | 1.733003x |
| state-rounding | 13.165 us | 22.826 us | 1.733891x |

## 6. Agent 时间线与合规审计

以 Agent execution 开始为 `t=0`：

| 约时间 | 行为 |
|---:|---|
| 0--138 s | 设计 warp ownership，直接创建 PTX；首个完整版本通过 `ptxas sm_100a` |
| 178--244 s | 创建临时 C Driver loader，第一次完整 state/output GPU 对照通过 |
| 244--465 s | 64-bit vector load/store、SASS 检查和 2-row interleaving |
| 465--595 s | 扩展为 4-row interleaving，约 13.3 us，53 registers、无 spill |
| 600 s | 正在做最后数值稳健性检查，被 Harbor timeout 取消 |

Trajectory 审计：

- 只提交 `/app/kernel.ptx`；
- 直接创建和多次修改 PTX，并使用 `ptxas` 验证；
- 使用 `cuobjdump` 和 `ncu` 做资源/SASS/profile 分析；
- 临时 `/tmp/validate_kda.c` 只通过 CUDA Driver API 加载已手写 PTX并计算独立真值，
  没有生成 PTX；
- 临时 `/tmp/redux.ptx` 是直接 PTX 指令实验，不是高层源码；
- 没有调用 `nvcc`、NVRTC、NVVM、Triton；
- 没有 `pip/apt` 安装，没有 `curl/wget/git clone`；
- 模型 API allowlist 生效，未列入白名单的 `*.oaiusercontent.com` 被明确拒绝。

因此该 attempt 满足当前 Direct-PTX 作答约束。

## 7. 运行后诊断修正与当前判断

Agent 运行时 `/tests/test.sh` 的额外 `ptxas` 日志命令仍写成 `-arch=sm_100`，对合法
`.target sm_100a` artifact 产生一条误导性的 diagnostic failure；CUDA Driver loader
和隐藏 correctness/performance 已正常运行，不受该 `|| true` 诊断影响。运行后已将
命令改为 `-arch=sm_100a`。该修改会改变后续 task checksum，但不改变公式、ABI、
hidden cases 或 reward 逻辑。

修正后做了两项复验：

```text
Agent artifact direct verifier:
  ptxas sm_100a: 53 registers, 0 stack, 0 spills, 0 barriers
  correctness: 1
  reward:      1
  raw speedup: 1.7330932472601117x

current canonical Oracle Job:
  jobs/ptx-kda-b200-oracle-final-current-20260822/
  task checksum: 4917f07359ae6bf32edc95583f990c54a25a52cdadc7f6fbd6100b162873298e
  correctness: 1
  reward:      1
  raw speedup: 1.0013577926115271x
```

因此当前 task 文件本身仍有运行证据；Agent feasibility 的正式 Harbor Job保留其运行时
checksum `86e036...`，不能把两个 checksum 写成相同。

当前结论：

- 任务和 harness 已完成 B200 feasibility；
- 单次强模型 artifact 通过，说明本题不是“不可做”；
- Agent 没有正常退出，600 秒对完整优化流程偏紧；
- 单次通过不能推出真实通过率，更不能把 D5 标签冻结为经验难度；
- Agent 比 task-owned Reference 快 `1.73x`，说明当前 Reference 太弱，不能用它冻结
  正式性能排行榜；
- 下一步若继续校准，应固定运行后 checksum，重复多个 Agent attempt，并加入开放权重
  模型；正式性能对照应加强 Reference 或接入固定 revision 的 INT21 路径。

## 8. 当前 canonical checksum 的两次重复校准

在不修改 task、prompt、模型、reasoning effort、600 秒预算、SIF 或 verifier 的前提下，
固定当前 task checksum：

```text
4917f07359ae6bf32edc95583f990c54a25a52cdadc7f6fbd6100b162873298e
```

于 B200 物理 GPU 3 串行运行两个全新 Trial：

```text
jobs/ptx-kda-b200-codex-gpt56-high-current-repeat2-r2-20260822/
```

| Trial | correctness | reward | raw speedup | Agent 时间 | 结束方式 |
|---|---:|---:|---:|---:|---|
| `ptx-kda-recurrent-decode-bf16-t1__PcV83aB` | 1 | 1 | 1.530861x | 602.780 s | `AgentTimeoutError`；截止 artifact 通过 verifier |
| `ptx-kda-recurrent-decode-bf16-t1__kra4cZY` | 1 | 1 | 1.015362x | 238.641 s | 正常结束，无异常 |

Artifact 证据：

| Trial | size | SHA-256 |
|---|---:|---|
| `PcV83aB` | 9,399 bytes | `941e172887278b7525c9b259d943a454e04668417432727147d4af7b23a1964a` |
| `kra4cZY` | 8,215 bytes | `66cf1577e77c11f98715d5bb7a7a1925e3664a92fad95598fa3ce56fd9227651` |

两次都通过全部六个 correctness case。Job 汇总中的一个 `n_errors` 来自第一轮的
Agent timeout，不是 verifier correctness failure；该轮仍有 `correctness=1` 和
`reward=1`。

启动前曾产生一个单独的零秒资源失败 Job：两次均在 Agent 启动前因所有 B200 超过
256 MiB 安全阈值而触发 `GpuAllocationError`。它没有 Agent tokens、artifact 或
verifier 结果，不计入模型通过率。随后等待 GPU 连续三次保持空闲，再完成上述两个
有效 Trial。

难度判断必须分两种口径报告：

- 当前 canonical checksum：Codex/gpt-5.6-sol/high correctness 为 `2/2`；
- 加上第 5 节仅诊断命令不同、评分语义相同的 feasibility Trial：语义等价结果为
  `3/3`。

因此，P11 虽有复杂的 state recurrence 和 launch mapping，但当前 `T=1` 固定形状对
该强模型的 correctness 已表现为容易题，不能再把 “D5 candidate” 当作经验难度结论。
600 秒内是否正常退出仍有方差（这三次中两次 timeout、一次约 239 秒自然结束），但
timeout 不应代替 correctness 难度。下一版应增加真正的语义维度，例如独立设计
`T>1`/speculative metadata 变体；若要评价性能，还必须先加强 Reference，不能只提高
现有 reward 门槛。
