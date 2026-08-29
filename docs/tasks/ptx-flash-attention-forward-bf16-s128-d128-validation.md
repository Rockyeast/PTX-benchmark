# `ptx-flash-attention-forward-bf16-s128-d128` 验证记录

**状态：B200 correctness/performance harness、错误负例、慢正确实现、Harbor Oracle
和五次 Codex/high pilot 已通过；当前固定 Reference 偏弱，任务尚不可进入 release。**

## 1. 固定合同

```text
operation:       single-request causal MHA forward
dtype:           BF16 Q/K/V/output，FP32 QK/Softmax/PV accumulation
layout:          NHD contiguous
shape:           S=128, D=128, num_heads runtime variable
entry:           flash_attn_fwd_bf16_s128_d128
launch:          grid=max(1, num_heads*128), block=128, dynamic smem=0
target:          NVIDIA B200 / sm_100
```

语义来源固定到 FlashInfer 官方 revision
`fb28d7242b3506a2348265962041acc1fb56cca4`（Apache-2.0）。详细文件和链接见规格文档。
FlashInfer 只提供算子语义与 shape 依据；当前 Oracle/Reference 不是从 FlashInfer 拷贝的
raw PTX。

## 2. 独立正确性路径

```text
attention_truth.c
  CPU double QK/exp/PV + BF16 round-to-nearest-even

Candidate kernel.ptx
  Agent submission

reference.cu -> trusted nvcc -> reference.ptx
  same ABI performance baseline；也必须先通过 CPU truth
```

Correctness cases 覆盖 `num_heads=0`、causal prefix average、future token 隔离、
NHD/非二次幂 head、极端 logits 稳定性、dominant key 和 head 隔离。每个非空 case
检查完整 BF16 output、Q/K/V 逐 bit 不变和 output guard。

CPU truth 另与 PyTorch 显式 FP32 `QK^T -> causal mask -> softmax -> PV -> BF16`
交叉检查：`random-layout/H=1` 最大绝对差 `0.000244140625`，
`dominant-key/H=3` 最大绝对差 `0.000030517578125`，两组按任务容差均为 0 个失败元素。

## 3. Reference 稳定性修正

第一版 CUDA Reference 使用 block 内并行 max/sum reduction。一次完整重复在
`heads=16/dominant-key` 暴露非确定性错误，因此该版本没有被保留为成功证据。

修正后：QK 仍由 128 线程并行，128 项 Softmax reduction 由 thread 0 确定性执行，
PV 再由 128 线程并行。修正版连续五次完整 verifier 结果：

| repeat | correctness | reward / raw speedup |
|---:|---:|---:|
| 1 | 1 | `0.9998713291` |
| 2 | 1 | `0.9999465570` |
| 3 | 1 | `1.000000` / `1.0000036407` |
| 4 | 1 | `0.9999905248` |
| 5 | 1 | `0.9999127155` |

这证明修正版在该批 B200 运行中能稳定用于 feasibility 对照，但串行 Softmax 也使它
不是高性能 leaderboard Reference。

## 4. 负例和连续性能分数

| Candidate | 故意行为 | correctness | reward |
|---|---|---:|---:|
| `non-causal.ptx` | 所有 query 都读取未来 K/V | 0 | 0 |
| `partial-output.ptx` | 只写 feature 0～63 | 0 | 0 |
| `slow-correct.ptx` | 每个 block 仅 thread 0 串行完成 attention | 1 | `0.0713971149` |

前两个 PTX 均可被 `ptxas` 正常汇编，分别被 causal prefix case 和 NaN sentinel 拒绝。
慢正确实现说明 correctness 与 performance 是两条独立判据。

固定文件 SHA-256：

```text
reference.cu     deab49fdafb821edd3089cb0d6badb928bc3adc00bfa08637515b97d8422392f
reference.ptx    bf951093fdf7aa3b4f9aa8ea3a3e7996e2dc53246ae0dae8de37addbbcdb9cf4
oracle.ptx       bf951093fdf7aa3b4f9aa8ea3a3e7996e2dc53246ae0dae8de37addbbcdb9cf4
attention_truth  39725aef345f33f951934b5ab9e47c54c80d1b812a48ce341d18bea91ecede50
verifier         8b1cc21e3eb3246e95b5b6fb0de1926987db8287298257f02da7c4d71768dc33
non-causal       96358e70cf90d6756e677d3cf4089f15f225952679cce50dc94161a50802117b
partial-output   ee2cf73191cecfb409f36656122af69739fc916ec363c3e8a183f240f28bffb1
slow-correct     28098a022d0639f3d652d1d7cf65fc52ec72b304d7e4ab925e1c7321f8f4f407
```

Reference 使用 CUDA 13.2 compiler build
`cuda_13.2.r13.2/compiler.37668154_0`，flags 为
`-ptx -O3 --use_fast_math -arch=compute_100 -code=compute_100`。

## 5. Harbor Oracle Job

```text
job:   jobs/ptx-flash-attention-b200-oracle-feasibility-20260822-r2
trial: ptx-flash-attention-oracle-task__Z6iqqgM
task checksum: 3c25ff4bf593960fcd5f81f4142d648d5af26ba0ec470859d79362c2cc90f676
```

结果：

```text
correctness: 1
reward:      0.9999735897
speedup:     0.9999735897
runtime:     15 seconds
```

Oracle 和下面的 Agent Job 使用临时 task 副本移除了 `[agent]
network_mode=no-network`，因为当前 patched Singularity backend 会拒绝它不能实施的动态
网络阶段切换。正式 task 仍保留 `no-network`。因此这些 Job 证明 task 打包、SIF、
Oracle/Agent、hidden verifier 和 reward 链路，但不证明严格 Agent 网络隔离；发布前必须
用支持该策略的 Docker backend 或增强后的 Singularity backend 重跑。

## 6. Codex/high 五次 pilot

固定配置：

```text
Agent:             Codex CLI 0.148.0
model:             gpt-5.6-sol
reasoning_effort:  high
timeout:           600 seconds
hardware:          one exclusive NVIDIA B200 per trial
```

Jobs：

```text
jobs/ptx-flash-attention-b200-codex-gpt56-high-feasibility-20260822
jobs/ptx-flash-attention-b200-codex-gpt56-high-pilot-remaining4-20260822
```

| trial | correctness | Agent 秒 | raw speedup | capped reward |
|---|---:|---:|---:|---:|
| `3QfcRuv` | 1 | `296.31` | `2.521571` | 1 |
| `2Urq4JK` | 1 | `331.96` | `2.495027` | 1 |
| `LaHRPm3` | 1 | `341.83` | `2.184512` | 1 |
| `sN8UaYc` | 1 | `286.75` | `2.136241` | 1 |
| `zydLoUJ` | 1 | `338.37` | `2.395121` | 1 |

汇总：correctness `5/5`，Agent 平均 `319.05` 秒，raw speedup 平均 `2.3465`。

Trajectory 审计显示五次都直接创建 `/app/kernel.ptx`，调用允许的 `ptxas` 和
`cuobjdump`，没有调用 `nvcc`、Triton、安装命令或下载工具。四次创建了临时 host
C/C++ loader，一次使用 Python loader；这些程序只加载和检查已直接编写的 PTX，没有
生成提交 PTX，因此不违反当前 Direct-PTX contract。最终归档提交均只有
`kernel.ptx`。

## 7. 当前判断

- 固定 ABI、CPU truth、正确性 hard gate、连续 reward 和 600 秒预算可运行；
- 这道题明显比 Softmax 消耗更多 Agent 时间，但 `5/5` 不满足强模型 30%～50% 目标；
- 原 D4 目前只能解释为结构复杂度标签，不能当作经验难度结论；
- 五个 Agent 都比固定 Reference 快 2 倍以上，当前性能分数已经饱和，不能用于区分强
  Agent；
- 下一版应优先加强同 ABI Reference，或一次只增加 GQA/paged metadata/更长序列中的
  一个难点，再重新校准；不能直接把本结果发布成成熟 Flash Attention leaderboard。
