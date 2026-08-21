# PTX-Benchmark 数据集总体设计

**状态：Draft v0.1（2026-08-20）。这是数据集级约定，不代表题目和总分已经冻结。**

## 1. 目标与当前边界

PTX-Benchmark 评测 Agent 能否在给定 NVIDIA GPU、固定输入输出接口和资源约束下，
交付正确且高性能的 PTX kernel。

v0.1 只做类似 ΦBench KFC（Kernel Function Completion）的局部任务：

```text
一个 instruction
    -> Agent 提交一个 /app/kernel.ptx
    -> correctness 硬门槛
    -> 同卡 performance 测量
    -> Harbor 保存 reward 和轨迹
```

暂不把多文件仓库修改、系统级优化或 evolution loop 混进同一个 task。它们以后可以
作为独立 track，复用本数据集提供的 correctness/performance reward。

卓明录音中的顺序是：先建 PTX benchmark 和现代 kernel 题集，再在题集上研究 Agent
方法。K3 总分约 30～40 是未来难度校准目标，不是当前单题的阈值。

## 2. 开发集与发布集

数据集分成两种用途：

```text
ptx-bench-dev
├── 允许修改 task、测试和评分公式
├── 包含基础设施校准题与未冻结 pilot
└── 用于 Oracle、负例、真实 Agent 和 evolution 开发

ptx-bench-release-vN（未来创建）
├── 只包含通过全部 release gate 的 task
├── task digest、环境和评分公式冻结
└── 用于正式模型对比和论文结果
```

当前 Harbor manifest 是 [`../../tasks/dataset.toml`](../../tasks/dataset.toml)，名称为
`ptx-bench/ptx-bench-dev`。它不是最终 leaderboard 数据集。

该 manifest 由本地 Harbor 0.21.0 创建并同步；远端个人补丁版 Harbor 0.13.2 已完成
只读解析，得到相同的数据集名称和两个 task。当前 CLI/schema 兼容，但正式发布前仍
需统一并固定 Harbor 版本。

若以后实现 evolution loop，应使用 dev/train task 或公开 shape；最终报告还需要
held-out task/shape，避免反复查询同一隐藏 verifier 造成过拟合。

## 3. 一道 task 的粒度

一道 task 由下面五项共同确定：

```text
operation
× direction / fusion
× dtype / layout
× hardware target
× shape specialization 与固定 launch ABI
```

例如：

```text
RMSNorm × forward/plain × BF16/contiguous × B200 × N=4096
```

同一个优化策略和 launch ABI 下，仅 `M` 等 batch 规模变化，可以作为一道 task 的
多个 verifier case。若需要不同 entry、不同输出、不同 kernel 拓扑或完全不同的
硬件 specialization，应拆成不同 task。forward/backward、plain/fused residual 也
默认拆开，不能用一个宽泛名称掩盖多个问题。

## 4. 题目家族

最终题集按 kernel 能力而不是按上游仓库分组。初始家族如下：

| 家族 | 主要能力 | 候选例子 |
|---|---|---|
| A. Normalization / Reduction | warp/block reduction、数值稳定性 | RMSNorm、LayerNorm、Softmax |
| B. Activation / Fusion | 向量化、融合访存、特殊函数 | SiLU-gate、GELU、bias-activation |
| C. Position / Data Transform | layout、索引、向量 load/store | RoPE、transpose、KV append |
| D. GEMM / Low Precision | Tensor Core、scale、tile、pipeline | BF16 GEMM、FP8/NVFP4 quant/GEMM |
| E. Attention / Sequence State | online softmax、KV cache、复杂数据流 | prefill、decode、paged attention、KDA |
| F. MoE / Routing / Sampling | irregular access、top-k、grouped work | grouped GEMM、routing、Top-P |

`vector_add`、`add_one` 等只用于基础设施和评分校准，不计入最终家族覆盖率。

50～100 道是后续规模目标，不是现在就要填满的数字。先完成五道 pilot，确认 task
粒度、Reference 适配成本和模型得分分布，再决定每个家族的数量。

## 5. 难度层级

| 等级 | 特征 | 作用 |
|---|---|---|
| C0 | 简单逐元素或内存操作 | 基础设施校准，不进正式总分 |
| D1 | 单 kernel、规则访存、少量融合 | 检查基本 PTX 与向量化能力 |
| D2 | reduction、共享内存或不规则索引 | 中等 pilot |
| D3 | Tensor Core、低精度 scale、复杂流水 | 高难正式题 |
| D4 | attention/stateful/多阶段算法 | 前沿题；可能需要新的 task format |

难度由真实 Agent 分布决定，而不是仅凭代码长度主观指定。

## 6. 每道题的统一 release gate

状态按下面顺序推进：

```text
idea
  -> sourced
  -> specified
  -> feasibility-validated
  -> verifier-validated
  -> agent-calibrated
  -> frozen
```

进入发布集前必须满足：

- operation、dtype、layout、shape 和 launch ABI 已冻结；
- correctness truth 独立于 Candidate、Oracle 和性能 Reference；
- 有边界、随机、内存 canary、错误 entry 和数值错误负例；
- 有同 ABI 性能 Reference，并记录来源、许可证和固定 revision；
- B200 上重复测量稳定，原始 batch timing 可审计；
- Oracle、正确慢实现、至少两类错误实现均得到可解释分数；
- 至少运行多个真实 Agent/模型，确认既不是全 0，也不是普遍满分；
- 记录 Harbor、CUDA、SIF、GPU、task digest 和超时；
- 单进程运行时间低于服务器 2.5 小时限制。

## 7. 正确性、性能与总分

单题继续使用“正确性硬门槛”：

```text
correctness = 0  -> reward = 0
correctness = 1  -> 才计算 performance
```

性能必须在同 GPU、相同输入和明确等价的测量边界内比较。PyTorch/CPU 公式负责真值，
Oracle 是已知可行提交，Reference 负责性能基线；三者不能互相替代。

单题 reward 暂定在 `[0,1]`。最终总分建议从 `100 × family-balanced macro average`
开始校准：先在每个家族内平均，再对家族平均，避免某个家族因为 shape 多而支配总分。
该公式要在多个模型跑完后才能冻结。

## 8. 工具设置

必须把以下两种 setting 分开报告：

```text
tool-assisted PTX delivery
    环境含 nvcc/ptxas；只要求最终提交 PTX

direct-PTX
    限制高层编译器，评测模型直接编写/修改 PTX 的能力
```

最初的共用 SIF 属于第一种，用于稳定任务。当前任务已经改为第二种：Reference 在
出题阶段预编译并固定，Agent/Verifier 共用的 direct-PTX SIF 移除了 `nvcc`、NVRTC
和 CUDA 高层编译前端，只保留 `ptxas` 等 PTX 级工具。旧的 tool-assisted 运行结果
只能保留为基础设施校准，不能混入 direct-PTX 排行榜。

## 9. 数据集构建顺序

1. 保留 `vector_add` 作为 C0 校准题；
2. 将 RMSNorm 从 feasibility 推进到 agent-calibrated；
3. 按 [`task-catalog.md`](task-catalog.md) 依次完成五道 pilot；
4. 检查不同家族的实现成本、耗时和模型分布；
5. 用 [`../task-authoring-template.md`](../task-authoring-template.md) 的 v0 模板完成
   不同家族 pilot，再冻结统一 task authoring v1 模板和 release gate；
6. 扩展到约 20 道，再决定是否扩到 50～100 道；
7. 最后建立 release manifest、模型矩阵和 evolution track。

## 10. 设计依据

- 卓明录音：[`../source/录音.txt`](../source/录音.txt)
- ΦBench：<https://faibench.org/>，将任务分为单文件 KFC、多文件 LHI 和系统级 E2EO
- Harbor：<https://github.com/harbor-framework/harbor>
- INT21 RMSNorm-B200：<https://github.com/Int21-AI/RMSNorm-B200>
- INT21 KDA-B200：<https://github.com/Int21-AI/KDA-B200>
- FlashInfer：<https://github.com/flashinfer-ai/flashinfer>
- FlashInfer-Bench：<https://github.com/flashinfer-ai/flashinfer-bench>
- NVIDIA CUTLASS：<https://github.com/NVIDIA/cutlass>
