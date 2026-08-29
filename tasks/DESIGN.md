# PTX-Bench Task 设计规范

本文件是 Direct-PTX task 的权威设计规范。目录、创建、验证和运行流程见
[`README.md`](README.md)。

## 测量目标

任务测量模型直接编写和优化 NVIDIA PTX 的能力。最终提交固定为
`/app/kernel.ptx`；允许使用 `ptxas`、`cuobjdump` 和规定的 PTX 文档，但不允许
通过 CUDA C++、Triton、NVRTC、NVVM 或其他高层工具链生成提交。

不同文档和网络设置必须分开报告，例如在线官方文档与只使用本地文档不能合并为同一组
结果。

## 一道 task 的边界

一道 task 由以下合同共同确定：

- operation，以及 forward/backward/fusion 边界；
- dtype 和物理 layout；
- 目标 GPU、SM 与 PTX ISA；
- entry 名称、参数顺序和类型；
- grid、block、动态 shared memory 和 launch 责任；
- shape specialization 和隐藏 verifier cases。

只改变运行时 size、但 entry、参数和 launch ABI 不变时，通常作为同一道 task 的多个
case。改变语义、entry、输出、layout、硬件 specialization 或 kernel 拓扑时，拆成新
task。多 entry module integration 是明确的另一种任务形态，不能用宽泛名称掩盖多个
无关问题。

## 四个固定角色

所有正式性能任务只使用以下术语：

```text
Candidate                  # Agent 修改后的 /app/kernel.ptx
正确性真值                  # verifier 内部的独立数学定义
性能 Baseline              # 正确、现实但普通的性能零分线
性能 Reference             # 正确、固定且高性能的性能满分线
```

`environment/reference.py` 是 Agent 可见的慢速数学参考，用于解释语义和公开自测；它不
代替隐藏正确性真值，也不自动等同于性能 Baseline。

性能 Baseline 和性能 Reference 必须使用与 Candidate 相同的 PTX entry、参数和 launch
合同。verifier 先检查三者在隐藏 cases 上都正确，再计算 Candidate 在零分线和满分线之间
的位置。具体归一化公式必须写入该题 verifier 和公开评分说明，不能只存在于实验笔记。

## 选题和性能来源

正式任务优先来自真实 ML 系统或维护中的开源 kernel 库。先固定一个可审计的高性能实现，
再从它支持的 operation、dtype、layout、shape 和硬件范围反推 task 合同。没有可信性能
Reference 的题可以作为 correctness、基础设施或 feasibility pilot，但不能冻结为正式
性能任务。

正式任务还必须同时满足：

- 计算属于真实模型或 ML 系统，而不是只为 benchmark 发明的玩具；
- 正确性可以由独立、简单且可审计的数学定义判断；
- 未修改骨架、常数输出、部分输出和明显错误语义不能通过；
- 人类实现能在规定硬件和时间预算内完成，任务不是不可解题；
- 目标模型的结果存在差异，任务能够提供区分信号。

性能 Baseline 必须是正确、现实且可复现的普通实现，不能为了制造高分故意写慢。性能
Reference 应来自可信高性能实现，并且和 Candidate 使用相同的 entry、参数、layout、shape
范围和 launch 合同。

性能 Reference 若来自第三方，任务设计记录必须写明来源 URL、固定 revision、许可证和
从源码得到 PTX 的方式。正式评分直接使用固定的
`tests/performance_reference.ptx`，不在评分时访问网络下载答案。

## 正确性和隐藏评分

`tests/verifier.py` 必须使用独立、简单的数学定义生成正确性真值，不能用 Candidate 或
性能 Reference 的输出互相证明。隐藏 cases 至少覆盖公开开发 case 之外的 shape、边界和
一个该 operation 常见的错误语义。

正确性是性能分的硬门槛：任何 case 错误、entry 缺失、模块加载失败、非法显存访问、输入
被修改或输出未完整写入，都得到 reward `0`。只有全部正确后才计时。

## 性能测量

Candidate、性能 Baseline 和性能 Reference 必须在同一块 GPU、相同输入、相同 launch 合同
和相同计时方法下运行。计时应使用 CUDA Event，包含充分 warmup、显式同步和多次独立重复，
并保留原始样本，不能只保存最终 reward。

一道 task 可以包含多个未公开的运行时 shape；当它们共享同一语义和 ABI 时，应共同参与
性能统计，避免只针对一个公开 shape 优化。具体 shape、warmup、重复次数和统计量属于冻结的
task 配置，修改后必须视为新版本。

正式 release 应使用统一的归一化公式。该公式只能在多道任务都具备合格的性能 Baseline、
性能 Reference 和目标硬件重复数据后冻结；在此之前，pilot 的分数不能作为正式排行榜分数。

## 防作弊边界

- Agent 只能提交 PTX，可以使用 `ptxas` 和 `cuobjdump`，不能借助高层编译器生成答案；
- 正式任务不能在运行时从公网下载实现、编译器或额外软件包；
- 隐藏 verifier、正确性真值、性能 Baseline 和性能 Reference 只在评分阶段提供；
- 公开 `measure.py` 不能暴露隐藏 cases、Reference 性能或最终 reward；
- 发布高分结果前必须审计 trajectory，确认没有下载答案、生成高层源码或修改评分逻辑。

## 难度与校准

目录复杂度不是难度。任务发布前必须固定 task 内容、目标 GPU、Agent wrapper、模型、
推理强度、网络/文档设置和墙钟预算，然后运行多个独立 attempt。全通过和全失败都只能
作为锚点或 probe；正式题组应在目标模型之间产生稳定差异。

一个 proprietary 模型的结果不能替代计划中的开放权重模型网格。Correctness、原始性能、
截断后的 reward、正常结束/超时和 trajectory 合规证据应分开记录。

## 发布条件

一道正式任务至少必须完成：

1. 静态 release checker 通过；
2. 性能 Reference 通过全部正确性检查并获得预期高分；
3. 未修改起始骨架和常见错误实现得到 `0`；
4. 性能 Baseline 与性能 Reference 均正确，且计时重复具有足够余量；
5. 固定配置下完成多个真实 Agent 的独立校准和 trajectory 审计。

缺少任意一项时，只能标记为 infrastructure、feasibility 或 calibration，不能进入正式分数。

## 数据集平衡

正式数据集按 kernel family 组织，并先在 family 内平均，再对 family 平均，避免某个 family
因 shape 或任务数量多而支配总分。最终公式只能在多模型校准后冻结。
