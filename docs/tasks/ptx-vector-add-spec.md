# `ptx-vector-add` 任务规格

**状态：Validated v0.1（2026-08-20）。实现、B200 正负例、慢实现、Oracle
重复性检查和真实 Agent 校准均已完成。**

## 1. 定位

`ptx-vector-add` 是 PTX-Bench 的第一道性能评测校准题，用来建立下面这条新链路：

```text
Agent 提交 PTX
    -> 隐藏测试先检查正确性
    -> 同一张 B200 上测 Candidate 和 Oracle
    -> 生成 correctness、performance 和总 reward
```

它比 `ptx-add-one` 多了性能测量，但 vector add 本身过于简单，不代表最终题库所需
的“现代且有意义的 kernel”。v0.1 默认把它视为 verifier 和评分方法的校准题；在
看到模型与 Oracle 的实测分布前，不决定是否计入最终公开总分。

录音中“K3 得 30–40 分”指未来完整题集的总体校准目标，不用于倒推这道单题的
阈值。

## 2. Agent 可见接口

Agent 必须创建：

```text
/app/kernel.ptx
```

文件必须包含下面的 visible entry：

```text
vector_add(const float* a, const float* b, float* output, uint32_t n)
```

逻辑语义：

```text
output[i] = a[i] + b[i],  0 <= i < n
```

启动约定由 verifier 固定，不由 Agent 提交：

- block：`(256, 1, 1)`；
- grid：`(max(1, ceil(n / 256)), 1, 1)`；
- dynamic shared memory：`0` bytes；
- 参数顺序固定为 `a, b, output, n`；
- 地址宽度为 64 bit；
- 目标运行设备为 NVIDIA B200；建议 PTX 使用 `.target sm_100`。

只有 `/app/kernel.ptx` 参与评分。文件大小上限暂定为 1 MiB；超过上限直接判定
错误，避免异常提交拖垮 JIT。

Agent timeout 暂定 600 秒，Verifier timeout 暂定 120 秒，均低于服务器单进程
2.5 小时限制。

## 3. 正确性测试

正确性是性能评分的硬门槛。任何编译、加载、entry 查找、启动、同步或结果错误都
令总 reward 为 `0`，不继续计时。

### 3.1 固定边界规模

至少覆盖：

```text
0, 1, 2,
31, 32, 33,
127, 255, 256, 257,
1023, 1024, 1025,
65537
```

这些规模用于发现遗漏边界判断、错误依赖 warp/block 整除，以及错误的 grid
索引计算。

### 3.2 隐藏随机规模

Verifier 使用固定但不写进 `instruction.md` 的随机种子，再生成若干不规则规模；
至少包含一个大于 `2^20` 且不是 256 整数倍的规模。固定种子保证同一 task 版本
可复现。

输入只使用有限 `float32` 数值，不加入 NaN 或 Inf。参考输出在 CPU 上以
`float32` 语义计算，比较暂定为：

```text
abs(candidate - reference) <= 1e-6 + 1e-6 * abs(reference)
```

### 3.3 内存边界

输出区域前后放置 canary。kernel 运行后同时检查：

- 所有有效输出元素；
- output 前后的 canary 未改变；
- 输入 `a` 和 `b` 未被修改。

这能区分“结果看起来正确”和“发生越界写入”。

## 4. 性能测量

性能测试仅在全部正确性测试通过后运行。

### 4.1 规模

v0.1 候选规模：

```text
2^20, 2^24, 2^26 elements
```

最大规模的三个 `float32` 数组约占 768 MiB。最终规模必须在 B200 实测后冻结；
若某个规模主要测到启动开销而不是 kernel，则调整或移除。

### 4.2 Candidate 与编译器参考实现

Verifier 在同一个 CUDA context、同一张 GPU、同一套输入和同一启动配置下分别
加载：

- Candidate：`/app/kernel.ptx`；
- Reference：隐藏的 `tests/reference.cu`，由 NVIDIA `nvcc` 在 verifier 启动时
  编译为临时 PTX。

Reference 改编自 NVIDIA CUDA Samples 的 `vectorAdd` 基本实现，来源与许可证记录在
`tests/REFERENCE.md`。它只提供同卡性能基线，不决定正确性；Candidate 与
Reference 都必须先通过独立 CPU `float32` 真值检查。两者都先完成编译/JIT 和预热，
编译、JIT、内存分配以及 Host/Device 拷贝不计入 kernel 时间。

### 4.3 计时方法

- 使用 CUDA Event，不使用 Python `time.time()`；
- 每个实现先 warmup 10 次；
- 一个测量 batch 连续启动多次 kernel，再用总时间除以启动次数；
- 自适应增加 batch 内迭代次数，目标是累计至少 50 ms，最多 10000 次；
- 每个规模测 5 个 batch，取中位数；
- Candidate/Reference 的先后顺序交替，降低温度和频率漂移偏差；
- 若 5 个 batch 的离散程度超过暂定 5%，记录 unstable，并完整重测一次。

日志必须保存每个规模的原始 batch 时间，不能只保留最终平均值。

## 5. Reward

Harbor 0.13.2 会优先读取 `/logs/verifier/reward.json`，其内容可以是多个数值指标。
本题写入：

```json
{
  "reward": 0.83,
  "correctness": 1.0,
  "performance": 0.83,
  "speedup_vs_reference": 0.83
}
```

设每个性能规模的比值为：

```text
r_n = reference_median_time_n / candidate_median_time_n
```

v0.1 暂定：

```text
speedup_vs_reference = geometric_mean(r_n)
performance          = clamp(speedup_vs_reference, 0, 1)
reward               = correctness * performance
```

正确性失败时四项均写入 `0`。比 Reference 更快的原始比值仍保存在日志中，但总 reward
暂时封顶为 `1`。

这只是首轮可运行公式，不是最终 leaderboard 公式。冻结前必须至少比较 Oracle、
Reference、一个明显较慢的正确实现和多个真实 Agent；如果正确实现普遍接近 1，
说明 vector add 缺乏区分度，而不是应该人为修改阈值以制造分差。

## 6. 已实现的 task 文件

```text
tasks/ptx-vector-add/
├── instruction.md
├── task.toml
├── environment/
│   └── Dockerfile
├── solution/
│   └── solve.sh
└── tests/
    ├── test.sh
    ├── verify_and_benchmark.py
    ├── reference.cu
    └── REFERENCE.md
```

职责：

- `instruction.md`：只公开接口、输出路径和固定启动约定；
- `solution/solve.sh`：Oracle Agent 提交的标准答案；
- `tests/reference.cu`：Verifier 用 `nvcc` 编译的独立性能参考；
- `verify_and_benchmark.py`：正确性、canary、CUDA Event 计时和指标输出；
- `test.sh`：运行 verifier，并保证任何异常最终产生合法的零分文件。

`solution/solve.sh` 中的手写 PTX 与 `tests/reference.cu` 是两条独立实现路径。
Verifier 不依赖 Agent 运行后的 Oracle 输出；两条路径都由 CPU 真值检查。

## 7. 工具与“直接写 PTX”的边界

当前 Codex SIF 含 `ptxas`、`nvcc` 和 CUDA Toolkit。Agent 可以用 `ptxas` 检查
PTX 语法，也可能先写 CUDA 再生成 PTX。因此 v0.1 只能证明：

> Agent 能在工具开放的环境中交付满足接口的 PTX。

它还不能证明 PTX 是模型直接手写的。正式论文若要声称“直接 PTX 能力”，必须在
题集冻结前选择并记录一种策略：

1. 构建受限 SIF，只保留 PTX 汇编/加载所需工具；或
2. 明确允许编译器辅助，并把它作为独立 benchmark setting；或
3. 审计 Agent trajectory，但不能把人工轨迹审计当作唯一自动评分机制。

v0.1 推荐先采用工具开放 setting，把受限 setting 留到评测方法稳定后实现。

## 8. 实现验收

实现阶段必须完成：

- [x] Oracle 在全部正确性 case 上通过；
- [x] 缺少 `vector_add` entry 的 PTX 得到 reward `0`；
- [x] 有越界写入但有效区间结果正确的 PTX 得到 reward `0`；
- [x] 一个正确但明显较慢的 PTX 得到非零且低于 Oracle 的 performance；
- [x] Oracle 连续运行三次的分数与原始时延可解释，并记录波动；
- [x] 真实 Codex Agent 至少运行一次；
- [x] `reward.json` 中每个指标都能进入 Harbor `result.json`；
- [x] 所有实验记录 GPU 物理编号、SIF SHA-256、task checksum 和 Harbor 版本。

## 9. 下一步顺序

1. [x] 人工阅读并确认本规格；
2. [x] scaffold `tasks/ptx-vector-add/`；
3. [x] 实现 Oracle PTX、CPU correctness verifier 和 `nvcc` Reference；
4. [x] 实现 CUDA Event 性能测量与多指标 reward；
5. [x] 在 B200 跑 Oracle 正例和至少两个负例；
6. [x] 在 B200 连续跑 Oracle 三次，检查时延波动；
7. [x] 运行真实 Agent；它得到 reward `1.0`，证明 vector add 适合作为流程校准题，
   但区分度不足，因此不冻结为最终题库的统一评分公式。
