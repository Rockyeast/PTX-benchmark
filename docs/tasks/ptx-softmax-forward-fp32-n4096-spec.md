# `ptx-softmax-forward-fp32-n4096` 任务规格

**状态：Agent-calibrated development task（2026-08-22）。第一版数学语义、张量布局、
entry 和 launch ABI 已实现并在 B200 验证；Codex/high 五次通过 5/5，因此当前是容易锚点，
尚未冻结为 release。**

## 1. 定位

这是 PTX-Bench 第一批 pilot 的 P4，属于 **Normalization / Reduction** 家族，暂定
难度为 **D2**。它只评测下面这个 specialization：

```text
row-wise softmax forward
+ FP32 input
+ FP32 output
+ row width N=4096
+ contiguous row-major layout
+ NVIDIA B200 / sm_100
```

本题不包含 temperature、mask、causal、dropout、backward、in-place、BF16/FP16 输出、
动态列宽或跨行归约。它们会改变公式、接口或优化方法，应作为独立 task variant。

运行时可以改变 `num_rows`，因为所有规模仍使用相同 entry、相同 `N=4096` 和相同
launch ABI。因此多个 `num_rows` 属于同一道 task 的 verifier cases，不拆成新 task。

本题排在 SiLU-gate 之后，因为它第一次要求 Agent 在 Direct PTX 中完成：

- 一行内的最大值归约；
- 数值稳定的指数计算；
- 一行内的指数和归约；
- block 内线程同步；
- FP32 load/store 与 FP32 中间计算。

## 2. 上游依据与许可证

设计依据固定到 FlashInfer 官方仓库的以下 clean revision：

```text
repository: https://github.com/flashinfer-ai/flashinfer
commit:     103fcf862b902e100775ac781a0de36d067de3a4
license:    Apache-2.0
```

直接审阅的材料：

- [`include/flashinfer/sampling.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/include/flashinfer/sampling.cuh)：
  `OnlineSoftmaxFusedKernel` 采用稳定的在线最大值/分母累计、block reduction 和最终归一化；
- [`csrc/sampling.cu`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/csrc/sampling.cu)：
  公开 binding 把输入解释成二维 `[batch_size, vocab_size]`；
- [`flashinfer/trace/templates/sampling.py`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/flashinfer/trace/templates/sampling.py)：
  correctness reference 先转 FP32、可选除 temperature，再沿最后一维做 softmax；
- [`tests/trace/test_reference_correctness.py`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/tests/trace/test_reference_correctness.py)：
  对 FP32 API 与独立 reference 做交叉检查；
- [`LICENSE`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/LICENSE)：
  上述代码采用 Apache-2.0。

FlashInfer 提供与本题相同的 FP32 softmax 数学语义和重要实现依据，但仍不是可直接提交
的 raw-PTX 标准答案：

- 当前 `csrc/sampling.cu` 实际实例化 `OnlineSoftmax<float>`，与本题 dtype 一致；
- 上游 operator 支持 temperature 和自己的 host dispatch，本题固定 `temperature=1`；
- 上游会根据 batch/vocab 选择单 block 或切片多 kernel 路径，本题固定单 entry、每行一个
  block；
- 本题的 entry 名、参数和 launch ABI 由 PTX-Bench 单独冻结。

因此不得把 FlashInfer 的完整 operator 冒充为现成的同 ABI PTX Reference。实现阶段可
基于其 CUDA 代码适配固定单-entry Reference，但必须记录 revision、适配 diff、
Apache-2.0 attribution 和生成 PTX 的工具链哈希。

## 3. Agent 可见接口

Agent 必须直接创建：

```text
/app/kernel.ptx
```

PTX 文件必须包含 visible entry：

```text
softmax_fwd_fp32_n4096(
    const float* input,
    float* output,
    uint32_t num_rows)
```

这里的 C 风格声明只表示参数顺序和语义。实际提交必须使用合法 PTX `.entry` 和
`.param` 声明；两个 device pointer 均按 64-bit 传递。

张量布局固定为：

```text
input  : [num_rows, 4096], float32, contiguous row-major
output : [num_rows, 4096], float32, contiguous row-major
```

`input` 与 `output` 不得 alias。Verifier 保证 allocation 基地址至少 16-byte 对齐；
每行大小为 `4096 * 4 = 16384` bytes，也保持 16-byte 对齐。

### 3.1 数学语义

对每一行 `r`：

```text
x_j = input[r, j]                              j = 0..4095
m   = max_j(x_j)
s   = sum_j(exp(x_j - m))
y_j = exp(x_j - m) / s

output[r, j] = float32(y_j)
```

最大值、指数、求和和除法均按 FP32 解释。输入 case 不包含 NaN、正负无穷，也不包含使
整行分母为零的未定义情况。

Candidate 可以使用 PTX 的近似指数指令，只要最终结果满足数值标准。直接计算
`exp(x_j)` 而不减去行最大值，不视为数学等价的可靠实现，因为隐藏 case 会覆盖大正值
和大负值。

### 3.2 固定 launch ABI

Verifier 固定使用普通 CUDA Driver API launch：

```text
block dimensions:         (1024, 1, 1)
grid dimensions:          (max(1, num_rows), 1, 1)
dynamic shared memory:    0 bytes
programmatic dependency:  disabled
```

一个 block 对应一行，每个线程负责四个或更多元素。`num_rows=0` 时 verifier 仍启动一个
block，Candidate 必须在任何全局内存读写前返回。

固定 1024 threads 的依据是：FlashInfer 在 compute capability >= 8 的 sampling softmax
路径中使用 1024-thread block；对 `N=4096`，这也恰好给每个线程分配四个元素。这里只
借用线程组织依据，不宣称 launch 与上游完整 operator 等价。

Candidate 可以在 PTX 中声明静态 `.shared` 数组完成归约，但 verifier 不提供 dynamic
shared memory。目标设备是 NVIDIA B200，建议 `.target sm_100`。

任务采用 Direct-PTX 环境：Agent 可使用固定版本 `ptxas`、`cuobjdump` 和离线 PTX ISA
文档，但没有 `nvcc`、NVRTC、NVVM 或 Triton，也不能下载安装高层到 PTX 的工具链。

## 4. 正确性标准

正确性是硬门槛。PTX 汇编、module load、entry 查找、launch、同步、数值、输入不变性
或内存安全任一失败，总 reward 为 `0`，不进入性能计时。

### 4.1 独立真值

Verifier 不读取 Oracle 输出，也不逐行比较 Candidate 与性能 Reference。它独立执行：

```python
x = input_fp32
expected = torch.softmax(x, dim=-1)
```

实现时还应保留一份不依赖 Candidate/Reference 的稳定 FP32 CPU 公式，用固定小 case 与
PyTorch 结果交叉检查，避免 verifier 自身把维度或归约范围写错。

Candidate 和 task-owned 性能 Reference 都必须分别通过独立真值。

### 4.2 初始数值门槛

近似指数和不同归约顺序不适合要求逐 bit 相同。第一版暂定同时检查：

```text
elementwise: abs(candidate - expected) <= 2e-6 + 2e-3 * abs(expected)
row sum:    abs(sum(float32(candidate_row)) - 1) <= 2e-4
all values: finite and >= 0
```

该容差必须通过 Oracle、CUDA Reference、慢正确实现和数值错误负例校准后才能冻结。
若容差无法区分“正确近似”与“明显错误分布”，优先增加结构化输入和独立分布指标，不能
为了让 Oracle 通过而随意放宽。

### 4.3 初始 correctness cases

公开 instruction 只承诺 `N=4096`、FP32、有限输入和普通 contiguous layout；具体数值
与 seed 留在隐藏 verifier。初始 case 至少覆盖：

| `num_rows` | 输入类型 | 主要检查 |
|---:|---|---|
| 0 | 空输入 | 零规模不得访问内存 |
| 1 | 全零/全相等 | 每项应接近 `1/4096` |
| 3 | 固定梯度与交替正负值 | 索引、整行归约、非整齐 row count |
| 17 | 随机值加大正/负 offset | 必须减最大值，不能直接指数溢出 |
| 128 | 随机值与单个 dominant logit | 多 block、尾部行和高概率元素 |

每个非空 case 都要检查：

- 全部 4096 列，不只检查前缀；
- input 在 launch 前后完全不变；
- output 前后 canary 未被修改；
- CUDA launch/synchronize 无错误；
- 不读取或写入相邻行；
- 多个隐藏 seed 可复现。

## 5. 性能评测草案

### 5.1 三条独立路径

```text
CPU/PyTorch stable FP32 formula
    只负责 correctness truth

Candidate /app/kernel.ptx
    Agent 直接提交并接受评分

Task-owned Reference PTX
    与 Candidate 使用相同 entry、参数、block/grid 和 dynamic-smem ABI
    只负责同卡性能基线，也必须先通过独立 correctness
```

Reference 实现阶段可依据 FlashInfer 的稳定在线 softmax 结构编写一份固定 ABI 的 CUDA
kernel，在可信出题环境中预编译为 PTX。trial 环境不包含编译器，只加载预编译
Reference。必须保存 CUDA source、许可证说明、nvcc 版本/flags、生成 PTX 和 SHA-256。

### 5.2 计时与 reward

沿用已有 PTX-Bench harness：同一 GPU/context、相同输入、CUDA Event、预热、自适应
batch、多批次中位数、Candidate/Reference 顺序交替，并保存全部原始时间。

初始计分规模使用 `R = num_rows`：

```text
R in {1, 32, 256}

r_R                  = reference_median_time_R / candidate_median_time_R
speedup_vs_reference = geometric_mean(r_1, r_32, r_256)
performance          = clamp(speedup_vs_reference, 0, 1)
reward               = correctness * performance
```

同时保存未封顶 speedup、每个规模的绝对时间和最低逻辑内存带宽：

```text
logical_bandwidth = num_rows * 4096 * (4-byte read + 4-byte write) / median_time
```

这里的 8 bytes/element 只是接口最低流量，不代表 kernel 真实 DRAM 流量；没有缓存输入的
实现可能重复读取。最终计分规模和公式必须经过 Reference 稳定性、慢正确实现及真实
Agent pilot 后才能冻结。

## 6. 实现阶段负例

至少准备以下作者侧 fixture，不提供给 Agent：

- 缺少 `softmax_fwd_fp32_n4096` entry；
- 直接计算 `exp(x)`，在大 offset case 溢出；
- 只做 warp 内 max/sum，没有合并整个 block；
- 只归约前 1024 个元素，遗漏其余三段；
- 使用最大值但忘记除以整行分母；
- 把所有行错误地共享同一个 max 或 sum；
- 只处理第一行、遗漏最后一行或错误计算 row stride；
- 写错 FP32 地址步长，或输出未归一化的指数值；
- 修改 input、越界写 output，或 `num_rows=0` 仍访问内存；
- 数值正确但故意采用低效串行/重复全局读取，用来确认 correctness 为 1、性能分数下降。

## 7. 从规格到可运行 task

- [x] 固定 operation、direction、dtype、layout、N 和硬件目标；
- [x] 审阅 FlashInfer 算法、binding、trace reference、测试、revision 和许可证；
- [x] 说明上游 FP32 operator 与本题固定 raw-PTX ABI 的差异；
- [x] 固定 entry 参数、张量布局和普通 launch ABI；
- [x] 草拟独立 truth、容差、cases、内存检查、Reference 和 reward；
- [x] 人工审阅并确认本规格；
- [x] 从 `templates/direct-ptx-task-v0/` 生成 fail-closed scaffold；
- [x] 编写 Agent-facing `instruction.md`；
- [x] 实现独立 correctness verifier；
- [x] 实现 Oracle PTX 和同 ABI 性能 Reference；
- [x] 添加错误负例与慢正确实现；
- [x] 在 B200 完成 Oracle、负例、重复性能和 Harbor Job；
- [x] 运行固定配置的五个真实 Agent，校准 D2 标签和 600 秒预算。

详细结果见
[`ptx-softmax-forward-fp32-n4096-validation.md`](ptx-softmax-forward-fp32-n4096-validation.md)。
本题当前对 Codex/high 为 5/5 容易锚点；在补齐开放权重模型和正式 no-network backend
验证前，不得进入 frozen release manifest。
