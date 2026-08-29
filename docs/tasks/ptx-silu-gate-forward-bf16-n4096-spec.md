# `ptx-silu-gate-forward-bf16-n4096` 任务规格

**状态：Feasibility v0.2（2026-08-22）。规格、Oracle、独立 correctness、固定
Reference、性能 harness、Harbor Singularity Oracle Job、同批三次重复测量和五次
Codex/high pilot 已在 B200 跑通；尚未完成开放权重模型校准和跨批次性能稳定性确认。**

## 1. 定位

这是 PTX-Bench 第一批 pilot 的 P3，属于 **Activation / Fusion** 家族。它只评测一个
明确 specialization：

```text
out-of-place SiLU-gate forward
+ fused silu(gate) * up
+ one concatenated BF16 input
+ BF16 output
+ output width N=4096
+ contiguous row-major layout
+ NVIDIA B200 / sm_100
```

本题不包含 backward、bias、residual、量化、MoE mask、GELU、FP16/FP32 输出、其他
`N` 或原地更新。它们会改变数学语义、内存流量或接口，应作为独立 task variant。

选择 `N=4096` 的依据是：

- FlashInfer 官方 activation 测试覆盖 `dim=4096`；
- `4096` 能被 BF16 的 16-byte 向量宽度整除，适合评测向量 load/store；
- 固定输出宽度后，Agent 只需优化一个融合 kernel，运行时只改变 token 数；
- 这是相对容易的 D1 锚点，可用于判断 Direct-PTX 数据集是否缺少基础但有实际意义的
  fused elementwise 任务。

这里的 `N=4096` 指 gate/up 融合后的输出宽度，因此输入每行宽度是 `2N=8192`。它不
等价于声明所有 LLM 的 FFN intermediate size 都是 4096。

## 2. 上游依据与许可证

设计依据固定到 FlashInfer 官方仓库的以下 clean revision：

```text
repository: https://github.com/flashinfer-ai/flashinfer
commit:     103fcf862b902e100775ac781a0de36d067de3a4
license:    Apache-2.0
```

直接审阅的官方材料：

- [`flashinfer/activation.py`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/flashinfer/activation.py)：
  公开 API 明确输入最后一维为 `2 * hidden_size`，输出为 `hidden_size`；
- [`include/flashinfer/activation.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/include/flashinfer/activation.cuh)：
  CUDA kernel 使用每 token 一个 block、16-byte 向量 load/store，并计算
  `Activation(gate) * up`；
- [`flashinfer/jit/activation.py`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/flashinfer/jit/activation.py)：
  定义 SiLU 为 `x / (1 + __expf(-x))`，并给出 grid/block 选择；
- [`tests/utils/test_activation.py`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/tests/utils/test_activation.py)：
  官方 FP16 测试覆盖 `dim=4096`，并以 `F.silu(first_half) * second_half` 为真值；
- [`flashinfer/trace/templates/activation.py`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/flashinfer/trace/templates/activation.py)：
  记录该算子的 LLaMA/Mistral FFN 用途和高层 correctness reference；
- [`tests/trace/fi_trace_out/silu_and_mul_h16384.json`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/tests/trace/fi_trace_out/silu_and_mul_h16384.json)：
  提供 BF16 trace workload，但固定宽度是 `16384`，不是本题的 `4096`；
- [PyTorch `torch.nn.SiLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html)：
  给出 `silu(x) = x * sigmoid(x)` 的公开数学定义；
- [`LICENSE`](https://github.com/flashinfer-ai/flashinfer/blob/103fcf862b902e100775ac781a0de36d067de3a4/LICENSE)：
  上述代码采用 Apache-2.0。

该 revision 的 trace 模板存在一处必须避免照抄的不一致：`Tensor` axes 把输入声明成
`[num_tokens, hidden_size]`，但同一文件的描述、reference、公开 API、CUDA kernel 和
测试都要求输入最后一维为 `2 * hidden_size`。本题以这些相互一致的实现证据为准，
固定输入 `[num_tokens, 8192]`、输出 `[num_tokens, 4096]`。

没有一份上游 case 同时固定 **BF16 + N=4096**：本题把官方 `dim=4096` 测试、BF16
trace、通用 dtype-dispatch API 和 CUDA kernel 组合为新的 benchmark specialization。
因此不得宣称 FlashInfer 已经直接提供本题的标准答案或完全相同的 benchmark 结果。

FlashInfer 不是可直接提交的独立 `.ptx` 标准答案：其公开 operator 包含 Python API、
JIT module、host launch 和 CUDA template。本题依据其语义与 workload 设计固定 raw-PTX
ABI。若实现阶段复制或改编官方源码，必须保留 Apache-2.0 attribution、准确文件、
commit 和改动说明。

## 3. Agent 可见接口

Agent 必须直接创建：

```text
/app/kernel.ptx
```

其中必须包含 visible entry：

```text
silu_gate_fwd_bf16_n4096(
    const bfloat16* input,
    bfloat16* output,
    uint32_t num_tokens)
```

这里的 C 风格声明只说明参数语义和顺序；实际提交必须是合法 PTX `.entry` 及对应
`.param` 声明。两个 device pointer 均按 64-bit 传递。

张量均为 contiguous row-major：

```text
input  : [num_tokens, 8192], bfloat16
output : [num_tokens, 4096], bfloat16
```

每行布局：

```text
input[t, 0:4096]    = gate
input[t, 4096:8192] = up
```

`input` 与 `output` 不得 alias。Verifier 保证 CUDA allocation 的基础地址至少 16-byte
对齐；每行字节数也为 16-byte 的整数倍。Candidate 可以使用向量 load/store，但这不是
正确性的必要条件。

### 3.1 数学语义

对每个 token `t` 和 `j = 0..4095`：

```text
g = float32(input[t, j])
u = float32(input[t, 4096 + j])

silu(g) = g / (1 + exp(-g))
        = g * sigmoid(g)

output[t, j] = bfloat16(silu(g) * u)
```

所有中间数学运算按至少 float32 精度解释，最终结果 round-to-nearest-even 到 BF16。
Candidate 可以使用 PTX 允许的近似指数实现，只要最终输出通过数值标准；不能把
`sigmoid(g) * u`、`relu(g) * u` 或 `silu(u) * g` 当成等价公式。

### 3.2 固定 launch ABI

Verifier 固定普通 CUDA launch：

```text
block dimensions:         (512, 1, 1)
grid dimensions:          (max(1, num_tokens), 1, 1)
dynamic shared memory:    0 bytes
programmatic dependency:  disabled
```

一个 block 对应一个 token。FlashInfer 在 BF16、`N=4096` 时的公开 launch 计算为：

```text
vec_size = 16 / sizeof(bfloat16) = 8
block.x  = min(N / vec_size, 1024) = 512
```

`num_tokens=0` 时仍启动一个 block，但 Candidate 必须在任何读写前返回。Candidate
不能依赖 Programmatic Dependent Launch 属性，因为 raw-PTX verifier 使用普通 Driver
API launch。

只有 `/app/kernel.ptx` 参与评分。目标设备是 NVIDIA B200，建议 PTX `.target sm_100`。
任务运行在 Direct-PTX 环境中：不提供 `nvcc`、NVRTC、NVVM 或 Triton，仅保留
`ptxas`、`cuobjdump` 和固定 PTX 文档。

## 4. 正确性标准

正确性是硬门槛。PTX 汇编、module 加载、entry 查找、launch、同步、数值、输入不变性
或内存安全任一失败，总 reward 为 `0`，不进入性能计时。

### 4.1 独立真值

Verifier 在 CPU 上把 BF16 输入转换为 float32，独立计算第 3.1 节公式，再按
round-to-nearest-even 转回 BF16。它不读取 Oracle 输出，也不逐行比较 Candidate 与
FlashInfer 源码。

可用 PyTorch 交叉检查该独立公式：

```python
gate = input[:, :4096].float()
up = input[:, 4096:].float()
expected = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
```

Candidate 和 task-owned 性能 Reference 必须分别与独立 CPU 真值比较。Reference 也要
检查，以防参数顺序、half layout、BF16 转换或编译产物出错。

初始比较容差：

```text
atol = 3e-2
rtol = 1e-2
```

FlashInfer 当前公开测试主要用 FP16 和 `1e-3/1e-3`，其 trace correctness 测试明确
指出 BF16 ULP 可超过 `1e-3`。上述 BF16 容差只是待校准起点，冻结前必须同时运行官方
BF16 external comparison 和故意错误的 kernel，确认它不会放过错误公式。输出中出现
NaN/Inf 一律拒绝。

### 4.2 Case 设计

完整输出检查的候选 token 数：

```text
num_tokens = 0, 1, 2, 3, 7, 31, 257
```

输入仅含有限 BF16 数，但至少覆盖：

- `gate=0`、正零/负零和 `up=0`；
- gate 位于 `-20, -8, -4, -1, 0, 1, 4, 8, 20` 附近；
- up 包含正负数、常数和不同尺度；
- 固定随机 normal 输入以及裁剪到有限范围的宽分布输入；
- 不同 token 使用不同内容，防止 Candidate 只计算第一行；
- 最后一个元素和非整 warp 边界对应的索引。

隐藏随机 seed 固定，保证同一 task version 可复现。性能规模在计时前也必须通过
correctness；可以抽查固定行以减少 CPU 成本，但必须覆盖首、中、末 token，且不能只
检查一小段 `N`。

### 4.3 内存安全和输入不变性

Verifier 至少检查：

- `output` 前后 canary 未变化；
- `input` 在 kernel 前后逐字节不变；
- output 预填非零 sentinel，确认全部 `4096 * num_tokens` 元素被写入；
- `num_tokens=0` 不发生访问；
- launch 和 synchronize 后 CUDA context 均无错误。

## 5. 性能评测边界

v0.1 候选规模：

```text
诊断但暂不计入聚合：num_tokens = 1
初始计分规模：        num_tokens = 32, 256, 4096
```

`num_tokens=1` 主要反映 kernel launch latency，区分度可能很小，因此先单独报告；完成
B200 重复计时后再决定是否加入总分。其余规模保持相同 entry、layout、block 和
specialization，只改变运行时 `num_tokens`，属于一道 task 的多个 verifier case。

本题每个输出元素的最低显式流量为：

```text
读取 gate BF16  = 2 bytes
读取 up BF16    = 2 bytes
写 output BF16  = 2 bytes
合计            = 6 bytes/output element
```

因此同时报告：

```text
effective_bandwidth = num_tokens * 4096 * 6 / median_kernel_time
```

SiLU 包含指数近似，简单的 TFLOPS 不能准确表达它的工作量；本题不用
`achieved_TFLOPS / theoretical_peak_TFLOPS` 作为主 reward。

### 5.1 三条路径

```text
CPU float32 formula
    只负责 correctness truth

Candidate /app/kernel.ptx
    Agent 直接提交，是真正被评分的对象

Task-owned Reference PTX
    与 Candidate 使用完全相同 ABI 和 launch，只负责性能基线
```

实现阶段从 FlashInfer Apache-2.0 `act_and_mul_kernel` 审阅或适配一份同 ABI CUDA
Reference，在可信出题环境中预编译为固定 PTX。Reference 必须禁用 PDL，并固定：

- FlashInfer revision 与适配 diff；
- CUDA/nvcc 版本、architecture 和 flags；
- CUDA source 与生成 PTX 的 SHA-256；
- Apache-2.0 attribution。

trial 环境不提供 `nvcc`，只加载预编译 Reference。FlashInfer 官方 operator 使用
`enable_pdl=False` 作为额外外部对照单独测量，但不直接计算 v0.1 reward，因为它仍
包含自己的 Python/JIT/host dispatch 边界。

### 5.2 计时与 reward 草案

沿用已有 PTX-Bench harness：同一 GPU/context、相同输入、CUDA Event、预热、自适应
batch、多批次中位数、Candidate/Reference 顺序交替，并保存全部原始时间。

对每个初始计分规模 `T`：

```text
r_T = reference_median_time_T / candidate_median_time_T

speedup_vs_reference = geometric_mean(r_32, r_256, r_4096)
performance          = clamp(speedup_vs_reference, 0, 1)
reward               = correctness * performance
```

同时保存未封顶 speedup、`T=1` 延迟和每个规模的 effective bandwidth。该公式只是待
校准草案；必须跑完 Oracle、慢但正确实现、错误实现和多个真实 Agent 后才能冻结。

## 6. 实现阶段的负例

至少准备以下作者侧 fixture；它们不提供给 Agent：

- 缺少 `silu_gate_fwd_bf16_n4096` entry；
- 只计算 `sigmoid(gate) * up`，漏掉 gate 乘法；
- 使用 `relu(gate) * up`；
- 把两半交换，计算 `silu(up) * gate`；
- 只计算 `silu(gate)` 或只复制 up；
- 只处理第一行、前几个 warp 或遗漏最后一个元素；
- 修改 input、越界写 output 或 `num_tokens=0` 仍访问内存；
- 对负 gate 的指数符号写反；
- 数值正确但使用标量 load/store并重复计算指数，用于确认 correctness 保持 1、性能分数
  明显下降。

## 7. 当前 task 结构

```text
tasks/ptx-silu-gate-forward-bf16-n4096/
├── instruction.md
├── task.toml
├── environment/
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── files/
├── solution/
│   ├── solve.sh
│   └── oracle.ptx
└── tests/
    ├── direct-ptx-policy-check
    ├── ptx-docs-manifest.toml
    ├── reference.ptx
    ├── test.sh
    ├── verify_hardware.py
    └── verify_and_benchmark.py
```

该目录由 `templates/direct-ptx-task-v0/` 的 scaffold generator 创建，再按本规格填充；
没有从其他已实现 task 复制可能遗留错误的 entry、shape、Reference 或隐藏 case。

## 8. 从规格到可运行任务的顺序

- [x] 审阅 FlashInfer API、CUDA kernel、launch、测试、revision 和许可证；
- [x] 识别并纠正上游 trace 输入 shape 元数据不一致；
- [x] 固定融合公式、BF16 layout、entry 参数和普通 launch ABI；
- [x] 规定 correctness truth、初始容差、cases、内存检查和性能协议；
- [x] 人工审阅本文并确认 `N=4096` 与计分 token 数；
- [x] 只运行 scaffold generator 创建 fail-closed task；
- [x] 实现独立 CPU truth、correctness verifier 和可审计 Oracle PTX；
- [x] 实现固定同 ABI 性能 Reference 与性能评分；
- [x] 完成本地 correctness 结构/合约检查；
- [x] 在 B200 完成 Oracle correctness-only feasibility；
- [x] 在 B200 运行漏乘 gate 和部分输出两个错误负例；
- [x] 在 B200 运行慢但正确实现并验证性能降分；
- [x] 完成 Harbor Singularity Oracle feasibility Job；
- [x] 完成同一 Oracle 的三个串行 Harbor 重复测量；
- [x] 完成一次 Codex/gpt-5.6-sol/high Agent feasibility；
- [x] 完成五次 Codex/gpt-5.6-sol/high pilot，结果 5/5；
- [ ] 运行开放权重模型，补充跨模型难度校准；
- [ ] 证据完整后再升级为 feasibility/validated/calibrated/frozen。

当前仍不把本题加入 frozen release manifest；Codex pilot 表明它适合作为 D1 easy
anchor，但必须补充开放权重模型校准，并补齐物理 GPU 标识后确认跨批次性能稳定性。
