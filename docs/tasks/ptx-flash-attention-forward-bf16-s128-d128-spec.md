# `ptx-flash-attention-forward-bf16-s128-d128` 任务规格

**状态：Agent-calibrated development v0.3（2026-08-22）。B200 Oracle、错误负例、
慢正确实现、Harbor Job 和五次 Codex/high pilot 已完成；当前 5/5 通过且固定 Reference
明显偏弱，因此尚未达到 release-ready。**

## 1. 定位

这是 PTX-Bench 第一批 pilot 的第一道 **Attention / Sequence State** D4 题。它只评测
下面这个 specialization：

```text
single-request causal self-attention forward
+ BF16 Q / K / V
+ BF16 output
+ sequence length S=128
+ head dimension D=128
+ MHA: num_q_heads == num_kv_heads
+ NHD contiguous layout
+ NVIDIA B200 / sm_100
```

本题不包含 batch、GQA、paged KV、变长序列、RoPE、ALiBi、sliding window、dropout、
logits soft cap、LSE 输出、backward、FP8/NVFP4 或 workspace。它们会改变接口、索引、
输出或 kernel 拓扑，应作为独立 task variant。

运行时可以改变 `num_heads`，因为所有规模仍使用相同 entry、相同 `S=128/D=128`、相同
NHD 语义和相同 launch ABI。因此不同 head 数是同一道 task 的 verifier cases。

相较于已经完成的 Softmax，本题要求 Agent 在一个 PTX entry 中同时完成：

- BF16 QK 点积和 FP32 累加；
- causal mask；
- 数值稳定的整行 Softmax；
- BF16 V 加权聚合和 FP32 累加；
- BF16 输出转换；
- query、key、head 和 feature 四维索引；
- block 内跨阶段同步和数据复用。

## 2. 上游依据与许可证

设计依据固定到 FlashInfer 官方仓库的以下 revision：

```text
repository: https://github.com/flashinfer-ai/flashinfer
commit:     fb28d7242b3506a2348265962041acc1fb56cca4
license:    Apache-2.0
```

直接审阅的材料：

- [`flashinfer/prefill.py`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/prefill.py)：
  single-request prefill 的公开接口、NHD/HND layout、causal 和 scale 语义；
- [`flashinfer/trace/templates/attention.py`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/trace/templates/attention.py)：
  独立 FP32 `QK^T -> causal mask -> softmax -> PV` reference 和 BF16 输入生成；
- [`tests/attention/test_blackwell_fmha.py`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/tests/attention/test_blackwell_fmha.py)：
  B200/SM100 BF16 FMHA 的独立 PyTorch reference、causal mask 和 shape 覆盖；
- [`tests/trace/test_single_prefill_reference_correctness.py`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/tests/trace/test_single_prefill_reference_correctness.py)：
  `qo_len=128`、`head_dim=128` 等真实 trace reference 检查及 `1e-2` 级容差依据；
- [`include/flashinfer/attention/prefill.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/include/flashinfer/attention/prefill.cuh)：
  tiled QK、causal masking、online Softmax、PV 和 BF16/FP32 数据路径；
- [`csrc/single_prefill.cu`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/csrc/single_prefill.cu)：
  binding 如何提取 `qo_len`、`kv_len`、head 数、stride 和 dispatch 参数；
- [`LICENSE`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/LICENSE)：
  上述代码采用 Apache-2.0。

FlashInfer 提供真实算子语义、测试形状和高性能算法依据，但不是本题可直接提交的 raw-PTX
答案。上游会在 FA2、Blackwell CUTLASS、CuTe DSL 等多个 backend 间 dispatch，并支持
多种 shape、layout 和额外功能；本题重新冻结单 entry、普通 launch 和固定 shape。

实现阶段可以依据上述结构编写 task-owned CUDA Reference，并在可信出题环境中预编译为
PTX，但必须保存作者源码、编译器版本、flags 和生成 PTX 的 SHA-256。Reference 只负责
性能基线，仍必须独立通过 correctness truth。

## 3. Agent 可见接口

Agent 必须直接创建：

```text
/app/kernel.ptx
```

PTX 文件必须包含 visible entry：

```text
flash_attn_fwd_bf16_s128_d128(
    const bf16* q,
    const bf16* k,
    const bf16* v,
    bf16* output,
    uint32_t num_heads)
```

这里的 C 风格声明只表示参数顺序和语义。实际提交必须使用合法 PTX `.entry` 和
`.param`；四个 device pointer 按 64-bit 传递，`num_heads` 为 32-bit unsigned。

张量布局固定为 FlashInfer 的 NHD 语义：

```text
q      : [128, num_heads, 128], bfloat16, contiguous
k      : [128, num_heads, 128], bfloat16, contiguous
v      : [128, num_heads, 128], bfloat16, contiguous
output : [128, num_heads, 128], bfloat16, contiguous
```

元素 `(token, head, feature)` 的线性索引为：

```text
((token * num_heads + head) * 128 + feature)
```

四个 tensor 不 alias。Allocation 基地址至少 16-byte 对齐；每个 head vector 含 128 个
BF16，即 256 bytes，也保持 16-byte 对齐。Candidate 不得修改 Q/K/V，必须写完整 output，
且不得访问任一 allocation 之外的内存。

### 3.1 数学语义

令：

```text
S = 128
D = 128
scale = 1 / sqrt(128)
```

对每个 `head h` 和 query position `i`：

```text
score[i,j] = scale * sum_d(float(q[i,h,d]) * float(k[j,h,d]))
              for j <= i

score[i,j] = -infinity
              for j > i

m       = max_j(score[i,j])
p[i,j]  = exp(score[i,j] - m) / sum_t(exp(score[i,t] - m))

acc[d]  = sum_j(p[i,j] * float(v[j,h,d]))
output[i,h,d] = bf16(acc[d])
```

QK 点积、Softmax 和 PV 累加均按至少 FP32 解释。输入是有限 BF16；NaN 和 infinity 不在
输入 contract 中。Candidate 可以使用近似指数，只要最终 BF16 output 满足容差。

### 3.2 固定 launch ABI

Verifier 固定使用普通 CUDA Driver API：

```text
block dimensions:         (128, 1, 1)
grid dimensions:          (max(1, num_heads * 128), 1, 1)
dynamic shared memory:    0 bytes
programmatic dependency:  disabled
```

每个 block 对应一个 `(query_position, head)`。当 `num_heads > 0` 时，推荐映射是：

```text
query_position = blockIdx.x / num_heads
head           = blockIdx.x % num_heads
```

`num_heads=0` 时 verifier 仍启动一个 block；Candidate 必须在除法及任何全局内存访问前
返回。Candidate 可以声明静态 `.shared` 存储 score、probability 和 reduction 临时值，
但 verifier 不提供 dynamic shared memory。

目标设备为 NVIDIA B200。普通 PTX 可以使用 `.target sm_100`；若依赖 architecture-
accelerated `sm_100a` 指令，必须在 PTX target 和验证记录中明确声明。

任务采用 Direct-PTX 环境：允许固定版本 `ptxas`、`cuobjdump` 和离线 PTX ISA 文档；没有
`nvcc`、NVRTC、NVVM 或 Triton，也不能下载安装高层到 PTX 的工具链。

## 4. 正确性标准

正确性是硬门槛。PTX 汇编、module load、entry、launch、同步、数值、输入不变性或内存
安全任一失败，总 reward 为 `0`，不进入性能计时。

### 4.1 独立真值

Hidden verifier 在 CPU 上独立执行稳定 attention 公式：

```text
FP32 QK dot
-> multiply by 1/sqrt(128)
-> causal mask
-> subtract row maximum
-> FP64/FP32 exp and denominator
-> FP32 PV accumulation
-> round to BF16
```

Candidate 与性能 Reference 都分别和 CPU expected 比较，不比较两份 PTX 的代码，也不把
Oracle 输出当 truth。实现阶段应再用固定小 case 与 PyTorch FP32 SDPA 交叉检查 CPU
公式的维度、mask 方向和 scale。

### 4.2 初始数值门槛

FlashInfer 上游 trace 对 BF16 single prefill 使用 `1e-2` 级容差。本题第一版考虑直接 PTX
近似指数和不同累加顺序，暂定对解码后的 BF16 output 检查：

```text
abs(candidate - expected) <= 2e-2 + 2e-2 * abs(expected)
all values finite
```

容差必须通过 Oracle、Reference、慢正确实现和数值错误负例校准后才能冻结。若明显错误
实现也能通过，应增强结构化输入，不能单纯放宽容差。

### 4.3 初始 correctness cases

| `num_heads` | 输入类型 | 主要检查 |
|---:|---|---|
| 0 | 空逻辑输入 | 一个 block launch 但不得除零或访问内存 |
| 1 | Q/K 全零，V 为可预测梯度 | causal prefix average、BF16 输出 |
| 1 | future K/V 极大，past 正常 | future token 不得影响 earlier query |
| 3 | 随机小值和不同 head pattern | NHD stride、非二次幂 head 数 |
| 8 | 大正/负 QK logits | 必须稳定 Softmax，不能直接指数溢出 |
| 16 | 随机值、dominant key | 多 block、head 隔离、完整输出 |

每个非空 case 都检查：

- 全部 token、head 和 feature；
- Q/K/V launch 前后逐 bit 不变；
- output 前后 canary；
- CUDA launch/synchronize 无错误；
- 不跨 head 或跨 tensor；
- 多个隐藏 seed 可复现。

## 5. 性能评测草案

### 5.1 三条独立路径

```text
CPU stable attention formula
    只负责 correctness truth

Candidate /app/kernel.ptx
    Agent 直接提交

Task-owned Reference PTX
    与 Candidate 使用相同 entry、参数和 launch ABI
    只负责同卡性能基线，也必须通过 CPU truth
```

### 5.2 计时与 reward

初始性能规模：

```text
H = num_heads in {1, 8, 32}
```

使用同一 GPU/context、相同输入、CUDA Event、预热、自适应 iteration、多批次中位数和
Candidate/Reference 交替顺序：

```text
r_H                  = reference_median_time_H / candidate_median_time_H
speedup_vs_reference = geometric_mean(r_1, r_8, r_32)
performance          = clamp(speedup_vs_reference, 0, 1)
reward               = correctness * performance
```

同时保存未封顶 speedup、绝对时间和 causal attention 的逻辑 FLOP/s。对每个 head，
`T = S*(S+1)/2` 个 causal QK/PV pair，每个 pair 的 QK 与 PV 各含 `D` 个 multiply-add：

```text
logical_flops(H) = 4 * H * T * D
                 = 4 * H * (128 * 129 / 2) * 128
```

该数值只描述数学接口工作量，不等于实际指令数或 DRAM 流量。最终性能 shape 和公式必须
经过 Reference 稳定性、慢正确实现和真实 Agent pilot 后才能冻结。

## 6. 实现阶段负例

至少准备以下作者侧 fixture 或等价隐藏 case：

- 缺少规定 entry；
- 完全不做 causal mask；
- 把 causal 方向写反；
- 忘记 `1/sqrt(128)` scale；
- 直接 `exp(score)`，大 logits 下溢出/溢出；
- 只输出 V 的 prefix average，忽略 Q/K；
- 只处理第一个 head 或错误使用 HND stride；
- 只写部分 feature；
- 修改 Q/K/V 或越界写 output；
- `num_heads=0` 仍除法或访问内存；
- 数值正确但 thread 0 串行完成全部 attention，用来检查连续性能分数。

## 7. 从规格到可运行 task

- [x] 固定 operation、direction、dtype、layout、S、D 和硬件目标；
- [x] 审阅 FlashInfer API、binding、trace reference、B200 测试、kernel、revision 和许可证；
- [x] 说明上游完整 FMHA 与本题固定 raw-PTX ABI 的差异；
- [x] 固定 entry 参数、张量布局和普通 launch ABI；
- [x] 草拟独立 truth、容差、cases、Reference 和 reward；
- [ ] 人工审阅本规格；
- [x] 生成 fail-closed scaffold 并替换全部 TODO；
- [x] 编写 Agent-facing instruction；
- [x] 实现独立 CPU correctness verifier；
- [x] 实现同 ABI CUDA Reference 并生成 Oracle/Reference PTX；
- [x] 添加 non-causal、partial-output 错误负例和慢正确实现；
- [x] 在 B200 完成 Oracle、负例、五次性能重复和 Harbor Oracle Job；
- [x] 运行五次 Codex/gpt-5.6-sol/high，校准 600 秒预算；结果为 5/5；
- [ ] 以更强同 ABI Reference 替换当前 feasibility 性能基线；
- [ ] 在开放权重模型和严格 Agent 网络隔离下完成 release 校准。

在上述验证完成前，本题不得进入 frozen release manifest。
