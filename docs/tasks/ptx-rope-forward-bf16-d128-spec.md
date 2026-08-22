# `ptx-rope-forward-bf16-d128` 任务规格

**状态：Feasibility-validated v0.2（2026-08-21）。任务、Oracle、CPU truth、同 ABI
Reference 和性能 harness 已实现，并已在 B200 上通过 Harbor Oracle；负例矩阵、重复
计时和真实 Agent 校准尚未完成，因此还不是 verifier-validated 或 frozen。**

## 1. 定位

这是 PTX-Bench 第一批 pilot 的 P2，属于 **Position / Data Transform** 家族。它只评测
一个明确 specialization：

```text
out-of-place Q/K RoPE forward
+ BF16 input/output
+ precomputed FP32 cos/sin cache
+ NeoX split-half layout
+ query_heads=32, kv_heads=8
+ head_dim=rotary_dim=128
```

本题不包含 backward、in-place 更新、interleaved/GPT-J layout、partial rotary、不同
head dimension、paged KV cache 或动态 head 数。这些改变会影响语义、内存布局或
kernel 拓扑，若以后需要，应拆成独立 task variant。

选择预计算 cache 版本，而不是要求 kernel 现场计算 `pow/sin/cos`，原因是：

- 它与 FlashInfer 面向 SGLang/vLLM 的公开接口和 trace workload 直接对应；
- Agent 的主要优化问题变成位置索引、cache 复用、BF16 转换和 Q/K 融合访存；
- 单个固定 raw-PTX entry 就能完整表达，不需要把数学库实现差异混进性能分数。

## 2. 上游依据与许可证

设计依据固定到 FlashInfer 官方仓库的以下 revision：

```text
repository: https://github.com/flashinfer-ai/flashinfer
commit:     5366177a074e27df7db527f5b744c77dfd748484
license:    Apache-2.0
```

直接审阅的官方材料：

- [`flashinfer/trace/templates/rope.py`](https://github.com/flashinfer-ai/flashinfer/blob/5366177a074e27df7db527f5b744c77dfd748484/flashinfer/trace/templates/rope.py)：
  定义 BF16 Q/K、FP32 cache、int32 position IDs 和 D=128 trace；
- [`tests/trace/fi_trace_out/rope_cos_sin_cache_d128.json`](https://github.com/flashinfer-ai/flashinfer/blob/5366177a074e27df7db527f5b744c77dfd748484/tests/trace/fi_trace_out/rope_cos_sin_cache_d128.json)：
  记录 `32` 个 query heads、`8` 个 KV heads 和 flattened input shape；
- [`flashinfer/rope.py`](https://github.com/flashinfer-ai/flashinfer/blob/5366177a074e27df7db527f5b744c77dfd748484/flashinfer/rope.py)：
  公开的 cos/sin-cache API 及默认 NeoX 语义；
- [`include/flashinfer/pos_enc.cuh`](https://github.com/flashinfer-ai/flashinfer/blob/5366177a074e27df7db527f5b744c77dfd748484/include/flashinfer/pos_enc.cuh)：
  CUDA 实现中的 split-half 配对和 cache 读取；
- [`benchmarks/bench_rope.py`](https://github.com/flashinfer-ai/flashinfer/blob/5366177a074e27df7db527f5b744c77dfd748484/benchmarks/bench_rope.py)：
  BF16、32/8 heads、D=128、base=500000 的官方性能 workload；
- [`tests/trace/reference_utils.py`](https://github.com/flashinfer-ai/flashinfer/blob/5366177a074e27df7db527f5b744c77dfd748484/tests/trace/reference_utils.py)：
  官方 trace correctness 使用的 RoPE 容差。

FlashInfer 不是可直接塞进本题的单一 `.ptx` 标准答案：它包含 Python API、host
dispatch、JIT/AOT extension 和不同 kernel 变体。本题依据其公开语义和 workload
设计固定 ABI；若实现阶段复制或改编官方源码，必须保留 Apache-2.0 attribution 和
准确的文件、commit、改动范围。

## 3. Agent 可见接口

Agent 必须直接创建：

```text
/app/kernel.ptx
```

其中必须包含 visible entry：

```text
rope_forward_bf16_d128(
    const bfloat16* q,
    const bfloat16* k,
    const int32_t* positions,
    const float* cos_sin_cache,
    bfloat16* q_out,
    bfloat16* k_out,
    uint32_t num_tokens)
```

这里的 C 风格声明用于说明参数语义和顺序；实际提交必须是合法 PTX `.entry` 及对应
`.param` 声明。所有 device pointer 均按 64-bit 传递。

张量均为 contiguous row-major：

```text
q             : [num_tokens, 32, 128], bfloat16
k             : [num_tokens,  8, 128], bfloat16
positions     : [num_tokens],          int32
cos_sin_cache : [8192, 128],           float32
q_out         : [num_tokens, 32, 128], bfloat16
k_out         : [num_tokens,  8, 128], bfloat16
```

输入和输出不得 alias。`positions[t]` 保证位于 `[0, 8191]`；position 可以重复、乱序，
不保证等于 token 下标。

### 3.1 数学语义

对每个 token `t`、每个 Q 或 K head，以及 `d = 0..63`：

```text
p = positions[t]
c = cos_sin_cache[p, d]
s = cos_sin_cache[p, 64 + d]

out[..., d]      = bfloat16(float32(x[..., d])      * c
                          - float32(x[..., 64 + d]) * s)
out[..., 64 + d] = bfloat16(float32(x[..., 64 + d]) * c
                          + float32(x[..., d])      * s)
```

这就是 NeoX/split-half layout：前 64 维与后 64 维一一配对，不是相邻偶数/奇数配对。
同一公式分别作用于 `q -> q_out` 和 `k -> k_out`。

Verifier 根据固定 seed 为被测 position 构造确定性的非平凡 cache 值。它不要求 cache
来自某个固定 `theta`，以免错误实现绕过输入并在 kernel 内重建特定 cos/sin 表。正式
模型常用的 cache 可由下面的公式生成，但 Candidate 的契约始终是读取传入值：

```text
theta = 500000
angle[p,d] = p * theta ** (-2*d/128),  d = 0..63
cache[p,0:64]   = cos(angle[p,:])
cache[p,64:128] = sin(angle[p,:])
```

`theta` 只负责离线构造输入 cache，不是 kernel ABI 参数。Candidate 必须使用传入的
cache，不能假设它永远可由固定公式现场重建。

### 3.2 固定 launch ABI

Verifier 固定启动方式：

```text
block dimensions: (256, 1, 1)
grid dimensions:  (max(1, num_tokens), 1, 1)
dynamic shared memory: 512 bytes
```

一个 block 对应一个 token。512 bytes 足以缓存该 token 的 64 个 FP32 cos 和 64 个
FP32 sin；Candidate 可以使用，也可以不使用这块 shared memory。`num_tokens=0` 时
block 仍会启动一次，但不得读取或写入任何 tensor 元素。

只有 `/app/kernel.ptx` 参与评分。目标设备是 NVIDIA B200，建议 PTX `.target sm_100`。
任务运行在 direct-PTX SIF 中：不提供 `nvcc`、NVRTC 或 CUDA 高层编译前端，仅保留
`ptxas` 等 PTX 检查工具。

## 4. 正确性标准

正确性是硬门槛。PTX 编译、module 加载、entry 查找、launch、同步、数值或内存安全
任一失败，总 reward 为 `0`，不进入性能计时。

### 4.1 独立真值

Verifier 用 CPU/Python float32 按第 3.1 节公式独立计算 expected，再按 BF16 舍入。
它不读取 Oracle 输出，也不把 Candidate 的 PTX 代码与 FlashInfer 源码逐行比较。

Candidate 和 task-owned 性能 Reference 都必须分别与该独立真值比较。Reference 也要
检查，是为了防止接口适配、参数顺序、layout 或编译产物版本出错；“来自官方项目”
不自动等于“符合本题固定 ABI”。

初始数值容差沿用 FlashInfer trace 的 RoPE 设置：

```text
atol = 1e-2
rtol = 1e-2
```

同时显式拒绝输出 NaN/Inf。冻结前必须用错误实现校准，确认该容差不会放过明显错误。

### 4.2 Case 设计

完整输出检查的候选 token 数：

```text
num_tokens = 0, 1, 2, 3, 7, 31, 257
```

至少覆盖：

- `position=0`，用于验证 identity rotation；
- 重复 position 和非单调 position，防止实现错误使用 token 下标；
- 接近 cache 上界的 `position=8191`；
- 固定随机 BF16 Q/K；
- 全零、常数和正负混合输入；
- Q 有 32 heads、K 只有 8 heads，防止实现把两者 shape 混为一谈。

隐藏随机 seed 固定，保证同一 task version 可复现。性能规模在计时前也必须通过
correctness 检查；若全量 CPU expected 成本过高，可固定检查首、中、末 token，但
不能只检查一个 head。

### 4.3 内存安全和输入不变性

Verifier 至少检查：

- `q_out`、`k_out` 前后 canary 未变化；
- 输入 `q`、`k`、`positions` 和 `cos_sin_cache` 在 kernel 前后不变；
- Q 与 K 两个输出均被完整写入，不能只实现其中一个；
- `num_tokens=0` 不发生非法访问；
- launch 和 synchronize 后 CUDA context 均无错误。

## 5. 性能评测边界

v0.1 候选规模：

```text
num_tokens = 256, 4096, 32768
```

它们保持相同 ABI 和同一 kernel specialization，只改变运行时参数 `num_tokens`，因此
属于一道 task 的多个 verifier case，不需要拆成三道 task。最终规模需在 B200 实测后
冻结；当前数值只覆盖小、中、大工作量，不构成性能结论。

### 5.1 三条路径

```text
CPU/Python float32 formula
    只负责 correctness truth

Candidate /app/kernel.ptx
    Agent 直接提交，是真正被评分的对象

Task-owned Reference PTX
    与 Candidate 使用完全相同 ABI 和 launch，只负责性能基线
```

实现阶段应编写可审计的同 ABI CUDA Reference，并在出题环境中一次性编译为固定
`reference.ptx`；trial 内不放 `nvcc`，Verifier 只加载预编译 PTX。Reference 的源码、
编译器版本、flags、PTX SHA-256、来源和许可证必须记录。

FlashInfer 官方 operator 可作为额外外部对照单独测量，但不直接计算 v0.1 reward。
原因是它的 Python/host dispatch 和 launch 边界与本题固定 raw-PTX ABI 不同。外部
operator timing 与同 ABI kernel timing 必须分开报告。

### 5.2 计时与 reward 草案

沿用已验证的 PTX-Bench 计时框架：同一 GPU/context、相同输入、CUDA Event、预热、
自适应 batch、多批次中位数、Candidate/Reference 顺序交替，并保存全部原始时间。

每个性能规模：

```text
r_T = reference_median_time_T / candidate_median_time_T
```

初始公式：

```text
speedup_vs_reference = geometric_mean(r_T)
performance          = clamp(speedup_vs_reference, 0, 1)
reward               = correctness * performance
```

同时保存未封顶 speedup。该公式只是待校准草案；必须跑完 Oracle、慢但正确实现、错误
实现和多个真实 Agent 后，才能判断 RoPE 的规模和评分是否有区分度。

## 6. 实现阶段的负例

至少准备以下作者自测 fixture；它们不提供给 Agent：

- 缺少 `rope_forward_bf16_d128` entry；
- 错把 split-half 写成相邻偶/奇维配对；
- 忽略 `positions`，直接使用 token 下标或始终使用 position 0；
- Q 正确但 K 未写，或混淆 Q/K head 数；
- 修改输入或越界写输出；
- 只处理前几个 head，或错误处理 BF16 转换；
- 数值正确但故意重复无效工作，用于验证性能分数可下降而 correctness 仍为 1。

## 7. 当前 task 结构

```text
tasks/ptx-rope-forward-bf16-d128/
├── instruction.md
├── task.toml
├── environment/
│   ├── Dockerfile
│   ├── docker-compose.yaml
│   └── files/
│       ├── direct-ptx-policy-check
│       ├── setup.sh
│       └── ptx-docs/
├── solution/
│   ├── solve.sh
│   └── oracle.ptx
└── tests/
    ├── test.sh
    ├── verify_and_benchmark.py
    └── reference.ptx
```

其中 `instruction.md` 是从本文提炼出的 Agent 可见说明；本文则主要给 task 作者和
审阅者使用。不会把来源调研、隐藏 case、负例或 Reference 实现细节全部暴露给 Agent。

## 8. 从草案到可运行任务的顺序

- [x] 审阅 FlashInfer 官方语义、workload、测试、revision 和许可证；
- [x] 确认 operation、NeoX layout、Q/K heads、cache 和 entry 参数顺序；
- [x] 确认固定 block/grid/shared-memory 与候选性能规模；
- [x] scaffold 标准 Harbor task 目录；
- [x] 实现独立 CPU truth、Oracle PTX 和同 ABI Reference；
- [x] 本地完成 Python/Bash/static 检查；
- [x] 在 `catalyst-b200` 完成一次 Harbor Oracle feasibility run；
- [ ] 跑负例、重复计时和真实 Agent；
- [ ] 全部校准完成后再把状态升级为 validated/calibrated/frozen。

本题已经越过 source-reviewed draft，但仍不应加入 release dataset，也不能把一次
Oracle 的计时或当前评分公式描述成已经冻结的 benchmark 结论。
