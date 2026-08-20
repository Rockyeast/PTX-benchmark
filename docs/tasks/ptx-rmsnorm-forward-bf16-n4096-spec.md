# `ptx-rmsnorm-forward-bf16-n4096` 任务规格

**状态：Feasibility-validated Draft v0.1（2026-08-20）。Scaffold、最小 Oracle、
独立 verifier 和 B200 首跑已完成；性能 Reference、负例和真实 Agent 尚未冻结。**

## 1. 定位

这是 PTX-Bench 的第一道正式算子候选题。它只评测一个明确 specialization：

```text
RMSNorm plain forward + BF16 input/output + FP32 weight + N=4096
```

暂不包含 backward、bias、residual、prenorm、输出 `rstd`、其他 dtype 或其他 `N`。
这些属于不同接口和优化问题，后续若需要应单独增加 task variant，而不是悄悄扩大
本题范围。

选择 `N=4096` 的依据是：

- 它是 INT21 RMSNorm-B200 默认 benchmark matrix 中的真实模型宽度；
- INT21 对 BF16、plain forward、`N=4096` 使用专门 fast path；
- 固定 `N` 后，Agent 只优化一个 kernel，不必同时实现 INT21 host 端的 shape/dtype
  动态分发。

## 2. 上游依据与许可证

设计参考官方开源仓库：

```text
https://github.com/Int21-AI/RMSNorm-B200
commit: 6b0a8beeaa306b1a55c0dc5c5e03dafeda914c5f
license: MIT
```

该仓库不是一份可直接提交的独立 `.ptx`：它是 PyTorch CUDA extension，内部包含
多个 CUDA C++/inline PTX kernel，并由 host 代码根据 dtype、shape、residual 等条件
选择实现。

与本题直接有关的上游事实：

- `benchmarks/benchmark_rmsnorm.py` 的默认矩阵包含 `(M,N)=(32768,4096)`；
- `rmsnorm/rmsnorm.py::rmsnorm_ref` 给出独立 PyTorch 数学参考；
- `rmsnorm/rmsnorm_cuda.cu::rmsnorm_fwd` 对 BF16、无 residual、无 `rstd`
  的 `N=4096` fast path 使用 `128` threads per row 和 `128` threads per block，
  即每个 block 处理一行；
- 官方 BF16 forward 测试使用 `atol=1e-1, rtol=1e-3`。

如果实现阶段复制或改编 INT21 代码，必须在 task 内保留其 MIT copyright/license
notice 和准确来源；仅依据接口重新实现则无需复制源文件。

## 3. Agent 可见接口

Agent 必须创建：

```text
/app/kernel.ptx
```

其中必须包含 visible entry：

```text
rmsnorm_fwd_bf16_n4096(
    const bfloat16* x,
    const float* weight,
    bfloat16* output,
    uint32_t M,
    float eps)
```

张量均为 contiguous row-major：

```text
x      : [M, 4096], bfloat16
weight : [4096],    float32
output : [M, 4096], bfloat16
```

对每行 `m`：

```text
mean_sq[m] = sum(float32(x[m, j]) ** 2 for j in 0..4095) / 4096
rstd[m]    = rsqrt(mean_sq[m] + eps)
output[m,j] = bfloat16(float32(x[m,j]) * rstd[m] * weight[j])
```

Verifier 固定启动方式：

```text
block dimensions: (128, 1, 1)
grid dimensions:  (max(1, M), 1, 1)
dynamic shared memory: 1024 bytes
```

约束：

- 一个 block 对应一行；`M=0` 时不得读写任何 tensor 元素；
- 地址为 64 bit，`M` 为 `uint32_t`，`eps` 为 `float32`；
- 输入 `x` 和 `weight` 不得修改，也不得越界读写；
- 只评分 `/app/kernel.ptx`；目标设备为 NVIDIA B200，建议 `.target sm_100`。

这里固定 launch configuration 是 raw PTX ABI 的一部分。PTX 文件自身不能替 host
程序决定 grid、block 和 dynamic shared memory。

## 4. 正确性标准

正确性是硬门槛。编译、加载、entry 查找、launch、同步、数值或内存安全任一失败，
总 reward 都为 `0`，不进入性能计时。

### 4.1 独立真值

真值由 task 自己按下面的高层公式计算，不读取 Oracle 或 INT21 的输出：

```python
x_f32 = x.float()
expected = x_f32 / torch.sqrt(x_f32.square().mean(dim=-1, keepdim=True) + eps)
expected = (expected * weight.float()).to(torch.bfloat16)
```

这与 INT21 的公开 `rmsnorm_ref` 语义一致，但属于独立计算路径。Candidate、task
内性能 Reference 和可选 INT21 对照都必须各自与该真值比较。

### 4.2 Case 设计

完整输出检查至少覆盖：

```text
M = 0, 1, 2, 3, 17, 199
eps = 1e-6，并加入至少一个不同的合法正数 eps
```

输入至少包含：固定随机数、全零行、常数行，以及数值尺度不同但不含 NaN/Inf 的
有限 BF16 数据。隐藏随机 seed 固定，保证同一 task version 可复现。

性能规模在计时前也检查正确性；为控制额外显存和耗时，可对固定的首、中、末行做
独立 float32 PyTorch 检查。

BF16 比较初始采用 INT21 的公开容差：

```text
atol = 1e-1
rtol = 1e-3
```

同时显式拒绝 Candidate 产生的 NaN/Inf。实现阶段要用若干故意低精度的错误 kernel
校准容差；如果上述容差无法区分明显错误，再收紧或增加逐行误差指标，而不是直接
宣称规格已冻结。

### 4.3 内存安全

Verifier 至少检查：

- `output` 前后 canary 未变化；
- `x` 与 `weight` 在 kernel 前后不变；
- `M=0` 不发生非法访问；
- CUDA context 在 launch 和 synchronize 后均无错误。

## 5. 性能评测边界

v0.1 候选规模：

```text
(M,N) = (1024,4096), (8192,4096), (32768,4096)
```

最后一个是 INT21 的官方默认 benchmark shape；前两个用于观察同一 kernel 在不同
row 数量下的启动和占用行为。B200 实测后才冻结规模。

### 5.1 三条实现路径不能混为一谈

```text
PyTorch float32 formula
    只负责 correctness truth，不负责性能分数

Candidate /app/kernel.ptx
    Agent 提交，是真正被评分的对象

Reference
    task 维护的同 ABI、同 launch configuration CUDA/PTX 实现，只负责性能基线
```

INT21 官方 extension 作为额外外部对照运行并记录，但 v0.1 不直接拿它计算 reward。
原因是 INT21 包含 PyTorch binding、host dispatch 和多种 kernel 路径，而本题是一个
固定 raw-PTX entry；两者接口边界不同。先将 Candidate 与同 ABI Reference 公平对比，
再单独报告 INT21 operator timing，含义更清楚。

实现阶段优先从 INT21 MIT 源码中审阅并适配 `N=4096` fast path，生成 task-owned
Reference；适配后仍必须通过独立 PyTorch 真值，不能因为来源是官方仓库就预设正确。

### 5.2 计时

沿用 `ptx-vector-add` 已验证的方法：

- Candidate 与 Reference 在同一 GPU/context、相同输入和固定 launch 下计时；
- 编译、JIT、分配和数据拷贝不计入 kernel 时间；
- CUDA Event 计时，先 warmup，再自适应累计到足够时长；
- 每个规模测多个 batch 取中位数，Candidate/Reference 顺序交替；
- 保存所有原始 batch 时间、GPU 物理编号、SIF checksum 和 task checksum。

INT21 外部对照必须明确记录其 commit、构建参数和测量边界，不能把 operator timing
伪装成同 ABI kernel timing。

## 6. Reward 草案

每个性能规模：

```text
r_M = reference_median_time_M / candidate_median_time_M
```

初始公式：

```text
speedup_vs_reference = geometric_mean(r_M)
performance          = clamp(speedup_vs_reference, 0, 1)
reward               = correctness * performance
```

同时保存未封顶的 `speedup_vs_reference` 和每个规模原始时间。该公式只是继承已跑通
的 verifier 机制，不代表 RMSNorm 最终评分已经合理。必须先跑 Oracle、Reference、
慢但正确实现、错误实现和多个真实 Agent，再决定是否冻结。

## 7. 预计 task 结构（尚未创建）

```text
tasks/ptx-rmsnorm-forward-bf16-n4096/
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

其中：

- `instruction.md`：Agent 可见的接口和提交要求；
- `solution/solve.sh`：Oracle 创建标准 Candidate PTX；
- `reference.cu`：隐藏、同 ABI 的性能参考，不提供给 Agent；
- `verify_and_benchmark.py`：独立真值、内存安全、计时和 reward；
- `REFERENCE.md`：INT21 来源、commit、许可证和具体改编范围。

## 8. 实现前必须确认

- [ ] 人工确认是否接受首题只做 `N=4096` specialization；
- [ ] 人工确认 entry 参数顺序和固定 block/grid/shared-memory；
- [ ] 从 INT21 fast path 确定可适配范围并保留 MIT attribution；
- [x] 用 Harbor CLI scaffold 标准 task 目录；
- [x] 本地完成 Python/Bash 结构与 static 检查；
- [ ] 在 `catalyst-b200` 做 Oracle、Reference、负例和真实 Agent 校准；
- [ ] 单次 Harbor/远端进程始终低于 2.5 小时。

## 9. 首次可行性结果

2026-08-20 在 `catalyst-b200` 的 NVIDIA B200 GPU 0 上运行 Harbor Oracle：

```text
job: ptx-rmsnorm-feasibility-oracle-1
runtime: 18 s
correctness: 1.0
speedup_vs_reference: 1.000141864
reward: 1.0
```

六个完整正确性 case（`M=0,1,2,3,17,199`）通过；性能 shape 原始中位数：

| M | Candidate ms | Reference ms | Reference/Candidate |
|---:|---:|---:|---:|
| 1024 | 0.009774 | 0.009775 | 1.000161 |
| 8192 | 0.049412 | 0.049409 | 0.999948 |
| 32768 | 0.170458 | 0.170513 | 1.000317 |

Candidate 与 Reference 当前采用同一个简单算法，所以接近 `1` 是预期校验结果，只能
说明 ABI、BF16 数值检查、三组规模和 CUDA Event 计时链路可运行。它不能说明题目有
模型区分度，也不能说明当前实现达到 INT21 性能。

原始结果保存在：

```text
jobs/ptx-rmsnorm-feasibility-oracle-1/
```
