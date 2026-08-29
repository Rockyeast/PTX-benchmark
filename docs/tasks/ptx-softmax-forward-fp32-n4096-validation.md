# `ptx-softmax-forward-fp32-n4096` 验证记录

**日期：2026-08-22。当前结论：任务已完成 feasibility 和首轮 Agent 校准，尚未冻结为
release。**

## 1. 固定对象

```text
task:       ptx-softmax-forward-fp32-n4096
hardware:   NVIDIA B200 / sm_100
entry:      softmax_fwd_fp32_n4096
input:      [num_rows, 4096] FP32
output:     [num_rows, 4096] FP32
block:      (1024, 1, 1)
grid:       (max(1, num_rows), 1, 1)
dyn. smem:  0 bytes
```

本记录只证明当前开发版本在指定环境中的可行性和初步区分能力，不代表正式数据集已经
冻结。正式 release 仍需固定 task revision、Harbor/backend 版本、镜像 digest、Agent
集合和运行次数。

## 2. 可信实现与哈希

| 对象 | SHA-256 | 用途 |
|---|---|---|
| `solution/oracle.ptx` | `454278767b4d79a97df2bb51248e7c2bd4c4eae06a4d72f87ba757b5dc59fc6b` | 作者侧 Direct-PTX Oracle |
| `tests/reference.cu` | `c08f129e9e9f7618d37021d4989e4344928c56b706cc4a56a8e4d65dd5c20484` | 性能 Reference 的可信作者源码 |
| `tests/reference.ptx` | `d5e5660a54816dfbddf968e3c49505e91b7ad75b86fdee0a9d70b4a8e6e6c056` | 隐藏 verifier 加载的同 ABI Reference |
| PTX 文档 source record | `55d3c57c672b6bd168e34fcf62bb791115e32c4bbce909993c946620020ba5ec` | 固定 PTX ISA 9.2 来源记录 |

Reference 由 CUDA 13.2.1 / nvcc 13.2.78 在可信出题环境中生成：

```bash
nvcc -ptx -O3 --use_fast_math -arch=compute_100 \
  -o reference.ptx reference.cu
```

Agent 环境不包含 `nvcc`、NVRTC、NVVM 或 Triton；预编译 Reference 只在隐藏 verifier
阶段出现，不能被 Agent 读取。

## 3. 静态与汇编检查

以下检查均通过：

- 普通 task checker；
- scaffold checker 的 3 个回归测试；
- Python 语法检查和 Bash 语法检查；
- public/hidden policy checker、hardware checker 和 trusted manifest 逐字一致；
- Oracle、Reference 和全部 PTX fixture 均可由 CUDA 13.2 `ptxas -arch=sm_100` 汇编；
- `git diff --check`。

`--release` 当前只因 `metadata.release_ready = false` 而拒绝，这是开发阶段的预期行为。

## 4. B200 直接 verifier 验证

| Candidate | correctness | reward | 说明 |
|---|---:|---:|---|
| Oracle | 1 | `0.996761` | 正确且与固定 Reference 性能接近 |
| missing entry | 0 | 0 | 缺少规定 entry，被拒绝 |
| uniform output | 0 | 0 | 非均匀输入下数值错误，被拒绝 |
| partial output | 0 | 0 | 未写完整输出，被 NaN sentinel 检出 |
| slow-correct | 1 | `0.021475` | 正确但串行，连续性能分数显著下降 |

该组结果说明 correctness 硬门槛和连续性能分数是分开的：错误实现不能靠速度得分，慢但
正确的实现仍保留低性能分。

## 5. Harbor Oracle 验证

单次 Job：

```text
jobs/ptx-softmax-b200-oracle-feasibility-20260822-r2
trial: ptx-softmax-oracle-task__v9PXPGa
correctness: 1
reward: 1
raw speedup: 1.001688
```

同配置三次重复 Job：

```text
jobs/ptx-softmax-b200-oracle-repeat3-20260822
```

| trial | reward | raw speedup |
|---|---:|---:|
| `RdzfChZ` | `0.995444` | `0.995444` |
| `UZiFNFi` | `1.000000` | `1.008746` |
| `jr7hF66` | `0.997699` | `0.997699` |

三次 reward 均值为 `0.997714`，样本标准差为 `0.002278`，变异系数约 `0.228%`。
因此当前三种 `num_rows` 的几何平均计时在该批 B200 运行中足够稳定，可用于 feasibility
阶段的相对性能评分。

## 6. 真实 Agent 校准

固定配置：

```text
Agent:             Codex 0.148.0
model:             gpt-5.6-sol
reasoning effort:  high
Agent timeout:     600 seconds
backend:           patched Harbor Singularity
GPU:               one NVIDIA B200
task checksum:     275b227079fba688c3848b1e591e1e6fcb9dd6f85d45b9ac29c32b0c6a491f81
```

Job：

```text
jobs/ptx-softmax-b200-codex-gpt56-high-feasibility-20260822-r2
jobs/ptx-softmax-b200-codex-gpt56-high-pilot-remaining4-20260822
```

| trial | Agent 秒数 | correctness | reward | raw speedup | Candidate SHA-256 前 12 位 |
|---|---:|---:|---:|---:|---|
| `kNpqJQ3` | `373.131` | 1 | `1.000000` | `1.011935` | `58714c261e7b` |
| `3EGvREc` | `336.538` | 1 | `0.997635` | `0.997635` | `d9114db1f372` |
| `9tFnAy5` | `321.393` | 1 | `1.000000` | `1.006577` | `044ec106b5cb` |
| `sPd2LCT` | `281.042` | 1 | `0.994724` | `0.994724` | `f501339aaa41` |
| `sdNm6a6` | `367.160` | 1 | `1.000000` | `1.010635` | `0013deeb3f23` |

汇总：

```text
correctness pass: 5/5
Agent time mean:  335.853 seconds
Agent time range: 281.042–373.131 seconds
reward mean:      0.998472
raw speedup mean: 1.004301
unique PTX hash:  5/5
```

五次 Candidate 哈希均不同，且使用相同 task checksum。轨迹审计显示 Agent 直接创建并
修改 `/app/kernel.ptx`，使用允许的 `ptxas`、`cuobjdump` 和 compute-sanitizer 做本地
验证；没有创建 CUDA kernel、Triton 或其他高层 GPU 实现，没有调用 `nvcc`、NVRTC、
NVVM，也没有下载或安装工具。部分 trial 使用 Python 或 C 宿主 loader 直接加载候选
PTX，另有一条创建了辅助 PTX probe；这些工具不生成最终提交，符合当前 Direct-PTX
policy。

按 pilot 预设的五次口径，`5/5` 表明本题对 `gpt-5.6-sol/high` 是容易锚点，不能承担
最强模型 30%–50% 通过率的主要区分任务。结构上仍保留 Normalization/Reduction D2
标签，但 catalog 应明确记录本次模型校准结果；600 秒预算在这五次中有充足但并非无限
的余量，暂不因单一模型结果冻结为所有 Agent 的统一正式预算。

## 7. Backend 边界

正式 task 保留 Agent `no-network` 策略。当前 patched Singularity backend 不能在同一
环境内从 setup 阶段的 public 网络动态切换到 Agent 阶段的 no-network，因此作者侧
Oracle 和 Agent pilot 使用了内容相同、仅继承环境 public 策略的临时 task copy，并逐条
审计 Agent trajectory。这个 pilot 证明 task、Agent 和 verifier 可以闭环运行，但不等于
Singularity 已经强制执行正式的网络隔离。

Docker backend 或后续支持分阶段网络策略的 Singularity backend 仍需补一轮正式
no-network 验证。因缺少 `pdftotext` 而在 Agent setup 阶段失败的首次 Job 属于环境检查
配置问题，没有调用模型，也不计入 Agent 成功率。
