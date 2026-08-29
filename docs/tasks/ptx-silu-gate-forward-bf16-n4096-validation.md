# `ptx-silu-gate-forward-bf16-n4096` 验证记录

**状态：B200 correctness、性能 harness、两个错误负例、慢正确校准和 Harbor
Singularity Oracle Job 已通过；尚未完成重复计时和真实 Agent 校准。**

## 2026-08-21 B200 Oracle correctness-only

在 `catalyst-b200` 的 NVIDIA B200 GPU 0 上，使用现有 Direct-PTX SIF：

```text
/home/zhuominc/rockyeast-work/containers/
ptx-bench-direct-cuda13.2.1-codex0.148-ubuntu24.04-v1.sif
```

先用 CUDA 13.2 的 `ptxas` 汇编：

```bash
ptxas -arch=sm_100 -o kernel.cubin kernel.ptx
```

汇编成功。随后在同一 SIF 中直接执行 correctness verifier：

```bash
python3 verify_and_benchmark.py \
  --candidate kernel.ptx \
  --log metrics.json \
  --reward reward.json \
  --correctness-only
```

输出：

```text
PTX_SILU_CORRECTNESS_OK: 7 cases; performance implemented=False
```

正确性覆盖：

```text
num_tokens = 0, 1, 2, 3, 7, 31, 257
atol = 0.03
rtol = 0.01
GPU = NVIDIA B200
```

结果为：

```json
{
  "correctness": 1.0,
  "performance": 0.0,
  "reward": 0.0,
  "speedup_vs_reference": 0.0
}
```

`reward=0` 不是 Oracle 错误，而是 correctness-only 阶段故意保持 fail-closed：尚未实现
固定同 ABI 性能 Reference 和正式性能评分。

本次固定文件哈希：

```text
oracle.ptx:
84bba6446663398663f6778a0dd72b67539a3a2f8fbb54a417b7e659063cd861

verify_and_benchmark.py:
d400ed5915d361b428b912189b3976e6177418163cfd4e06f6ca2d878b93ed45
```

运行后 GPU 0 回到 4 MiB、0% utilization。

## 证据边界

这次验证证明：

- Oracle PTX 能被 `ptxas -arch=sm_100` 汇编；
- 固定 entry、参数 ABI 和 launch ABI 能在 B200 上运行；
- Oracle 通过独立 CPU 真值、BF16 容差、输入不可修改、输出完整性和 guard 检查；
- `num_tokens=0` 的提前返回路径通过。

这次 correctness-only 运行不是完整 Harbor Job，因此当时尚未证明：

- task 的环境构建和隐藏 policy checker 已完整对齐；
- 性能 Reference、计时稳定性与 reward 公式正确；
- Codex、Qwen 或其他真实 Agent 的通过率与难度。

## 2026-08-21 B200 GPU 负例

两个 author-only fixture 都先通过 CUDA 13.2：

```text
ptxas -arch=sm_100
```

因此测试的是可执行 PTX 的语义错误，不是用汇编错误冒充 correctness 检查。

| Fixture | 故意错误 | verifier 结果 | correctness / reward |
|---|---|---|---:|
| `missing-gate-multiply.ptx` | 计算 `sigmoid(gate) * up`，漏掉 gate 因子 | `candidate_rejected`，index 12 数值不符 | 0 / 0 |
| `partial-output.ptx` | 每行只写前 512/4096 个元素 | `candidate_rejected`，index 512 仍为 NaN sentinel | 0 / 0 |

具体首个错误分别为：

```text
output tokens=1: wrong output at index=12:
expected 0.2158203125, got -0.053955078125

output tokens=1: wrong output at index=512:
expected -0.07177734375, got nan
```

两次 verifier 退出状态均为 `1`，`metrics.json` 均记录
`status=candidate_rejected`，而不是 `infrastructure_error`。两份 `reward.json` 均为：

```json
{
  "correctness": 0.0,
  "performance": 0.0,
  "reward": 0.0,
  "speedup_vs_reference": 0.0
}
```

固定 fixture 哈希：

```text
missing-gate-multiply.ptx:
6b562d7205cd722ea44f2e0f7e53fff7cb2080e7d47c499322727d0e52ef1907

partial-output.ptx:
fe5756f102d579fab7201344406e31d9163b219e8394e6875acf2cc5e7f4902c
```

两次运行均使用 `catalyst-b200` GPU 0；结束后 GPU 0 回到 4 MiB、0% utilization。

更新性能 verifier 后又用固定 Reference 重新运行两个负例，错误位置、退出状态和
reward 均保持不变。

## 2026-08-22 固定性能 Reference

固定 Reference 从 FlashInfer 的向量化 `act_and_mul_kernel` 结构适配，来源为：

```text
repository: https://github.com/flashinfer-ai/flashinfer
commit:     103fcf862b902e100775ac781a0de36d067de3a4
file:       include/flashinfer/activation.cuh
license:    Apache-2.0
```

任务适配固定 BF16、`N=4096`、512-thread launch 和 raw PTX ABI，并删除 PDL 与框架
依赖。可信作者环境使用：

```text
CUDA 13.2 / nvcc V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```

生成的 PTX 随任务固定在 `tests/reference.ptx`。Agent 运行环境仍不包含 `nvcc`。

```text
reference.cu SHA-256:
c53f7f713687fddbfda5161eaf00b9712d0d47922fdb45a623a05a489bfbcfab

reference.ptx SHA-256:
89889b1e098239bbbb0e47adc47152ce4af1de063220951c85e6b179a38dd39c

performance verifier SHA-256:
d262eadbddc978e3b48078d39a85f0070793099125ad4697c8282adada4cf379
```

Reference PTX 在 Direct-PTX SIF 中通过 `ptxas -arch=sm_100`，并独立通过与 Candidate
相同的 7 个 CPU correctness cases。

## 2026-08-22 B200 性能 feasibility

性能输入使用固定的非均匀 BF16 gate/up 行，而不是全 0 或全 1。每个规模计时前检查
首、中、末 token 的代表性输出，并逐字节确认完整输入未被修改。Candidate 与 Reference
在同一 CUDA context、同一输入和同一输出 allocation 上交替执行，使用 CUDA Event、
预热、自适应 iterations、5 批次中位数。`T=1` 只诊断；`T=32,256,4096` 参与几何平均。

### 手写 Oracle 对固定 Reference

Oracle correctness 为 `1`：

| num_tokens | Oracle median ms | Reference median ms | Reference / Oracle |
|---:|---:|---:|---:|
| 1（诊断） | 0.006176 | 0.006160 | 0.997337 |
| 32 | 0.006090 | 0.006147 | 1.009431 |
| 256 | 0.006102 | 0.006140 | 1.006241 |
| 4096 | 0.038767 | 0.031906 | 0.823005 |

```text
speedup_vs_reference geometric mean = 0.9420205813
performance                           = 0.9420205813
reward                                = 0.9420205813
```

### 慢但正确实现

`slow-correct.ptx` 每个 token 只使用 thread 0，串行计算全部 4096 个元素。它通过全部
correctness cases，但性能显著下降：

| num_tokens | Slow median ms | Reference median ms | Reference / Slow |
|---:|---:|---:|---:|
| 1（诊断） | 0.273934 | 0.006545 | 0.023893 |
| 32 | 0.273927 | 0.006319 | 0.023068 |
| 256 | 0.275078 | 0.006320 | 0.022977 |
| 4096 | 0.944890 | 0.032280 | 0.034163 |

```text
correctness                           = 1.0
speedup_vs_reference geometric mean = 0.0262592363
performance                           = 0.0262592363
reward                                = 0.0262592363
```

慢实现 SHA-256：

```text
c9dc7d3861861f1c69c25cb77f794f86636dba9fc78d4408582fd54caaf1b47d
```

这证明当前连续 reward 能区分“正确且较快”和“正确但极慢”。但以上每种实现只完成一轮
正式记录，不能据此冻结性能分数或宣称 Reference 已达到 FlashInfer 官方端到端性能。

## 2026-08-22 Harbor Singularity Oracle feasibility

任务随后使用 patched Harbor 的 Singularity backend 和固定 Direct-PTX SIF，在
NVIDIA B200 上完成一次正式 Oracle Job。第一次尝试在 trusted setup 阶段失败：环境
policy 错误要求安装 `pdftotext`，但当前任务只随附 hash-pinned 的纯文本 PTX 文档来源
记录，并未分发 NVIDIA PDF。该结果属于环境基础设施错误，没有进入 Candidate 评分。
删除这项无关依赖后，setup policy smoke 和第二次 Job 均通过。

成功 Job：

```text
local job: jobs/ptx-silu-b200-oracle-performance-r2-20260822
job id:    3779821a-e137-48fd-bbcb-f8ccfea6dfaa
trial:     ptx-silu-harbor-task-20260822__r7A7LiE
result:    1 completed trial, 0 exceptions
```

阶段耗时：

```text
environment setup: 3.05 s
Oracle execution:   0.20 s
verifier:          11.30 s
whole Job:         15.13 s
```

隐藏 verifier 确认实际设备是 `NVIDIA B200`、compute capability `10.0`（`sm_100`），
并确认 Direct-PTX 环境 policy 通过。归档的 `/app/kernel.ptx` 与仓库 Oracle 哈希一致。

### Reward 公式

正确性是硬门槛。令 `c` 为 correctness（通过为 1，失败为 0），每个计分规模 `T` 的
相对速度为：

```text
r_T = Reference median time at T / Candidate median time at T
```

本题只对 `T = 32, 256, 4096` 计分，`T=1` 仅用于诊断。三个规模用几何平均聚合：

```text
raw_speedup = (r_32 * r_256 * r_4096)^(1/3)
performance = min(max(raw_speedup, 0), 1)
reward      = c * performance
```

因此 Candidate 与 Reference 一样快时约为 1；慢一半时相应比例约为 0.5；超过
Reference 的部分在当前 v0 中封顶为 1。任何汇编、加载、数值、输入不变性或内存安全
检查失败都会令 `c=0`，最终 reward 直接为 0，不会用速度抵消错误。

本次 Oracle 的实测值为：

| num_tokens | Oracle median ms | Reference median ms | `r_T` | 是否计分 |
|---:|---:|---:|---:|:---:|
| 1 | 0.006224 | 0.006209 | 0.997617 | 否 |
| 32 | 0.006219 | 0.006238 | 1.003166 | 是 |
| 256 | 0.006231 | 0.006257 | 1.004211 | 是 |
| 4096 | 0.039820 | 0.031864 | 0.800196 | 是 |

```text
correctness = 1
raw_speedup = (1.003166 * 1.004211 * 0.800196)^(1/3)
            = 0.9306749916
performance = 0.9306749916
reward      = 1 * 0.9306749916
            = 0.9306749916
```

这里 CPU 公式负责判断“算得对不对”，固定 Reference 只负责提供“跑得多快”的比较
基线。Oracle 和 Reference 的代码不必相同，也不是互相比较输出。

### 本次证据边界

- 成功 Job 证明 Harbor 能完成 task setup、Oracle 提交、隐藏 policy/hardware 检查、
  correctness、性能计时和 reward 收集的完整链路；
- 为适配当前 patched Singularity backend，author-only 的临时 Oracle task copy 删除了
  Agent 阶段 `no-network` override；仓库内正式 task 仍保留 `no-network`；
- 因而这次 Oracle Job 不用于证明真实 Agent 的网络隔离，必须由后续 Agent Job 单独审计；
- `T=4096` 的 Candidate dispersion 为 `12.40%`，即使 harness 已自动重测一次，仍偏高；
  当前 `0.930675` 只能作为 feasibility 结果，不能冻结为正式分数。

## 2026-08-22 Harbor Oracle 三次重复测量

为判断同一份 PTX 重复评分是否稳定，使用相同 task 临时副本、相同 Direct-PTX SIF、
相同 Oracle、相同 Reference 和相同 verifier，串行运行三个全新 trial：

```text
local job:    jobs/ptx-silu-b200-oracle-repeat3-20260822
job id:       e442c849-d61d-49c4-9b52-377848af89cc
task digest:  748bdd44cad57878b3a4d780048e87f6cf5bdcfcacc673d8e84533e2f0f320ef
result:       3 completed trials, 0 exceptions
correctness:  3/3
```

三个 trial 归档的 Candidate SHA-256 都是：

```text
84bba6446663398663f6778a0dd72b67539a3a2f8fbb54a417b7e659063cd861
```

因此测量对象确实是同一份 `oracle.ptx`。

| Trial | `r_32` | `r_256` | `r_4096` | reward |
|---|---:|---:|---:|---:|
| `QTbygNV` | 1.009191 | 1.202605 | 0.766399 | 0.976151 |
| `tGYHDMa` | 1.008804 | 1.203227 | 0.788537 | 0.985505 |
| `DnEYv8j` | 1.008389 | 1.206280 | 0.805473 | 0.993213 |

三次 reward 的统计量：

```text
mean             = 0.984956
sample std       = 0.008544
coefficient var  = 0.867%
min / max        = 0.976151 / 0.993213
range            = 0.017062
```

这说明同一批串行重复的 task-level reward 比较集中，三次只相差约 `1.7` 个百分点。
但是它们都高于更早的单次结果 `0.930675`；同时本批次的 Candidate 和 Reference 绝对
用时都与早前记录不同。这说明同卡交替计时和比值聚合能够抵消大部分共同波动，但当前
证据还不足以断言跨时间、跨物理 B200 的性能比值已经完全稳定。

另一个可观测性缺口是：隐藏 hardware 记录确认容器内设备为 B200/sm_100，但当前 Job
证据没有保存 host 物理 GPU 编号、UUID、时钟或功耗状态。正式冻结前应补齐这些字段，
再做一次跨批次复测；这不会阻止下一步进行 Agent feasibility，但会阻止冻结性能分数。

## 2026-08-22 Codex Agent feasibility

使用真实 Codex Agent 运行一个全新 trial：

```text
local job:          jobs/ptx-silu-b200-codex-gpt56-high-feasibility-20260822
job id:             5f399eb7-8822-403a-a244-bb775c7c8d8f
trial:              ptx-silu-codex-high-20260822__fFdy4BM
task digest:        28c7cede74b4cc025c7ed4c494533c092dea0240fd3641d2b51c9b6bfa1735b5
Agent:              Codex CLI 0.148.0
model:              gpt-5.6-sol
reasoning_effort:   high
Agent timeout:      600 s
```

运行正常结束，没有 timeout 或 exception：

```text
environment setup:   6.09 s
Agent setup:          1.02 s
Agent execution:    532.27 s
verifier:            11.80 s
whole Job:          553.39 s
```

Harbor 记录的用量为 `1,144,678` input tokens、`1,045,504` cached input tokens、
`19,989` output tokens；其结果文件给出的估算成本为 `$1.618292`，该字段是 Harbor
估算值，不在这里推断实际订阅或账单扣费方式。

### Correctness 与性能

隐藏 verifier 的 7 个 correctness cases 全部通过，环境 policy 和 B200/sm_100 检查也
通过：

| num_tokens | Candidate median ms | Reference median ms | `r_T` | 是否计分 |
|---:|---:|---:|---:|:---:|
| 1 | 0.006224 | 0.006233 | 1.001567 | 否 |
| 32 | 0.006181 | 0.006205 | 1.003781 | 是 |
| 256 | 0.006164 | 0.006218 | 1.008649 | 是 |
| 4096 | 0.013543 | 0.031787 | 2.347120 | 是 |

```text
correctness          = 1
raw speedup          = 1.3344571943
capped performance   = 1
reward               = 1
```

Candidate 使用 `tanh.approx.f32` 和恒等式
`silu(g) = (g / 2) * (1 + tanh(g / 2))`，每 thread 向量化处理 8 个 BF16 元素。
归档 PTX 为 3,944 bytes，SHA-256：

```text
e1191499a2892515a0edfffdbbddee07389c5e202603050f7a37ef0a4fd00062
```

Agent 自己的 `ptxas` 结果为 26 registers、0 spill、0 shared memory；隐藏 verifier
确认该实现满足正式 CPU truth 和保护检查。它在 `T=4096` 上明显快于当前固定
Reference，说明 task 可以接受与 Reference 不同且更好的 PTX 策略；同时也说明当前
封顶 reward 会丢失 Reference 以上的性能差异，正式校准时必须同时保留 raw speedup。

### Direct-PTX trajectory 审计

- Agent 直接创建并迭代 `/app/kernel.ptx`；最终唯一评分产物仍是该 PTX；
- 使用 `ptxas`、`cuobjdump` 和 `compute-sanitizer` 做 PTX 汇编、资源和内存检查；
- 没有调用或安装 `nvcc`、NVRTC、NVVM、Triton，也没有下载工具或文档；
- Agent 创建 `/tmp/test_kernel.cpp`，用 `g++` 编译成只负责 CUDA Driver API 加载和
  自测已有 `/app/kernel.ptx` 的 host utility；它不包含 GPU kernel，也没有生成 PTX；
- 因此该行为符合当前 Direct-PTX v0 的“禁止高层语言生成提交 PTX”规则。如果未来要
  禁止任何辅助 C++ host harness，必须作为更严格的新 policy 明确规定，不能事后把本次
  合规结果判成作弊；
- 当前 patched Singularity backend 不能严格切换 Agent 阶段网络。本次临时 task 副本
  使用 public 网络以访问托管 Codex API；trajectory 没有网络下载行为，但这次运行不能
  证明 `no-network` enforcement。

一次成功只得到 `1/1` feasibility 结果，不能据此估计 Codex 的正式通过率，也不能把
本题升级为 calibrated 或 frozen。

## 2026-08-22 Codex 五次 pilot

按照预先固定的 pilot 规则，在第一次成功之后补充四个独立 attempt。四次均使用与
第一次相同的 task content digest、Agent、模型、reasoning effort、600 秒 timeout 和
Direct-PTX SIF；后四次串行固定在物理 GPU 4：

```text
task content digest:
sha256:9cc2ae34b755f3f73eec7a7753dafa44f90affa8c3308afd736bac9ef60f888f

legacy trial task_checksum:
28c7cede74b4cc025c7ed4c494533c092dea0240fd3641d2b51c9b6bfa1735b5

remaining-four job:
jobs/ptx-silu-b200-codex-gpt56-high-pilot-remaining4-r2-20260822

job id:
edd4d036-bd1b-480e-a73e-208890fe3c8f
```

每个 trial 都从新的 `/app` 和新的 Codex session 开始。五份归档 PTX 的 SHA-256 全部
不同，说明不是恢复或复制前一个成功答案。

| Pilot | Trial suffix | Agent 秒数 | correctness | raw speedup | reward |
|---:|---|---:|---:|---:|---:|
| 1 | `fFdy4BM` | 532.27 | 1 | 1.334457 | 1 |
| 2 | `32DpNUW` | 413.13 | 1 | 1.341005 | 1 |
| 3 | `gQSQpxb` | 298.02 | 1 | 1.318838 | 1 |
| 4 | `XqiqC7w` | 256.67 | 1 | 1.317576 | 1 |
| 5 | `icJQcnm` | 542.46 | 1 | 1.301971 | 1 |

汇总：

```text
correctness pass rate = 5 / 5 = 100%
Agent time mean       = 408.51 s
Agent time range      = 256.67–542.46 s
raw speedup mean      = 1.322769
raw speedup range     = 1.301971–1.341005
```

Harbor 对五次合计记录 `4,519,240` input tokens、`4,211,456` cached input tokens、
`89,187` output tokens，估算成本合计 `$6.320258`。该金额仍只是 Harbor 结果字段，
不推断实际订阅账单。

五份 trajectory 均直接创建 `/app/kernel.ptx`，没有调用或安装 CUDA/Triton 高层
kernel 编译器，也没有下载工具或文档。不同 trial 创建过 `.c`、`.cpp` 或 `.py` 的
辅助 host loader/测试程序，但搜索不到 `__global__`、CUDA launch syntax、NVRTC 或
Triton kernel；这些工具只加载和测试手写 PTX，符合当前 Direct-PTX v0。五次隐藏环境
policy 均通过，下载的 Job 中也未发现明显认证 token。

后四次的每个 trial 都有完整 `result.json`、trajectory、artifact、hidden verifier
结果，且 Job 统计为 4 completed / 0 errors。但该 Job 顶层 `result.json` 的
`finished_at` 意外保持 `null`，`job.log` 也在最后一个 Singularity server 关闭后结束，
没有正常打印 Job summary。这属于 Job 聚合收尾异常，不改变四个已完成 trial 的独立
评分，但正式基础设施验收前需要另行修复或复现。

按预先约定的五次 pilot 判定，`5/5` 表明本题对
`Codex + gpt-5.6-sol + high + 600 s` 是容易题。它可以作为 D1 easy anchor 保留，但
不能承担区分最强模型的主要作用。整组 benchmark 的 `30%–50%` 暂定目标应由不同
难度 task 的总体平均实现，不要求每一道 task 都单独落在该区间。

五次 capped reward 都是 1，而 raw speedup 仍能区分 `1.302–1.341`。这证明当前封顶
reward 会丢失 Reference 以上的性能差异；在冻结公式前应决定 leaderboard 是否额外
排名 raw speedup、增强 Reference，或调整连续性能分设计。现在不根据结果回头修改这
五次 pilot 的公式。

## 2026-08-22 Direct-PTX v0.2 pre-Agent healthcheck

五次 pilot 完成后，共享模板升级为 `direct-ptx-v0.2`，要求在 Agent 开始前同时运行
环境 policy 和目标 GPU 检查。SiLU task 对齐了该接口：trusted setup 安装与隐藏
verifier 完全相同的 `verify_hardware.py`，`task.toml` 声明 fail-fast healthcheck，
Docker 与 Singularity 共用同一份检查器。

使用当前 task 临时副本完成一次不调用模型的 Harbor Singularity Oracle smoke：

```text
local job: jobs/ptx-silu-b200-v02-preflight-oracle-smoke-20260822
job id:    6e84083d-b932-4131-9137-59cbddc5aa1e
result:    1 completed trial, 0 exceptions
```

Job 日志明确记录：

```text
Running healthcheck: /usr/local/bin/direct-ptx-policy-check &&
  /usr/local/bin/direct-ptx-verify-hardware \
    --output /logs/environment/hardware-preflight.json
Healthcheck passed
```

随后 Oracle correctness 为 `1`。该 smoke 只验证 v0.2 环境 preflight，不计入五次
Codex Agent pilot 分母。

## 下一步

1. 将本题保留为 D1 easy anchor，开始规格化下一道更有区分度的 task；
2. 开放权重 Agent 可用后再补本题的跨模型校准；
3. 给性能证据补充 host GPU 编号/UUID/时钟状态，并调查 Job 聚合收尾异常；
4. 根据多模型数据决定是否增强 Reference、调整 timeout 或保留未封顶性能指标。
