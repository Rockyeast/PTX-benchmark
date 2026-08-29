# `ptx-paged-decode-attention-bf16-hq32-hkv8-d128-p16` 验证记录

**状态：B200 feasibility 已通过；完成 5 次公开网络开发期校准和 1 次严格 Agent
allowlist 复验，`release_ready=false`。日期：2026-08-22。**

## 1. 已验证的基础设施与正确性

- Oracle 在 NVIDIA B200 / `sm_100` 上通过全部 9 个 `kv_len` correctness cases；
- Oracle Harbor Job 正常结束，`correctness=1`、`reward=0.999944731798595`；
- 固定 Reference 的三次手工重复均通过，Reference-vs-Reference speedup 为
  `1.000030`、`1.000067`、`1.000052`；
- `contiguous-pages`、`wrong-gqa`、`round-last-page` 三个错误 fixture 均被拒绝；
- `slow-correct` fixture 保持 `correctness=1`，speedup/reward 为
  `0.020915426348095238`。

Oracle Job：

```text
jobs/ptx-paged-decode-b200-oracle-feasibility-20260822/
```

## 2. 第一次真实 Agent feasibility

固定配置：

```text
Agent:              Codex CLI 0.148.0
model:              gpt-5.6-sol
reasoning_effort:   high
Agent budget:       600 seconds
hardware:           NVIDIA B200, GPU 0
benchmark setting:  Direct-PTX
```

运行时兼容性限制：当前远端 Singularity backend 不能在 `[environment] public` setup
之后动态切换到 `[agent] no-network`。本次使用临时 task 副本，仅移除了 Agent 阶段的
`no-network` override；正式 task 文件仍保持 `no-network`。Trajectory 中没有下载、
安装或高层编译行为，但本次不能作为严格网络隔离的 release 证据。

Job：

```text
jobs/ptx-paged-decode-b200-codex-gpt56-high-feasibility-20260822/
```

结果同时包含两类事实，不能只报告其中一类：

1. Harbor 在 600 秒边界取消 Agent，记录 `AgentTimeoutError`；
2. 截止取消时 `/app/kernel.ptx` 已存在，隐藏 verifier 随后判定：
   `correctness=1`、`reward=1`、`speedup_vs_reference=2.34855152485358`。

三个性能 case：

| `kv_len` | Candidate median | Reference median | speedup |
|---:|---:|---:|---:|
| 128 | 0.009844 ms | 0.021127 ms | 2.1463× |
| 1024 | 0.062151 ms | 0.151185 ms | 2.4325× |
| 4096 | 0.240397 ms | 0.596467 ms | 2.4812× |

`reward` 被上限截断到 1；`2.3486×` 只表示相对当前 task-owned CUDA
Reference，不表示相对 FlashInfer 或硬件最佳实现。

## 3. 超时原因时间线

以 Agent 收到题目为约 `t=0`：

| 时间 | 行为 |
|---:|---|
| 175 s | 首次创建 `kernel.ptx` |
| 178–250 s | 汇编并修复 PTX 地址/共享内存问题 |
| 335 s | 已有正确实现，开始专门优化性能 |
| 356–590 s | 多轮改写、测试和 `ncu` profiling |
| 590 s | 最终 artifact 检查成功 |
| 593 s | Agent 内部计划全部标记完成 |
| 600 s | Harbor timeout；最终自然语言回复尚未返回 |

因此本次不是 harness、PTX 汇编或 correctness 故障。模型在约 335 秒已经完成正确
实现，之后继续利用剩余预算优化，最终提交文件在截止时有效，但 Agent 进程没有正常
退出。

## 4. 五次固定配置校准

随后保持 task、task checksum、prompt、模型、reasoning effort 和 600 秒预算不变，
增加 4 次独立 attempt。五次共同的 task checksum 为：

```text
86da56acd4f17b639004654264b706347bbc6192fb4bda04ff8648bb6cc92652
```

| Trial | Agent 时间 | 正常退出 | correctness | reward | raw speedup |
|---|---:|---:|---:|---:|---:|
| `T7bgvtJ` | 600.24 s | 否，timeout | 1 | 1.0000 | 2.3486× |
| `JMPEk7d` | 598.32 s | 是 | 1 | 1.0000 | 2.6477× |
| `PZyXJ59` | 378.80 s | 是 | 1 | 0.6213 | 0.6213× |
| `iSssV4k` | 458.55 s | 是 | 1 | 1.0000 | 1.4804× |
| `t3f5YaU` | 421.71 s | 是 | 1 | 1.0000 | 2.1854× |

汇总：

```text
verifier correctness pass: 5/5
normal Agent completion:    4/5
Agent timeout:              1/5
raw speedup range:          0.6213x -- 2.6477x
```

本项目的 task contract 是在截止前交付 `/app/kernel.ptx`，最终 reward 由隐藏 verifier
决定。因此 `T7bgvtJ` 必须同时记为 verifier pass 和 timeout exception：不能把它简单
写成“完全失败”，也不能隐去 timeout 写成普通成功。

## 5. 当前判断

- 对 `gpt-5.6-sol/high`，本题 correctness 仍是容易题：5/5，不满足希望由强模型形成
  30%～50% correctness 通过率的区分目标；
- 600 秒足以让 5/5 留下正确 artifact，但正常退出仅 4/5，预算存在尾部风险；
- performance 有明显区分度，但当前 Reference 较弱，4/5 attempt 达到或超过其性能
  上限，尚不能据此冻结排行榜；
- 下一步不是继续重复同一模型，而是增强性能 Reference/门槛，或加入更难的 ABI/拓扑，
  并用开放权重模型校准；
- 这 5 次运行时未严格隔离公网，因此仍是开发期校准；后续第 6 节已经补充一次严格
  Agent allowlist 复验，但单次结果还不足以冻结难度分布或 release。

新增四次 Job：

```text
jobs/ptx-paged-decode-b200-codex-gpt56-high-repeat4-20260822/
```

## 6. Singularity 严格网络隔离复验

在 Harbor Singularity backend 增加逐 Agent 命令网络命名空间和 hostname allowlist
后，使用正式 task 的 `[agent] network_mode = "no-network"`，仅在本次运行追加：

```text
chatgpt.com
*.chatgpt.com
```

固定配置仍为 Codex CLI `0.148.0`、`gpt-5.6-sol`、high reasoning、600 秒和 B200。
本次 canonical task checksum 为：

```text
3928d3efdb79518f11280d45238acf273be5aba6d6fa9d3d4bcbef0067fe508f
```

它和第 4 节旧的临时 public task checksum 不同，主要原因是本次保留并实际执行了
正式 Agent 网络策略；不能把两个 checksum 伪装成完全相同的 task package。

结果：

| Trial | Agent 时间 | 正常退出 | correctness | reward | raw speedup |
|---|---:|---:|---:|---:|---:|
| `X2csiN8` | 504.30 s | 是 | 1 | 1.0000 | 2.2566× |

三个性能 case：

| `kv_len` | Candidate median | Reference median | speedup |
|---:|---:|---:|---:|
| 128 | 0.010139 ms | 0.021235 ms | 2.0945× |
| 1024 | 0.064871 ms | 0.150770 ms | 2.3241× |
| 4096 | 0.251993 ms | 0.594835 ms | 2.3605× |

合规审计：

- Agent 直接创建和修改 `/app/kernel.ptx`，最终 PTX SHA-256 为
  `b8e836126bc0c466df993c5d66359101863427a7059c0df47c403fc1cd24706c`；
- 使用了允许的 `ptxas`、`cuobjdump` 和 `compute-sanitizer`；
- Agent 为运行期自检创建了临时 C Driver loader，并用 `gcc` 编译；该程序只加载
  已手写的 PTX，没有生成 PTX，也不是提交 artifact；
- 没有使用 `nvcc`、NVRTC、NVVM、Triton，没有下载或安装工具；
- 网络代理只允许 `chatgpt.com` 和 `ab.chatgpt.com`；对
  `*.oaiusercontent.com` 的请求被明确拒绝，任务仍正常完成；
- 隐藏 verifier 独立通过 9 个 correctness case、硬件检查和性能测量；
- `exception_info=null`，Agent 正常退出，没有复现此前 600 秒边界 timeout。

Job：

```text
jobs/ptx-paged-decode-b200-codex-high-network-isolated-20260822/
```

这次结果补齐了“严格限制 Agent 公网”这一项运行证据，但仍只有一个隔离 attempt，
而且当前 Reference 偏弱。因此任务继续保持 `release_ready=false`。
