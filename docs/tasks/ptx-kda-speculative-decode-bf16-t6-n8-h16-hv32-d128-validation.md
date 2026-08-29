# `ptx-kda-speculative-decode-bf16-t6-n8-h16-hv32-d128` 验证记录

**日期：2026-08-22。当前为 feasibility 证据，不是 release 结论。**

## 固定实现

```text
upstream commit: 4896cd2da4ca421829907ab9fd3550e3f84a141b
route:           d128_t6_cached_gram_split1
entry:           flash_kda_decode_sm100a_t6_split1
shape:           T6, N8, H16, HV32, D128
launch:          grid=(32,8,1), block=256, dynamic smem=50560
hardware:        NVIDIA B200 / sm_100a
```

Artifact hashes：

```text
reference.cu:  49128e5a09c78267ebabf0514f9f6105d02a6cf73b9e9482133e4c67ed720dc1
reference.ptx: a98b57adf13e2d0df0841801c1ff679188a149b6463f664037aee5a4b1d0fe72
oracle.ptx:    a98b57adf13e2d0df0841801c1ff679188a149b6463f664037aee5a4b1d0fe72
CPU truth:     ef1b3a117152ccf885019bbaebee40aebac00cfb0aa38576c2dbf3f40d694348
```

## B200 直接 verifier

最初直接把 `.ptx` 交给 Driver API JIT 时，官方 T6 PTX 在同步阶段返回
`CUDA_ERROR_INVALID_ADDRESS_SPACE (717)`；相同源码通过 CUDA Runtime/AOT cubin 路径
可正常运行。最终 verifier 改为固定 CUDA 13.2.1 `ptxas` 先汇编 Candidate 和
Reference，再加载 cubin。修正后：

```text
Oracle correctness cases: 8/8
correctness:               1
reward:                    1
raw speedup:               1.000579843759767x
ptxas:                     123 registers, no stack/spill, 2 barriers
```

同时修正了性能阶段的 ABI 参数排列，使其保持
`q,k,v,g,beta,state,output,indices,accepted,scale`。

三份能通过 `ptxas` 的语义负例全部被 CPU truth 拒绝：

| fixture | 破坏内容 | correctness | reward |
|---|---|---:|---:|
| `wrong-head-map.ptx` | 错误 `hv -> qh` 映射 | 0 | 0 |
| `final-checkpoint-only.ptx` | 只保存最后 checkpoint | 0 | 0 |
| `wrong-accepted.ptx` | 忽略 accepted-token 初始槽 | 0 | 0 |

## Harbor Oracle

错误使用含 `nvcc` 的旧 CUDA SIF 时，Direct-PTX setup 按策略在 Agent/Oracle 运行前
拒绝环境；该 Job 是基础设施失败，不是 task 得分。

使用不含 `nvcc/NVRTC/NVVM/Triton` 的 Direct-PTX Codex SIF 后：

```text
remote Job:  jobs/ptx-kda-t6-n8-b200-oracle-final-r2-20260822/
trial:       ptx-kda-speculative-decode-bf16__o2w5yrz
checksum:    0ba8388b3a12b0adf8f0b281b2dd968ba023eea35841d1c652e396190921d967
trial count: 1
exceptions:  0
correctness: 1
performance: 1
reward:      1
raw speedup: 1.0476426405301735x
runtime:     about 25 seconds
```

Oracle 与 Reference 内容相同，Harbor 单次 speedup 的 4.8% 偏离属于需继续调查的计时
方差；直接 verifier 的同源比值更接近 1。该单次结果只证明完整 Harbor 路径可运行，
不能用来冻结性能分。

## 真实 Agent

第一次启动未追加模型 API allowlist，`chatgpt.com` 被 `no-network` 阻断，Agent 没有
得到模型响应；该 Trial 被终止并作为配置错误保留，不计入模型通过率。

第二次固定配置：

```text
Job:             jobs/ptx-kda-t6-n8-b200-codex-gpt56-high-feasibility-r2-20260822/
trial:           ptx-kda-speculative-decode-bf16__9x4DMwK
task checksum:   0ba8388b3a12b0adf8f0b281b2dd968ba023eea35841d1c652e396190921d967
Agent:           Codex CLI 0.148.0
model:           gpt-5.6-sol
reasoning:       high
budget:          600 seconds
API allowlist:   chatgpt.com, *.chatgpt.com
```

结果：

```text
normal exit:         yes
Agent execution:     419.635 seconds
correctness:         1 (8/8 hidden cases)
raw speedup:         0.8963789704554211x
performance/reward:  0.8963789704554211
artifact size:       12,985 bytes
artifact SHA-256:    131686a2580c5f805863d2a710d2141e18246bd4e505f0e316e1f12bf1d014ca
ptxas:               32 registers, no stack/spill, 1 barrier
input/cache/output:  1,063,849 / 1,008,128 / 19,351 tokens
Harbor cost estimate: USD 1.0131552 (not an external billing receipt)
```

四个 performance cases：

| case | Candidate | Reference | speedup |
|---|---:|---:|---:|
| random-mixed-accepted | 303.323 us | 272.529 us | 0.898478x |
| permuted-checkpoints | 320.891 us | 287.502 us | 0.895951x |
| structured-head-map | 302.046 us | 270.233 us | 0.894675x |
| accepted-boundaries | 304.785 us | 273.214 us | 0.896417x |

Agent 直接创建并修改 `/app/kernel.ptx`，用 `ptxas`/`cuobjdump` 检查，并临时写 C
Driver loader 与 CPU recurrence 验证已手写 PTX；没有用 C/CUDA 生成 PTX，没有安装或
下载工具，也没有创建 CUDA/Triton 高层 kernel。API allowlist 之外的
`*.oaiusercontent.com` 被代理拒绝。

环境审计起初在 `ldconfig -p` 中看到 `libnvrtc.so` 名称，但复查确认实际 NVRTC
runtime 已不存在，`ctypes.CDLL("libnvrtc.so")` 也无法加载；这些是过期 cache 记录，
旧 SIF 只残留 NVRTC stub。Agent 只枚举了它们，没有调用 NVRTC，artifact 和行为仍是
直接 PTX。

审计仍暴露出一个真实 checker bug：`/usr/local/cuda` 是 symlink，而旧 checker 的
`find` 没有 `-L`，因此没有遍历并发现 stub。模板和本题现已对 NVRTC/NVVM 检查使用
`find -L`。该修复会改变 task checksum，而且旧 SIF 在清理 stub 前会被新 checker
拒绝，所以本次仍只作为 Agent feasibility，不能作为最终 release 证据。

## 当前结论

- 这是当前 pilot 第一题直接选择公开最复杂 T6 路径的上限探针；
- Oracle、三个定向负例和真实 Agent 都证明 ABI/harness 可运行；
- 强模型单次可在 600 秒内正确完成，但耗时约 420 秒且比 Reference 慢约 11.6%；
- `1/1` 不能冻结难度或推出 30%--50% 通过率；
- 当前最先要做的是按更严格 policy 重建 SIF，而不是再增加 T7。

## 严格 v2 环境复验

上述 checker 修复后，重新构建 Codex Direct-PTX SIF：

```text
path:   /home/zhuominc/rockyeast-work/containers/
        ptx-bench-direct-cuda13.2.1-codex0.148-ubuntu24.04-v2.sif
SHA256: 475016c4e67018326cae47ef7b8badeb9f9ecd895a206dc77718f99bc9c30edc
```

镜像验证确认 `ptxas/cuobjdump/codex` 存在，`nvcc`、所有 NVRTC/NVVM 文件和 Triton
不存在。同步修正后 task，在新 canonical checksum 上完成 Harbor Oracle：

```text
Job:        jobs/ptx-kda-t6-n8-b200-oracle-strict-v2-20260822/
trial:      ptx-kda-speculative-decode-bf16__Zf2rjzk
checksum:   98f17f4b294bfba0f04975cc21c9516104ad37d5e973a87a4a6d8dbcd401824b
exceptions: 0
correctness: 1
reward:      1
speedup:     1.004232039636413x
runtime:     about 38 seconds
```

同源 Oracle/Reference 比值回到接近 1。严格 v2 真实 Agent：

```text
Job:            jobs/ptx-kda-t6-n8-b200-codex-gpt56-high-strict-v2-20260822/
trial:          ptx-kda-speculative-decode-bf16__AKykwwm
checksum:       98f17f4b294bfba0f04975cc21c9516104ad37d5e973a87a4a6d8dbcd401824b
Agent:          Codex CLI 0.148.0 / gpt-5.6-sol / high
Agent time:     602.560 seconds including cancellation tail
ending:         AgentTimeoutError at the 600-second budget
correctness:    1 (8/8 hidden cases)
performance:    0.1826534372739719
reward:         0.1826534372739719
artifact:       7e459e765bb56442522629df50f21221e8a00f29119108b7c658c2731be470f7
ptxas:          48 registers, no stack/spill, 1 barrier
tokens:         1,250,760 input / 1,186,816 cached / 27,295 output
cost estimate:  USD 1.2764024 (not an external billing receipt)
```

四个隐藏 performance case：

| case | Candidate | Reference | speedup |
|---|---:|---:|---:|
| random-mixed-accepted | 159.652 us | 24.589 us | 0.154015x |
| permuted-checkpoints | 159.745 us | 24.621 us | 0.154129x |
| structured-head-map | 160.733 us | 34.725 us | 0.216043x |
| accepted-boundaries | 160.420 us | 34.816 us | 0.217032x |

这次 trajectory 中，Agent 第一版 PTX 能汇编但有 84,833 个 state 和 62,545 个 output
元素超容差；它定位并修复 8-lane shuffle 分段后达到零错误。随后继续做性能优化；在
600 秒前的最后一次向量化保持 correctness，但性能比此前约 81 us 的中间版本明显退化。
这说明只归档最终 PTX 会丢失 Agent 迭代中出现过的更好 candidate，未来 evolution loop
应显式保存 best-so-far artifact。

合规审计：只提交 `kernel.ptx`；使用 `ptxas`、`cuobjdump`、CUDA Driver loader、GCC
编译临时 host harness 和 B200 自测；没有 `nvcc`、NVRTC、NVVM、Triton，没有安装或
下载工具。模型 API 是唯一放行公网。该 Trial 满足修正后的严格 Direct-PTX 环境约束。

两次语义相同的强模型 artifact correctness 为 `2/2`：旧 checksum 一次正常结束，
当前 strict checksum 一次 timeout 但 artifact 正确。正式统计只应把 current checksum
记为 `1/1`，不能混成同一冻结样本。

## 3600 秒候选预算

上述两次真实 Agent 都属于历史 600 秒校准。由于 strict Trial 在保持 correctness 的
情况下于优化阶段超时，当前 task 将 `[agent].timeout_sec` 调整为 3600 秒。该修改会
产生新的 canonical task checksum；必须在新 checksum 上重新运行 Oracle 和至少一次
真实 Agent，之后才能判断 3600 秒是否适合作为正式预算。旧结果不得与新结果合并统计。

### 新 checksum 的 Oracle

```text
Job:        jobs/ptx-kda-t6-n8-b200-oracle-budget3600-r2-20260822/
trial:      ptx-kda-speculative-decode-bf16__bvSoHRv
checksum:   18686a936565b9c3dbc02c545f433d73a75cd14a70dc395f094b1a5763d21cb9
exceptions: 0
correctness: 1
reward:      1
speedup:     1.0030215579765698x
```

### Codex/high 3600 秒上限 Trial

```text
Job:            jobs/ptx-kda-t6-n8-b200-codex-gpt56-high-budget3600-20260822/
trial:          ptx-kda-speculative-decode-bf16__cBVzGc8
checksum:       18686a936565b9c3dbc02c545f433d73a75cd14a70dc395f094b1a5763d21cb9
Agent:          Codex CLI 0.148.0 / gpt-5.6-sol / high
Agent time:     359.783 seconds
ending:         normal, no exception
correctness:    1 (8/8 hidden cases)
performance:    0.2774790061213799
reward:         0.2774790061213799
artifact SHA:   a632b93a30e75df13ff01f48991a9d1c870917e7deb7bbea1c397036abe8bd0b
ptxas:          29 registers, no stack/spill, 1 barrier
tokens:         791,791 input / 748,544 cached / 17,923 output
```

四个隐藏 performance case 的 Candidate/Reference 中位数为：

| case | Candidate | Reference | speedup |
|---|---:|---:|---:|
| random-mixed-accepted | 107.697 us | 28.156 us | 0.261433x |
| permuted-checkpoints | 47.198 us | 13.375 us | 0.283375x |
| structured-head-map | 47.084 us | 13.196 us | 0.280277x |
| accepted-boundaries | 47.101 us | 13.448 us | 0.285504x |

Agent 在约六分钟内手写出 342 行 PTX，自建 CPU truth/Driver harness，在 B200 上达到
零 tolerance violation，并用 compute-sanitizer 得到 0 error；临时 host harness 随后
删除，最终只提交 `kernel.ptx`。它没有使用 `nvcc`、NVRTC、NVVM 或 Triton。

关键结论是：3600 秒是允许的最大墙钟时间，不会强迫 Agent 用满。该 Trial 实际在
359.783 秒自然结束，没有做多轮 Reference 导向性能搜索；因此它不能证明“一小时预算
改善了结果”，只证明新预算不会截断这一次运行。当前 checksum 仍只有 `1/1`，正式预算
和难度都不能冻结。

## Direct-PTX v0.3 公开优化闭环

v0.3 在不暴露固定 Reference、隐藏 case、CPU 真值、期望输出或 reward 的前提下，给
Agent 提供 Candidate-only 公开测速命令：

```text
direct-ptx-benchmark-candidate /app/kernel.ptx
```

该工具使用一个固定、公开且不参与隐藏评分的 development input，只报告 Candidate
绝对延迟、5 组原始样本、离散度、逻辑吞吐和 `ptxas` 资源。它只用于优化反馈；隐藏
correctness 仍是最终 performance reward 的硬门槛。

在 B200 GPU 0 上用 Oracle PTX 直测公开工具：

```text
median:      13.7035 us
dispersion:  0.001495
logical:     11.0187 TFLOPS
ptxas:       123 registers, no stack/spill, 2 barriers
```

### v0.3 Oracle

```text
Job:        jobs/ptx-kda-t6-n8-b200-oracle-public-bench-v03-20260822/
trial:      ptx-kda-speculative-decode-bf16__kNRkHZG
checksum:   32e52d172ba54373c1c4dbd482052754f936fe4f4789bd34080a83fe2aa32cfd
exceptions: 0
correctness: 1
reward:      1
speedup:     1.0035026695622253x
runtime:     about 36 seconds
```

### Codex/high 公开优化行为

```text
Job:          jobs/ptx-kda-t6-n8-b200-codex-gpt56-high-public-bench-v03-r2-20260822/
trial:        ptx-kda-speculative-decode-bf16__YLkCbFW
checksum:     32e52d172ba54373c1c4dbd482052754f936fe4f4789bd34080a83fe2aa32cfd
Agent:        Codex CLI 0.148.0 / gpt-5.6-sol / high
Agent time:   603.649 seconds
ending:       AgentTimeoutError caused by hidden 600-second HTTP transport cap
correctness:  1 (8/8 hidden cases)
performance:  0.31968095788138706
reward:       0.31968095788138706
artifact SHA: 4c84acefc7c51cff3e670bb9975c4e44a71c8cad7d3e73316ee6f269c0aabd76
ptxas:        32 registers, no stack/spill, 1 barrier
tokens:       1,211,153 input / 1,094,144 cached / 24,064 output
```

公开反馈形成了真实的多轮优化轨迹：

| Candidate | 自建完整 state/output 检查 | public median | 相对第一版 |
|---|---:|---:|---:|
| scalar | 通过 | 60.0869 us | 1.000x |
| adjacent BF16 pair | 通过 | 45.8796 us | 1.310x |
| four-value widened access | 通过 | 42.4882 us | 1.414x |

最终 hidden performance case：

| case | Candidate | Reference | speedup |
|---|---:|---:|---:|
| random-mixed-accepted | 42.647 us | 13.685 us | 0.320884x |
| permuted-checkpoints | 42.642 us | 13.676 us | 0.320725x |
| structured-head-map | 42.631 us | 13.473 us | 0.316038x |
| accepted-boundaries | 42.630 us | 13.689 us | 0.321104x |

合规审计：最终 artifact 只有 `kernel.ptx`；Agent 使用了允许的 `ptxas`、
`cuobjdump`、GCC 和 CUDA Driver 编写临时 host correctness harness，没有用高层语言
生成 PTX。它尝试从 PyPI 安装 NumPy 时被模型 API allowlist 以 HTTP 403 拒绝。隐藏
Direct-PTX policy 复核通过。

本 Trial 暴露的基础设施问题不是 task 的 3600 秒预算：installed Codex 没有把单条
命令 timeout 传给 Singularity，后者又给 `environment.exec` 使用固定 600 秒 HTTP
read timeout，因此内层先在约 600 秒断开，外层却报告 3600 秒 AgentTimeoutError。
`patches/harbor/0004-singularity-long-command-timeout.patch` 已把无显式命令 timeout 的
transport deadline 改为由外层 Agent phase 控制；显式命令 timeout 仍保留 10 秒收尾
余量。个人 Harbor 副本的相关 GPU allocator、network guard 和 timeout 单测共
`19 passed`，ruff check/format 通过。由于本次 Agent 没有正常结束，修复后仍需再跑
至少一次真实 Agent，才能把 v0.3 记为正常完成样本。

### transport timeout 修复后的正常完成 Trial

```text
Job:          jobs/ptx-kda-t6-n8-b200-codex-gpt56-high-public-bench-v03-postfix-r3-20260822/
trial:        ptx-kda-speculative-decode-bf16__bW73nds
checksum:     32e52d172ba54373c1c4dbd482052754f936fe4f4789bd34080a83fe2aa32cfd
Agent:        Codex CLI 0.148.0 / gpt-5.6-sol / high
Agent time:   1299.736 seconds
ending:       normal, no exception
correctness:  1 (8/8 hidden cases)
performance:  0.5124852675720918
reward:       0.5124852675720918
artifact SHA: 0032148d32de1d60cb8fa742994ed55f82e7bad25ab59ff75e5e630bfc06a4e2
ptxas:        56 registers, no stack/spill, 1 barrier
tokens:       7,471,032 input / 7,298,048 cached / 59,403 output
cost estimate: USD 4.7992152 (not an external billing receipt)
```

四个隐藏 performance case：

| case | Candidate | Reference | speedup |
|---|---:|---:|---:|
| random-mixed-accepted | 26.574 us | 13.665 us | 0.514213x |
| permuted-checkpoints | 26.546 us | 13.664 us | 0.514730x |
| structured-head-map | 26.577 us | 13.454 us | 0.506218x |
| accepted-boundaries | 26.545 us | 13.666 us | 0.514831x |

Agent 在公开工具上经历了约 `48.74 -> 41.77 -> 40.66 -> 40.23 -> 40.07 ->
27.15 -> 26.99 -> 26.29 us` 的多轮优化。期间一个激进共享内存布局版本触发
`misaligned address`，被自建 CPU/Driver correctness harness 拒绝；最终版本在普通和
压力输入上均为 `state_bad=0`、`out_bad=0`。公开 median 为约 26.28 us，隐藏 case
稳定在 26.55--26.58 us，说明公开反馈没有只针对一个偶然输入过拟合。

该 Trial 的 Agent 阶段超过 600 秒后仍持续运行，并在 1299.736 秒正常结束；随后隐藏
verifier 正常完成。因此 `0004-singularity-long-command-timeout.patch` 已获得真实端到端
运行证据：Singularity transport 不再错误地覆盖 task 的 3600 秒 Agent 预算。该结论
只适用于个人隔离 Harbor 副本中已应用补丁的 backend，尚未提交到 Harbor 上游。

合规审计：最终 artifact 只有纯手写 `kernel.ptx`；Agent 使用允许的 `ptxas`、
`cuobjdump`、GCC 和 CUDA Driver 编写临时 host harness，没有用 `nvcc`、NVRTC、
NVVM、Triton，也没有下载或安装工具。最终 reward 的计算仍是 correctness 硬门槛后
的 `clamp(reference_time / candidate_time, 0, 1)`；本次 0.512485 表示 Candidate
约为固定 Reference 速度的 51.25%，不是 51.25% 的正确率。

### 同 checksum 的两次并行重复

```text
Job:      jobs/ptx-kda-t6-n8-b200-codex-gpt56-high-public-bench-v03-postfix-repeat2-20260823/
checksum: 32e52d172ba54373c1c4dbd482052754f936fe4f4789bd34080a83fe2aa32cfd
attempts: 2
errors:   0
mean reward: 0.33844895460317015
```

| Trial | Agent time | correctness | reward | public final | registers | artifact SHA |
|---|---:|---:|---:|---:|---:|---|
| `BTvnjJP` | 846.292 s | 1 (8/8) | 0.3089905974 | 42.706 us | 40 | `2633f7c7100b80f2a2d542d791ac258ac2267338f706221b2671977f772895f9` |
| `Lr9x9aW` | 828.845 s | 1 (8/8) | 0.3679073118 | 37.141 us | 96 | `7d139ef9b51751964f6bc19812d471f2bee6ea1a8db650a7d0d3480e7b6f6e09` |

隐藏 performance case：

| Trial | case | Candidate | Reference | speedup |
|---|---|---:|---:|---:|
| `BTvnjJP` | random-mixed-accepted | 43.201 us | 13.379 us | 0.309691x |
| `BTvnjJP` | permuted-checkpoints | 43.147 us | 13.385 us | 0.310222x |
| `BTvnjJP` | structured-head-map | 43.191 us | 13.154 us | 0.304547x |
| `BTvnjJP` | accepted-boundaries | 42.937 us | 13.377 us | 0.311549x |
| `Lr9x9aW` | random-mixed-accepted | 37.151 us | 13.723 us | 0.369375x |
| `Lr9x9aW` | permuted-checkpoints | 37.125 us | 13.710 us | 0.369306x |
| `Lr9x9aW` | structured-head-map | 37.149 us | 13.505 us | 0.363540x |
| `Lr9x9aW` | accepted-boundaries | 37.150 us | 13.725 us | 0.369444x |

把这两次与前一节正常结束的 `bW73nds` 合并，只讨论相同 checksum 和相同
GPT-5.6/high 配置，correctness 为 `3/3`；reward 为 `0.308991--0.512485`，均值
`0.396461`，中位数 `0.367907`。这说明当前题目尚未形成 GPT-5.6 的 correctness
失败边界，但能分辨不同 rollout 的优化质量。

需要单独标记工具口径：两个新 Trial 都只提交纯手写 PTX，也没有使用高层编译器，
但它们通过 Codex 内置 `web_search` 查询了 NVIDIA 官方 PTX 文档。`BTvnjJP` 查询
PTX `pragma unroll`，`Lr9x9aW` 查询 `f32x2`、`tcgen05` 和 BF16 转换，并在最终 PTX
使用 `f32x2`。因此这两次可以作为“Direct-PTX + 在线官方文档”样本，不能作为严格
离线文档样本。容器网络 allowlist 没有阻止模型服务侧工具，说明未来若要冻结严格离线
setting，还必须在 Agent/CLI 层禁用 web search；只限制容器网络不够。

### 2026-08-23 Web Search 控制修复

个人 Harbor 副本已应用 `patches/harbor/0005-codex-web-search-policy.patch`。根据后续
setting 决策，Codex 默认启动参数包含 `-c web_search=live`；严格离线文档对照必须
显式覆盖为 `disabled`。对应 flag descriptor 单元测试为 `53 passed`。

仓库新增 `scripts/audit_direct_ptx_trajectory.py`。它对当前两份 repeat trajectory
分别报告 1 次和 4 次 Web Search，对 `bW73nds`、`YLkCbFW` 报告无调用。这些
repeat 样本现在与主 setting 一致，分类仍是“Direct-PTX + 在线文档”；若运行严格离线
对照，下一次校准必须确认生成命令含禁用参数，再用 `--policy strict-offline` 审计新的
trajectory。
