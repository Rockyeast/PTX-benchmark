# KDA full-dispatch BF16 D128 验证记录

**状态：verifier-validated + Codex-calibrated pilot（2026-08-23）；release_ready=false**

## 固定规格

- task：`ptx-kda-full-dispatch-bf16-d128`
- 硬件：NVIDIA B200，`sm_100a`
- setting：Direct-PTX + online PTX docs，correctness-only
- 上游：`Int21-AI/KDA-B200@4896cd2da4ca421829907ab9fd3550e3f84a141b`
- 覆盖：官方 30-shape matrix、T1～T6、七条可见 entry
- Agent / verifier 预算：7200 / 1800 秒

七条 entry 是 T1 split16、T1 split8、T2 split4、T3 lower-bound split4、
T4 split2、T5 split1、T6 split1。T3 固定 `HV=16`，其余固定 `HV=32`；
所有路径固定 `D=128,H=16`。

## 真值与 Oracle 独立性

隐藏真值由 `tests/kda_full_truth.c` 实现，不包含或调用 INT21 CUDA。它独立计算：

- Q/K L2 normalization；
- KDA state recurrence；
- 每个 token 后的 BF16 state rounding；
- accepted-token initial checkpoint 和任意 checkpoint slot；
- T3 `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))` gate；
- 完整 output 和 state pool。

Oracle 则由固定 INT21 revision 的七个 CUDA 文件分别编译成 PTX，再合成一个七入口
module。因而 Oracle 是待测实现，C truth 才是 correctness 标准，二者不会通过共享代码
自动互相放行。

## 静态证据

本地已通过：Python compile、Bash syntax、`gcc -Wall -Wextra -Werror`、
Direct-PTX release checker 和 `git diff --check`。远端 CUDA 13.2.1 `ptxas
-arch=sm_100a` 已成功汇编完整 Oracle 和三个负例。

固定文件 SHA-256：

```text
task.toml                 ef93fef2be2dbcde75aa16d54558dada84f4333acb4ecfdce36c589d9b9052aa
instruction.md            31005dc553710e9120064d24b17b1c2dbb33695ab6fc9dd24f0422bf19276888
solution/oracle.ptx       d98b0007826326c18921932bc102312de54348428d9687e3b8a5fed163f6cf71
tests/kda_full_truth.c    66870aeae3e41b94b423c6d820a63ea62a00a1640e93f55c3b97d01f84106b3c
verifier                  66c9449717133d659b93433ed7da8731e0dea2adf9c6bb2430e52b9defda738e
```

三个可汇编定向负例为：缺 T3 entry、缺 T5 entry、T6 错误 accepted-token
checkpoint 选择。只有三者均被 canonical hidden verifier 判为 correctness `0`，才算
verifier-validated。

## B200 Oracle 与负例

Canonical Harbor checksum：

```text
eb85be543a259e2c499a231ae3b3e5ee3f9c991aee951a34f708a317fd2da086
```

Oracle Job：

```text
jobs/ptx-kda-full-dispatch-bf16-d128-b200-oracle-20260823-124443/
trial: ptx-kda-full-dispatch-bf16-d128__T4fnCZ8
correctness/reward: 1/1
verified cases: 14
Harbor total runtime: 23.88 s
```

三个可汇编负例均被拒绝，证据位于：

```text
jobs/direct-ptx-negative-fixtures-20260823-124530/
missing-t3.ptx:                 correctness 0
missing-t5.ptx:                 correctness 0
full-wrong-accepted-t6.ptx:     correctness 0
```

其中第三个负例包含完整七条 entry，失败于 T6/N8 state 比较，不是缺符号或汇编失败。

## Codex 五次独立校准

固定 setting 为 Codex 0.148 / GPT-5.6 / `reasoning_effort=high` /
`web_search=live`。五次先在无 GPU diagnostic task 中独立生成，再把 artifact 串行送给
同一个 canonical B200 hidden verifier。diagnostic checksum 均为 `09ab582f…`，评分
时 canonical task snapshot 均为 `b699f945…`；task、instruction 和 verifier 在五次间
没有修改。

| Trial | Agent 秒 | Candidate SHA | Web Search | canonical correctness | 结果 |
|---|---:|---|---:|---:|---|
| `NRQ6qas` | 539.99 | `d830cdc8…` | 1 | 1 | 14 cases 全过 |
| `2gyD4Ur` | 373.09 | `341f2d49…` | 0 | 1 | 14 cases 全过 |
| `JMHhrm3` | 376.17 | `9e00fc2f…` | 0 | 1 | 14 cases 全过 |
| `Q65Hozi` | 404.89 | `f17e7387…` | 0 | 0 | T1/N8 illegal address |
| `xbQnA7p` | 274.01 | `0620e35f…` | 0 | 0 | T1/N8 state mismatch |

汇总：correctness `3/5 = 60%`，正常退出 `5/5`。所有 artifact 都能静态汇编并包含
七条 entry；两个失败均由真实 B200 动态 correctness 捕获，而不是 Agent timeout、
基础设施异常或缺文件。

五条 trajectory 均只创建 PTX 文件，没有 CUDA/Triton/Python 高层源码，没有安装或
下载编译工具。online-docs 审计为 `1` 次 NVIDIA 官方文档搜索、`0` 次 policy violation。

## 当前判断

这是当前第一道让 GPT-5.6/high 在五次独立 attempt 中不能全过的 task，说明七入口
系统契约确实比此前固定单入口题更有 correctness 区分度。但 `60%` 仍高于暂定的
`30%～50%` 目标，且尚未运行开放权重强模型，所以不能冻结经验难度或 release。不要
根据这五个已见结果事后修改 hidden cases 再复用同一批 candidate；下一步应保持该
checksum，补 K3/GLM 等模型校准，或把新的更难语义定义成独立 task。
