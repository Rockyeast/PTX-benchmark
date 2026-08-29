# PTX-Benchmark 实验进度总表

**最后更新：2026-08-27**

这份文档只记录已经实际运行的实验与下一步进度，不负责介绍 task 设计。题目候选、
家族和规划见 [`../dataset/task-catalog.md`](../dataset/task-catalog.md)，单题完整证据见
[`../tasks/`](../tasks/)，原始 Harbor 输出位于仓库根目录的 `jobs/`。

## 维护规则

每次完成新的 Harbor Job 后更新本文件：

1. 在“实验批次”中追加一行，不覆盖历史结果；
2. 只有 task checksum、模型、reasoning effort、时间预算和 setting 相同的 Trial 才能
   合并计算通过率；
3. `correctness` 与“Agent 是否正常退出”分开记录。超时但 artifact 通过 verifier，
   必须同时写成 correctness pass 和 timeout；
4. capped reward 与 raw speedup 分开记录，不能用 reward `1` 代替真实性能数据；
5. 明确记录 `online-docs`、`strict-offline` 或历史未固定的 Web Search setting；
6. Oracle、负例、真实 Agent 和 release-ready 是不同证据，不互相替代；
7. 更新批次后，同步更新下面的“当前总览”和“下一步队列”。

状态统一使用：

```text
infrastructure -> feasibility -> verifier-validated -> agent-calibrated -> frozen
```

其中 `agent-calibrated` 必须注明实际跑过哪些模型；目前多数任务只完成了 Codex 校准。
GLM-5.2 目前只有一次可评测但未正常结束的 generation-only 诊断，不代表已经完成开放
权重模型校准。

## 当前总览

| Task | 硬件 | 当前阶段 | 已完成的真实 Agent 证据 | 当前判断 | 下一步 |
|---|---|---|---|---|---|
| `harbor-smoke` | 本地/L40 | infrastructure | Codex、Qwen smoke 均通过 | 只验证 Harbor→Agent→Verifier | 保留，不计 benchmark 分数 |
| `ptx-add-one` | B200/L40 | infrastructure | B200 Codex 通过；L40 Qwen 见下表 | Direct-PTX 最小链路已通 | 仅用于环境回归 |
| `ptx-vector-add` | B200 | agent-calibrated（Codex） | Codex `1/1`，raw speedup `1.000609x` | 基础性能校准题，区分度低 | 不进入正式难度分母 |
| `ptx-rmsnorm-forward-bf16-n4096` | B200 | feasibility | 尚无真实 Agent 校准 | Oracle 已通过，但验证证据不完整 | 补负例、INT21 对照和 Agent 校准 |
| `ptx-rope-forward-bf16-d128` | B200 | feasibility | 尚无真实 Agent 校准 | Oracle、隐藏 policy 和两个负例通过 | 补剩余负例、慢正确实现和 Agent 校准 |
| `ptx-silu-gate-forward-bf16-n4096` | B200 | agent-calibrated（Codex） | Codex `5/5`，全部正常退出 | easy anchor；当前 Reference 偏弱 | 开放权重模型校准；加强 Reference |
| `ptx-softmax-forward-fp32-n4096` | B200 | agent-calibrated（Codex） | Codex `5/5`，全部正常退出 | easy anchor；不满足 30%–50% 目标 | 开放权重模型校准 |
| `ptx-flash-attention-forward-bf16-s128-d128` | B200 | agent-calibrated（Codex） | Codex `5/5`，全部正常退出 | 结构更复杂，但 correctness 仍容易 | 加强 Reference 或增加单一语义难点 |
| `ptx-paged-decode-attention-bf16-hq32-hkv8-d128-p16` | B200 | agent-calibrated（Codex） | 开发批次 `5/5`；allowlist 复验 `1/1` | correctness 容易；600 秒有尾部 timeout | 加强 Reference；开放权重模型校准 |
| `ptx-kda-recurrent-decode-bf16-t1-h16-hv32-d128-n32` | B200 | agent-calibrated（Codex） | 当前 checksum `2/2`；语义等价结果 `3/3` | T=1 对 Codex 仍容易 | 不再重复 T1；保留为 KDA 基础题 |
| `ptx-kda-speculative-decode-bf16-t6-n8-h16-hv32-d128` | B200 | agent-calibrated（Codex） | 当前 v0.3 checksum `3/3` | correctness 未触及上限，但性能有差异 | 转向更难系统题或开放权重模型 |
| `ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128` | B200 | verifier-validated + agent-calibrated pilot（Codex） | 拆分生成/评分 `3/3`，均正常退出；负例 `5/5` 拒绝 | correctness 偏容易；联合 reward `0.551890–0.630822`，T6 性能稳定形成区分 | 补 canonical 单 Trial与开放权重模型 |
| `ptx-flash-attention-backward-bf16-s128-d128` | B200 | agent-calibrated pilot（Codex） | Codex `2/2`；Oracle `1/1`、负例 `2/2` 拒绝 | correctness 偏容易；raw `2.575370–13.054470x` 暴露 Reference 偏弱 | 加强 Reference；开放权重模型 |
| `ptx-block-sparse-attention-forward-bf16-s1024-d128-b16` | B200 | verifier-validated + agent-calibrated pilot（Codex） | Codex `3/3`；负例 `1/1` 拒绝 | CSR/layout correctness 偏容易；raw `3.527634–5.482798x` 暴露 Reference 偏弱 | 加强 Reference→开放权重模型 |
| `ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16` | B200 | verifier-validated + agent-calibrated pilot（Codex） | Codex `3/3`；负例 `2/2` 拒绝 | causal+GQA+page-table correctness 偏容易；raw geomean `1.991123–4.653339x` 暴露 Reference 偏弱 | 加强 Reference→开放权重模型 |
| `ptx-gemm-rowwise-fp8e4m3-m128-n128-k128` | B200 | verifier-validated + agent-calibrated pilot（Codex） | 拆分生成/评分 `3/3`；负例 `1/1` 拒绝 | correctness 偏容易；raw `1.547658–1.679253x`，Reference 仍偏弱 | 加强 Reference→开放权重模型 |
| `ptx-kda-full-dispatch-bf16-d128` | B200 | verifier-validated + agent-calibrated pilot（Codex） | Codex `3/5`，五次均正常退出；Oracle `1/1`、负例 `3/3` 拒绝 | 当前首道让 GPT-5.6/high 非全过的 correctness upper probe；60% 尚未达到 30%～50% 目标 | 保持 checksum；补开放权重模型校准，不事后改 hidden cases |

当前没有任何正式算子 task 达到 `frozen`。目前的主要结论不是“benchmark 已完成”，
而是 Harbor/Direct-PTX 评测闭环已经能够稳定构建多类题，但 GPT-5.6/high 对现有固定
单入口 task 的 correctness 仍普遍偏高。

`2026-08-24`～`2026-08-25` 的 Modal A100/H100 本地开发任务保存在 Git 忽略的
`local-dev/modal/` 及相关 Job 目录中。它们验证了开放权重 API 接入和 Agent-visible GPU
迭代机制，但尚未迁入正式 `tasks/`，因此不混入上表的正式 task 阶段判断。

## B200 主实验批次

| 日期 | Task / 批次 | 固定标识 | Agent setting | correctness | 正常退出 | 性能结果 | 证据 |
|---|---|---|---|---:|---:|---|---|
| 2026-08-20 | Vector Add 最终校准 | checksum `6b683aa3…` | Codex 0.148 / GPT-5.6 / high / 600 s；历史 Direct-PTX，trajectory 无 Web Search | `1/1` | `1/1` | raw `1.000609x`，reward `1` | [`ptx-vector-add-validation.md`](../tasks/ptx-vector-add-validation.md) |
| 2026-08-20 | RMSNorm Oracle feasibility | 早期 feasibility task | Oracle | `1/1` | `1/1` | reward `1` | [`task-catalog.md`](../dataset/task-catalog.md) |
| 2026-08-21 | RoPE Oracle + policy/负例 | manifest `937b3c37…` | Oracle；无真实 Agent | Oracle `1/1`；负例 `2/2` 被拒绝 | Oracle 正常 | reward 约 `0.9963–0.9995` | [`ptx-rope-forward-bf16-d128-validation.md`](../tasks/ptx-rope-forward-bf16-d128-validation.md) |
| 2026-08-22 | SiLU-gate 五次 pilot | content digest `9cc2ae34…` | Codex 0.148 / GPT-5.6 / high / 600 s；trajectory 无 Web Search | `5/5` | `5/5` | raw `1.301971–1.341005x`；均值 `1.322769x` | [`ptx-silu-gate-forward-bf16-n4096-validation.md`](../tasks/ptx-silu-gate-forward-bf16-n4096-validation.md) |
| 2026-08-22 | Softmax 五次 pilot | checksum `275b2270…` | Codex 0.148 / GPT-5.6 / high / 600 s；trajectory 无 Web Search | `5/5` | `5/5` | raw `0.994724–1.011935x`；reward 均值 `0.998472` | [`ptx-softmax-forward-fp32-n4096-validation.md`](../tasks/ptx-softmax-forward-fp32-n4096-validation.md) |
| 2026-08-22 | Flash Attention 五次 pilot | checksum `3c25ff4b…` | Codex 0.148 / GPT-5.6 / high / 600 s；trajectory 无 Web Search | `5/5` | `5/5` | raw `2.136241–2.521571x`；Reference 明显偏弱 | [`ptx-flash-attention-forward-bf16-s128-d128-validation.md`](../tasks/ptx-flash-attention-forward-bf16-s128-d128-validation.md) |
| 2026-08-22 | Paged Decode 五次开发批次 | checksum `86da56ac…` | Codex 0.148 / GPT-5.6 / high / 600 s；历史 public/开发 setting，trajectory 无 Web Search | `5/5` | `4/5` | raw `0.6213–2.6477x` | [`ptx-paged-decode-attention-bf16-hq32-hkv8-d128-p16-validation.md`](../tasks/ptx-paged-decode-attention-bf16-hq32-hkv8-d128-p16-validation.md) |
| 2026-08-22 | Paged Decode Agent allowlist 复验 | checksum `3928d3ef…` | Codex 0.148 / GPT-5.6 / high / 600 s；容器 allowlist，trajectory 无 Web Search | `1/1` | `1/1` | raw `2.2566x` | 同上 |
| 2026-08-22 | KDA T1 当前 checksum 重复 | checksum `4917f073…` | Codex 0.148 / GPT-5.6 / high / 600 s；trajectory 无 Web Search | `2/2` | `1/2` | raw `1.015362x`、`1.530861x` | [`ptx-kda-recurrent-decode-bf16-t1-h16-hv32-d128-n32-validation.md`](../tasks/ptx-kda-recurrent-decode-bf16-t1-h16-hv32-d128-n32-validation.md) |
| 2026-08-22～23 | KDA T6 Direct-PTX v0.3 | checksum `32e52d17…` | Codex 0.148 / GPT-5.6 / high / 3600 s；online-docs 可用，`2/3` trajectory 实际使用 Web Search | `3/3` | `3/3` | reward `0.308991–0.512485`；均值 `0.396461` | [`ptx-kda-speculative-decode-bf16-t6-n8-h16-hv32-d128-validation.md`](../tasks/ptx-kda-speculative-decode-bf16-t6-n8-h16-hv32-d128-validation.md) |
| 2026-08-23 | KDA T1+T6 multi-entry Oracle | checksum `a062e49b…` | Harbor Oracle / Direct-PTX v0.3 | `1/1` | `1/1` | T1 `1.001288x`，T6 `0.971696x`；联合 reward `0.986381` | [`ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128-validation.md`](../tasks/ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128-validation.md) |
| 2026-08-23 | KDA T1+T6 multi-entry Codex pilot | canonical checksum `a062e49b…`；candidate `c4f86a69…` | Codex 0.148 / GPT-5.6 / high；online-docs 可用、实际 Web Search `0`；无 GPU 生成后送 canonical B200 verifier | `1/1` | 生成与评分均正常 | T1 `1.058481x`，T6 `0.375951x`；联合 reward `0.630822` | 同上；`jobs/candidate-verification-kda-generation-only-gm5gepF-20260823-040804/` |
| 2026-08-23 | KDA T1+T6 multi-entry Codex repeat2 | canonical task snapshot `1613b080…`；candidate `4e0908eb…`、`500f4b75…` | Codex 0.148 / GPT-5.6 / high；online-docs 可用、两次实际 Web Search `0`；无 GPU 生成后分别送同一 canonical B200 verifier | `2/2` | `2/2` 生成与评分正常 | 联合 reward `0.551890`、`0.558140`；三次总计 correctness `3/3` | 同上；`jobs/candidate-verification-kda-repeat-*/` |
| 2026-08-23 | Flash Attention backward Oracle | checksum `f36175d5…` | Harbor Oracle / Direct-PTX v0.3 | `1/1` | `1/1` | raw / reward `0.999669`；dQ/dK/dV 全通过 | [`ptx-flash-attention-backward-bf16-s128-d128-validation.md`](../tasks/ptx-flash-attention-backward-bf16-s128-d128-validation.md) |
| 2026-08-23 | Flash Attention backward Codex pilot | canonical checksum `f36175d5…`；candidate `f796b607…` | Codex 0.148 / GPT-5.6 / high；online-docs，2 次官方文档搜索；无 GPU 生成后送 canonical B200 verifier | `1/1` | 生成与评分均正常 | raw `2.575370x`；capped reward `1`；Reference 偏弱 | 同上；`jobs/candidate-verification-flash-bwd-generation-only-HBi4g6L-20260823-042942/` |
| 2026-08-23 | Flash Attention backward Codex repeat | canonical task snapshot `66b5870b…`；candidate `90071bb7…` | Codex 0.148 / GPT-5.6 / high；online-docs，2 次官方文档搜索；无 GPU 生成后送同一 canonical B200 verifier | `1/1` | 生成与评分均正常 | raw `13.054470x`；capped reward `1`；PTX-level MMA 暴露 Reference 很弱 | 同上；`jobs/candidate-verification-flash-bwd-repeat-8Dj8RLV-20260823-080041/` |
| 2026-08-23 | Block-sparse Attention Oracle | checksum `cfd99319…` | Harbor Oracle / Direct-PTX v0.3 | `1/1` | `1/1` | raw `1.000144x`；capped reward `1` | [`ptx-block-sparse-attention-forward-bf16-s1024-d128-b16-validation.md`](../tasks/ptx-block-sparse-attention-forward-bf16-s1024-d128-b16-validation.md) |
| 2026-08-23 | Block-sparse Attention Codex pilot | canonical task snapshot `a9788045…`；candidate `6ee2092a…` | Codex 0.148 / GPT-5.6 / high；online-docs 可用、实际 Web Search `0`；无 GPU 生成后送 canonical B200 verifier | `1/1` | 生成与评分均正常 | raw `3.802588x`；capped reward `1`；Reference 偏弱 | 同上；`jobs/candidate-verification-block-sparse-generation-only-3xYXwxe-20260823-043945/` |
| 2026-08-23 | Block-sparse Attention Codex repeat2 | canonical task snapshot `a9788045…`；candidate `20c3ebe8…`、`3cb6e2b7…` | Codex 0.148 / GPT-5.6 / high；online-docs，实际 Web Search `0/1`；无 GPU 生成后分别送同一 canonical B200 verifier | `2/2` | `2/2` 生成与评分正常 | raw `3.527634x`、`5.482798x`；三次总计 correctness `3/3` | 同上；`jobs/candidate-verification-block-repeat-*/` |
| 2026-08-23 | Paged Prefill Attention Oracle | checksum `00e540e0…` | Harbor Oracle / Direct-PTX v0.3 | `1/1` | `1/1` | 3-shape geometric raw / reward `0.999978` | [`ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16-validation.md`](../tasks/ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16-validation.md) |
| 2026-08-23 | Paged Prefill Attention Codex pilot | canonical task snapshot `a278ff75…`；candidate `7e55bf2a…` | Codex 0.148 / GPT-5.6 / high；online-docs 可用、实际 Web Search `0`；无 GPU 生成后送 canonical B200 verifier | `1/1` | 生成与评分均正常 | context `128/512/1024` raw `2.721006/3.603968/3.758730x`；geomean `3.328004x`；capped reward `1` | 同上；`jobs/candidate-verification-paged-prefill-generation-only-yFD8j5o-20260823-051822/` |
| 2026-08-23 | Paged Prefill Attention Codex repeat（已评分部分） | canonical task snapshot `a278ff75…`；candidate `9dc82ba2…` | Codex 0.148 / GPT-5.6 / high；online-docs 可用、实际 Web Search `0`；无 GPU 生成后送同一 canonical B200 verifier | `1/1` | 生成与评分均正常 | context `128/512/1024` raw `1.623676/2.167932/2.242588x`；geomean `1.991123x` | 同上；`jobs/candidate-verification-paged-repeat-w6HDVeX-20260823-070628/` |
| 2026-08-23 | Paged Prefill Attention Codex repeat `u257Gsp` | canonical task snapshot `a278ff75…`；candidate `ca4f7560…` | Codex 0.148 / GPT-5.6 / high；online-docs 可用、实际 Web Search `0`；无 GPU 生成后送同一 canonical B200 verifier | `1/1` | 生成与评分均正常 | context `128/512/1024` raw `3.321180/5.323319/5.699271x`；geomean `4.653339x` | 同上；`jobs/candidate-verification-paged-repeat-u257Gsp-20260823-070732/` |
| 2026-08-23 | BF16 GEMM + FP8 E4M3 row quant Oracle | checksum `5c88a688…` | Harbor Oracle / Direct-PTX v0.3 | `1/1` | `1/1` | raw `1.000087x`；capped reward `1` | [`ptx-gemm-rowwise-fp8e4m3-m128-n128-k128-validation.md`](../tasks/ptx-gemm-rowwise-fp8e4m3-m128-n128-k128-validation.md) |
| 2026-08-23 | BF16 GEMM + FP8 E4M3 Codex pilot | canonical task snapshot `f10de262…`；candidate `2539bbf5…` | Codex 0.148 / GPT-5.6 / high；online-docs，实际 Web Search `5`；无 GPU 生成后送 canonical B200 verifier | `1/1` | 生成与评分均正常 | raw `1.551661x`；capped reward `1`；Reference 偏弱 | 同上；`jobs/candidate-verification-fp8-generation-only-HXJcPzJ-20260823-052311/` |
| 2026-08-23 | BF16 GEMM + FP8 E4M3 Codex repeat2 | canonical task snapshot `f10de262…`；candidate `f86e83f2…`、`89a69775…` | Codex 0.148 / GPT-5.6 / high；online-docs，实际 Web Search `4/6`；无 GPU 生成后分别送 canonical B200 verifier | `2/2` | `2/2` 生成与评分正常 | raw `1.547658x`、`1.679253x`；三次总计 correctness `3/3` | 同上；`jobs/candidate-verification-fp8-repeat-*/` |
| 2026-08-23 | KDA full dispatch Oracle + negatives | canonical checksum `eb85be54…` | Harbor Oracle / Direct-PTX v0.3；correctness-only | Oracle `1/1`；负例 `3/3` 拒绝 | Oracle 正常 | reward `1`；缺 T3/缺 T5/错 accepted 均 reward `0` | [`ptx-kda-full-dispatch-bf16-d128-validation.md`](../tasks/ptx-kda-full-dispatch-bf16-d128-validation.md) |
| 2026-08-23 | KDA full dispatch Codex 5-attempt calibration | diagnostic checksum `09ab582f…`；canonical snapshot `b699f945…`；五个不同 candidate SHA | Codex 0.148 / GPT-5.6 / high / online-docs；无 GPU 独立生成后串行 B200 评分；实际 Web Search `1/0/0/0/0` | `3/5` | `5/5` 正常生成与评分 | correctness-only reward：`1/1/1/0/0`；失败为 illegal address、state mismatch | 同上；`jobs/candidate-verification-kda-full-*/` |

上表只聚合同一行内部的固定配置。不同 checksum 的 feasibility、旧 SIF 或 timeout
诊断仍保留在各自 validation 文档和 `jobs/`，不混入当前通过率。

## 无 GPU 生成诊断（不计 correctness）

这些 Job 只利用共享 B200 全忙时的等待时间生成并静态汇编 PTX。它们的 task 副本移除
硬件 healthcheck 且禁用 verifier，故诊断 checksum 与 canonical checksum 不同。只有
同一 candidate SHA 后续通过 canonical B200 hidden verifier，才会进入上面的正式
Agent 证据。

| 日期 | Canonical task | 生成侧 checksum | Candidate SHA | Agent / 时间 / Web Search | 当前状态 |
|---|---|---|---|---|---|
| 2026-08-23 | Block-sparse Attention | `6f63902e…` | `6ee2092a…` | Codex 0.148 / GPT-5.6 / high / `369.5 s` / `0` | 已送 canonical B200 verifier，通过 `4/4` hidden cases；见主表 |
| 2026-08-23 | Paged Prefill Attention | `de3d144a…` | `7e55bf2a…` | Codex 0.148 / GPT-5.6 / high / `572.6 s` / `0` | 已送 canonical B200 verifier，通过 `4/4` correctness lengths；见主表 |
| 2026-08-23 | BF16 GEMM + FP8 E4M3 | `9c2e567b…` | `2539bbf5…` | Codex 0.148 / GPT-5.6 / high / `606.3 s` / `5` | 已送 canonical B200 verifier，通过 `4/4` correctness cases；见主表 |
| 2026-08-23 | Paged Prefill repeat `w6HDVeX` | `de3d144a…` | `9dc82ba2…` | Codex 0.148 / GPT-5.6 / high / `362.4 s` / `0` | canonical B200 correctness `1`；3-shape geomean raw `1.991123x` |
| 2026-08-23 | Paged Prefill repeat `u257Gsp` | `de3d144a…` | `ca4f7560…` | Codex 0.148 / GPT-5.6 / high / `507.2 s` / `0` | 已送 canonical B200 verifier，correctness `1`、geomean raw `4.653339x` |
| 2026-08-23 | Block-sparse repeat `3XGLujB` | `6f63902e…` | `20c3ebe8…` | Codex 0.148 / GPT-5.6 / high / `719.3 s` / `0` | canonical B200 correctness `1`；raw `3.527634x` |
| 2026-08-23 | Block-sparse repeat `hiyedih` | `6f63902e…` | `3cb6e2b7…` | Codex 0.148 / GPT-5.6 / high / `358.0 s` / `1` | canonical B200 correctness `1`；raw `5.482798x` |
| 2026-08-23 | Flash backward repeat `8Dj8RLV` | `9f2d1a1b…` | `90071bb7…` | Codex 0.148 / GPT-5.6 / high / `604.5 s` / `2` | canonical B200 correctness `1`；raw `13.054470x` |
| 2026-08-23 | KDA full dispatch pilot `NRQ6qas` | `09ab582f…` | `d830cdc8…` | Codex 0.148 / GPT-5.6 / high / `540.0 s` / `1` | 已送 canonical B200 verifier，correctness `1` |
| 2026-08-23 | KDA full dispatch repeat4 | `09ab582f…` | `341f2d49…` / `9e00fc2f…` / `f17e7387…` / `0620e35f…` | Codex 0.148 / GPT-5.6 / high / `274.0–404.9 s` / Web Search 均 `0` | canonical correctness `1/1/0/0`；全部正常生成，两个动态失败 |

## L40 / Qwen 环境校准批次

这些实验验证 Lovelace 单卡 L40 上部署的 `Qwen3.8-27B-FP8` 和 Harbor Docker
链路，不计入 B200 正式 benchmark 分数。

| 日期 | Task / 批次 | 模型与预算 | correctness | 结束方式 | 结论 / 失败原因 | Job |
|---|---|---|---:|---|---|---|
| 2026-08-21 | Harbor smoke | Qwen3.8 / 131K context | `1/1` | 正常 | 模型 API、工具调用和 Harbor verifier 链路通过 | `jobs/harbor-smoke-qwen38-131k/` |
| 2026-08-21 | Add-one hardened 600 s | Qwen3.8 / 600 s | `0/1` | 约 355 s 正常 | PTX `shl.b64` 操作数位宽错误；不是服务吞吐瓶颈 | `jobs/ptx-add-one-l40-qwen38-allowlist-max1024-20260821/` |
| 2026-08-21 | Add-one 当前 digest 三次重复 | Qwen3.8 / 600 s | `0/3` | `3/3` timeout | 三次最终 PTX 均未完成汇编修正 | `jobs/ptx-add-one-l40-qwen38-direct-v0-600-repeat3-current-digest-20260821/` |
| 2026-08-21 | Add-one 1200 s 诊断 | Qwen3.8 / 1200 s | `1/1` | 正常，约 570 s | 能自行修正 PTX 并通过 4 个 case；说明 600 s 方差/余量不足 | `jobs/ptx-add-one-l40-qwen38-direct-v0-diagnostic-timeout1200-20260821/` |
| 2026-08-22 | Vector Add 1200 s | Qwen3.8 / 1200 s | `0/1` | 正常，约 1069 s | 输出数值错误，reward `0` | `jobs/ptx-vector-add-l40-qwen38-direct-v02-timeout1200-20260822/` |
| 2026-08-22～23 | SiLU-gate 两次 1200 s | Qwen3.8 / 1200 s | `0/2` | `2/2` timeout | 候选 PTX 无效或无法加载，reward 均为 `0` | `jobs/ptx-silu-gate-l40-qwen38-direct-v02-*/` |

L40 结果说明单卡能够稳定承载模型服务，但 Qwen3.8 的 Agent 时间管理和 PTX 首版
正确性仍是主要瓶颈。它不能被 B200 Codex 结果替代，也不能与 B200 task 分数直接平均。

## A100 / GLM-5.2 generation-only 诊断

这次诊断发生在 Agent-visible GPU 接口完成之前。Agent 在无 GPU 的 Modal sandbox 中
直接编写 PTX，只能用 `ptxas`/`cuobjdump` 做静态检查；最终 artifact 另送 A100 隐藏
verifier。它不能与后面的 Agent-visible H100 结果合并。

| 日期 | Task | 模型 / Agent | Agent GPU | 可评测 correctness | Agent 结束方式 | 结论 | Job |
|---|---|---|---:|---:|---|---|---|
| 2026-08-24 | FP32 Softmax N4096 / A100 | GLM-5.2 / qwen-coder | 否 | `0/1` | API `429`；artifact 已在此前生成 | PTX 可汇编，但 A100 隐藏 case `num_rows=3` 数值错误；只证明 GLM API→Harbor→Modal 诊断链路可用 | `jobs/glm52-softmax-a100-generation-only-20260824/` |

前四次 bring-up 中，前三次属于 Modal/Harbor 依赖与 allowlist 兼容问题，不计作模型
错误；只有第四次产生可评测 Candidate。第四次也不是正常结束的独立 attempt，因此当前不能
报告 GLM-5.2 的稳定通过率。重复校准仍受 API 余额或资源包阻塞。

## Modal H100 Agent-visible GPU 开发批次

`2026-08-25` 起，本地 Modal 运行器已经让 Agent 和后置 verifier 共用同一个 H100
sandbox。Agent 可调用 `direct-ptx-benchmark-candidate /app/kernel.ptx`，在公开开发输入
上运行 correctness、Compute Sanitizer 和 CUDA Event 计时，然后继续修改 PTX；隐藏
verifier 仍在 Agent 结束后独立评分。Modal runner 与这些本地硬件变体位于 Git 忽略的
`local-dev/modal/`，不是 Harbor 正式 task 源码或 release 结果。

KDA forward+backward 的最终 smoke 使用 checksum `e21b23ef…`。Codex/GPT-5.6
在 H100 上通过公开 correctness 和内存检查，比较两个实现后把公开 forward+backward
中位时间从约 `154.33 us` 改善到 `151.56 us`；后置 hidden correctness 为 `1`。这证明
Agent-visible GPU 迭代闭环已经实际跑通，但该 local smoke 的隐藏 reward 仍是
correctness-only，不是正式性能校准。

随后运行的十道 local H100 开发题如下。每行只有一次完成的 Codex attempt；初始没有
完成 trial 的启动 Job 不计入分母。

| 日期 | Local task / checksum | 模型与 setting | correctness | 正常退出 | raw speedup / reward | 当前判断 | Job |
|---|---|---|---:|---:|---|---|---|
| 2026-08-25 | Grouped GEMM `15eca1f7…` | Codex 0.149.1 / GPT-5.6 / Agent-visible H100 | `1/1` | `1/1` | `1.579174x` / `1` | 本地 feasibility pilot | `jobs/next10-agent-grouped-gemm-h100-codex-1/` |
| 2026-08-25 | MoE Top-2 routing `75a8c7ac…` | 同上 | `1/1` | `1/1` | `1.511609x` / `1` | 公开 GPU harness 中实际迭代优化 | `jobs/next10-agent-moe-top2-h100-codex-2/` |
| 2026-08-25 | Top-K/Top-P renorm `7bafb906…` | 同上 | `1/1` | `1/1` | `814.438273x` / `1` | Reference 极弱，性能数字不可冻结 | `jobs/next10-agent-topk-topp-h100-codex-2/` |
| 2026-08-25 | Varlen FlashAttention `8b72526a…` | 同上 | `1/1` | `1/1` | `2.680306x` / `1` | 本地 feasibility pilot | `jobs/next10-agent-varlen-attn-h100-codex-1/` |
| 2026-08-25 | Paged KV FP8 gather/dequant `01b660b7…` | 同上 | `1/1` | `1/1` | `1.012004x` / `1` | 本地 feasibility pilot | `jobs/next10-agent-paged-kv-h100-codex-1/` |
| 2026-08-25 | KV append + RoPE `0bf63961…` | 同上 | `1/1` | `1/1` | `1.011690x` / `1` | 本地 feasibility pilot | `jobs/next10-agent-kv-rope-h100-codex-1/` |
| 2026-08-25 | FP8 group quant `4243da09…` | 同上 | `1/1` | `1/1` | `1.002289x` / `1` | 本地 feasibility pilot | `jobs/next10-agent-fp8-quant-h100-codex-1/` |
| 2026-08-25 | LayerNorm forward+backward `da3c29fd…` | 同上 | `1/1` | `1/1` | `27.335015x` / `1` | Reference 很弱，性能数字不可冻结 | `jobs/next10-agent-layernorm-fb-h100-codex-1/` |
| 2026-08-25 | Softmax forward+backward `4f95c51e…` | 同上 | `1/1` | `1/1` | `48.661649x` / `1` | Reference 很弱，性能数字不可冻结 | `jobs/next10-agent-softmax-fb-h100-codex-1/` |
| 2026-08-25 | Fused MoE SwiGLU `ee0d7f40…` | 同上 | `1/1` | `1/1` | `167.167156x` / `1` | Reference 极弱，性能数字不可冻结 | `jobs/next10-agent-fused-moe-h100-codex-1/` |

本批次的 correctness 为 `10/10`，说明 Agent-visible GPU 和公开迭代接口工作正常，
但不能据此宣称十道题已经完成正式 Agent 校准：它们仍位于 local Job 树、每题只有一次
attempt，并且多个异常高 speedup 明确暴露出 feasibility Reference 过弱。后续应先挑选
少量有代表性的题迁入正式 `tasks/`，补 Oracle、负例、可信 Reference 和多模型重复，
而不是直接把本批次作为 release 分数。

## 当前下一步队列

| 优先级 | 工作 | 完成标准 | 状态 |
|---:|---|---|---|
| 1 | 保存在线文档 setting、Harbor `0005` 补丁和本实验总表 | 独立 commit；相关测试通过；push 到个人分支 | deferred：overnight goal 明确不 commit/push |
| 2 | 冻结下一轮 Agent setting | 默认 `online-docs`；记录 `web_search=live`；trajectory 审计只记录而不判违规 | completed |
| 3 | KDA T1+T6 multi-entry 系统题 | 双 entry correctness/性能联合门控；Oracle 与 Agent | completed pilot：Oracle `1/1`、负例 `5/5`、Codex correctness `3/3` / reward `0.551890–0.630822` |
| 4 | FlashAttention backward BF16 S128 D128 | dQ/dK/dV 独立 truth；Oracle 与 Agent | completed pilot：Oracle `1/1`、负例 `2/2`、Codex correctness `1` / raw `2.575370x` |
| 5 | Block-sparse attention BF16 S1024 D128 | CSR hidden cases；Oracle 与 Agent | completed pilot：Oracle `1/1`、负例 `1/1`、Codex correctness `1` / raw `3.802588x` |
| 6 | Paged prefill attention BF16 Q128 | causal/GQA/page-table hidden cases；Oracle 与 Agent | completed pilot：Oracle `1/1`、负例 `2/2`、Codex correctness `1` / geomean raw `3.328004x` |
| 7 | BF16 GEMM + FP8 E4M3 row quant | scale/encoding/full GEMM truth；Oracle 与 Agent | completed pilot：Oracle `1/1`、负例 `1/1`、Codex correctness `3/3` / raw `1.547658–1.679253x` |
| 8 | 用开放权重强模型校准现有题组 | K3/GLM 等固定配置，多次独立 attempt | partial diagnostic：GLM-5.2 Softmax 产生一次可评测 Candidate，correctness `0/1` 且 API `429` 异常结束；仍缺稳定 API/部署与重复 |
| 9 | 修补现有任务证据缺口 | RoPE/RMSNorm 补验证；Flash/Paged/KDA 加强 Reference | pending |
| 10 | 冻结首个 release manifest | 多模型结果、硬件/setting 固定、release checker 通过 | pending |
| 11 | KDA full-dispatch correctness upper probe | 七 entry、30-shape ABI；Oracle/3 负例/5 次 Codex | completed pilot：Oracle `1/1`、负例 `3/3`、Codex `3/5`；待开放权重模型 |
| 12 | Modal H100 Agent-visible GPU 迭代链路 | Agent 可运行公开 correctness/性能 harness，结束后由隐藏 verifier 评分 | completed infrastructure pilot：KDA smoke 跑通；十道 local H100 开发题均完成 `1/1` Codex attempt，但尚非正式 task 校准 |

## 新实验追加模板

复制下面一行到对应批次表中填写：

```markdown
| YYYY-MM-DD | task / batch | 完整或缩写 checksum | Agent / model / effort / timeout / web setting | pass/attempts | normal/attempts | raw speedup 与 reward | validation / Job |
```

若新结果改变了任务判断，再修改“当前总览”；如果只是同一批次增加 attempt，只更新该
批次统计与对应 validation 文档，不在本文件复制完整 per-trial timing。
