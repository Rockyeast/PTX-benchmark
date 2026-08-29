# `ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128` 验证记录

**状态：B200 Oracle、5/5 定向负例与 Codex/high pilot 均已通过。**

## 任务边界

该题是 `multi-entry-system` 难度探针，不与单入口 KDA T1/T6 的通过率合并。Agent 必须
在同一个 `/app/kernel.ptx` 中实现两条已独立验证过的 entry；Verifier 分别运行 T1
标准 recurrent decode 和 T6 speculative decode 的完整 correctness/performance
harness，再以联合 correctness 硬门槛和两条速度比的几何平均给分。

## 当前静态实现

- 复用 T1 的 6 个 correctness case、4 个 performance case 和独立 C truth；
- 复用 T6 的 8 个 correctness case、4 个 performance case 和独立 C truth；
- Oracle 是把两个已验证 PTX entry 合并成一个 `sm_100a` module；
- 最终 reward 仅在两条 entry 都正确时非零；
- Agent 上限为 6000 秒，Verifier 上限为 1200 秒；连同 setup/teardown 仍低于
  单进程 2.5 小时限制。

## 已完成的非 GPU 验证

- `scripts/check_direct_ptx_task.py`：通过；
- Python `py_compile`、Shell `bash -n`、TOML 解析：通过；
- 固定 CUDA 13.2.1 / Codex 0.148 v2 SIF 中 `ptxas -arch=sm_100a`：通过；
- 合并 Oracle 同时编译出两条 entry：T1 使用 40 registers/0 barriers，T6 使用
  123 registers/2 barriers，均无 stack/spill。
- `reference_t1.ptx` 与 `reference_t6.ptx` 分别作为“缺失另一条 entry”的可汇编负例；
  尚待 GPU verifier 确认拒绝。

## B200 Harbor Oracle（2026-08-23）

- task checksum：`a062e49b4db1a82635d867713e92af4a23a7b237a9f640cb869d2a0c25be2301`；
- Agent：Harbor Oracle；正常结束，无异常；总运行时间约 43 秒；
- hardware preflight：`NVIDIA B200`；
- correctness：T1 和 T6 均通过，联合 correctness `1`；
- T1 raw speedup：`1.0012875671x`；
- T6 raw speedup：`0.9716963982x`；
- 联合 raw speedup / reward：
  `sqrt(1.0012875671 * 0.9716963982) = 0.9863810230`；
- Job：
  `jobs/ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128-b200-oracle-20260823-025304/`。

本次结果证明同一个 PTX module 能被两套不同 launch ABI 正常加载并执行，也证明联合
correctness/performance 聚合链路可用。它还不证明 verifier 能拒绝所有定向错误，也不
构成 Agent 难度校准。

## Codex/high/online-docs pilot（2026-08-23）

由于当时 8 张共享 B200 都被占用，本次把生成与评分拆成两个可审计阶段：

1. 在只把 `required_gpus` 改为 `0`、移除硬件 healthcheck 且禁用 verifier 的临时诊断
   副本中运行 Codex；题目 instruction、Direct-PTX 工具、Agent timeout 与 online-docs
   setting 均不变；
2. 将该次生成的**同一份** `kernel.ptx` 送入未修改的 canonical B200 hidden verifier。

生成阶段证据：

- Agent：Codex 0.148 / GPT-5.6 / high / `web_search=live`；
- 正常退出；Agent execution 约 `448.3 s`；
- 生成侧诊断 checksum：`f508c51b9004980aaa116f522ab5343f36a216c6b1116d6dddbdbf0a6ca3d4e0`；
  该 checksum **不是** canonical task checksum，不与正式评分混用；
- trajectory 审计：纯 PTX；调用 `ptxas`/`cuobjdump`；没有 CUDA/Triton/高层源码；
  online-docs 可用，但本次实际 Web Search 调用数为 `0`；
- 最终 PTX SHA-256：
  `c4f86a6975ab025938750d32548bf102ec6d7aece40e91833ca4c97c33abe7c0`；
- `ptxas -arch=sm_100a` 通过且无 spill；T1 `39` registers，T6 `40` registers / `1` barrier；
- Job：
  `jobs/ptx-kda-dispatch-generation-only-codex-gpt56-high-online-docs-20260823/`。

canonical B200 评分阶段：

- canonical task checksum 仍为
  `a062e49b4db1a82635d867713e92af4a23a7b237a9f640cb869d2a0c25be2301`；
- physical GPU `4`，环境 policy 通过，评分进程正常结束；
- T1：6 个 correctness case 全过，raw speedup `1.0584812692x`；
- T6：8 个 correctness case 全过，raw speedup `0.3759507319x`；
- 联合 correctness `1`；联合 raw speedup / reward：
  `sqrt(1.0584812692 * 0.3759507319) = 0.6308223267`；
- 评分证据：
  `jobs/candidate-verification-kda-generation-only-gm5gepF-20260823-040804/`。

### 两次独立重复（同一 canonical snapshot）

为避免把单次成功当成稳定通过率，又运行了两个全新 no-GPU generation Trial；两次
Agent 均正常退出，trajectory 均为纯 PTX，online-docs 可用但实际 Web Search `0`：

| Trial | Agent execution | Candidate SHA | T1 raw | T6 raw | 联合 reward | correctness |
|---|---:|---|---:|---:|---:|---:|
| `WGkevLv` | `476.87 s` | `4e0908eb…` | `0.987360x` | `0.308481x` | `0.551890` | `1` |
| `BbgqBxw` | `436.82 s` | `500f4b75…` | `0.988814x` | `0.315044x` | `0.558140` | `1` |

- 生成 Job：
  `jobs/ptx-kda-dispatch-generation-only-repeat2-codex-gpt56-high-online-docs-20260823/`；
- 两次评分均使用 task snapshot SHA-256
  `1613b080ecd493db0337f50a43ad430d65f89be7df3d5701cd0005959fc6561b`，与首个拆分
  评分一致；该目录快照哈希不是 Harbor task checksum；
- 评分证据：
  `jobs/candidate-verification-kda-repeat-WGkevLv-20260823-055621/`、
  `jobs/candidate-verification-kda-repeat-BbgqBxw-20260823-055826/`。

合并首个 pilot 后，Codex/high correctness 为 `3/3`，联合 reward 范围
`0.551890–0.630822`、均值 `0.580284`。这说明当前 correctness 对 GPT-5.6/high 仍偏
容易，但 T6 相对固定 Reference 的性能差距稳定存在。

因此这道题当前已证明 Codex 能稳定完成 correctness，但 T6 性能明显落后于固定 Reference；
相比先前单入口题，它提供了更有用的性能区分。该拆分实验不等同于“一次 canonical
Harbor Trial 内完成 Agent+Verifier”，发布前仍应在资源允许时补一次完整 Trial。

## 定向负例（2026-08-23）

- `reference_t1.ptx`：只含 T1，因缺失 T6 entry 被拒绝；
- `reference_t6.ptx`：只含 T6，因缺失 T1 entry 被拒绝；
- 三份 T6 语义负例最初也只含 T6，因此第一次运行只能证明缺失 T1 会被拒绝，不能作为
  T6 语义证据；该无效证据已保留并明确标记；
- 修正后的队列在运行时临时合并正确 T1 + 错误 T6，不修改 canonical task checksum；
- `final-checkpoint-only`、`wrong-accepted`、`wrong-head-map` 均实际进入 T6 hidden case，
  因 state 数值不匹配被拒绝；
- 合计 5/5 有效负例被拒绝，reward/correctness 均为 `0`；
- 证据：
  `jobs/direct-ptx-negative-fixtures-20260823-032914/ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128/`。

## 待完成

1. 资源允许时补一次 canonical Harbor Agent+Verifier 单 Trial；
2. 用开放权重强模型做独立校准；
3. 增加开放权重模型和更强性能 Reference 后再决定是否冻结难度。
