# `ptx-block-sparse-attention-forward-bf16-s1024-d128-b16` 验证记录

**状态：规格、独立 C truth、固定 PTX Reference、Oracle 和 verifier 已写入；四种
CSR/layout case 已由 B200 Harbor Oracle 与固定 Codex 候选通过，定向负例已拒绝，等待更强
Reference。**

## 任务边界

- `S=1024, D=128`，BF16 输入输出；
- 16-token query/key block，CSR 每行 1--8 个唯一 key block；
- hidden case 改变稀疏布局、顺序和数值；
- grid 1024、block 128、单 entry 完成 sparse QK/Softmax/PV；
- correctness 硬门槛后按同 ABI Reference speedup 连续给分。

## 固定来源

```text
reference.cu  210e850a8ae9ad467c01732a9f480647b4e9659b6c22ecea18486d4abdc11024
reference.ptx b5cd0274c11bb1f38a48580bf853c7608f4eb48ab012da9dfec39751d28a2ac2
truth C       d0ee4963973bdd892b098af006092d4118fd432bf455e38e30d486fa781d5423
```

## B200 Harbor Oracle（2026-08-23）

- task checksum：`cfd99319f84c6a55718f6c2cf57b585a27ca8147df9d11b6c44e096d0ad0b237`；
- Agent：Harbor Oracle；正常结束，无异常；总运行时间约 27 秒；
- GPU：`NVIDIA B200`；
- correctness：`1`，`single-block/varied/shuffled/random-values` 全部通过；
- 8 blocks/row performance case：Candidate `0.074609 ms`，Reference
  `0.074620 ms`；
- raw speedup：`1.0001443833`；capped reward：`1`；
- Job：
  `jobs/ptx-block-sparse-attention-forward-bf16-s1024-d128-b16-b200-oracle-20260823-025838/`。

## 待完成

固定 CUDA 13.2.1 v2 SIF 汇编结果：32 registers、1 barrier、512-byte static
shared memory，无 stack/spill。

`first-block-only.ptx` 同 entry 负例已通过 `ptxas`，并由 B200 verifier 在 `varied`
CSR case 因输出 index 2061 数值错误而拒绝；correctness/reward 均为 `0`。证据位于
`jobs/direct-ptx-negative-fixtures-20260823-032914/ptx-block-sparse-attention-forward-bf16-s1024-d128-b16/`。

### Codex/high 生成与 canonical B200 评分

- 无 GPU 临时副本仅移除硬件 healthcheck 并禁用 verifier；canonical task 未修改；
- Agent：Codex 0.148 / GPT-5.6 / high / online-docs；正常退出，约 `369.5 s`；
- 生成侧诊断 checksum：`6f63902e73e24e8ac903e73c4275e6281e9b93e7c7a85393d31a492f3f7155c2`；
- trajectory 为直接 PTX，实际 Web Search `0`；
- 固定 candidate SHA-256：
  `6ee2092a756f3ae33f3cae0ab6abf4ee29c5d11b3b7a27d8f7822f8b0fb1b924`；
- `sm_100a` 汇编通过：31 registers、1 barrier、1040-byte static shared、无 spill；
- 生成 Job：
  `jobs/ptx-block-sparse-attention-generation-only-codex-gpt56-high-online-docs-20260823/`。
- canonical task snapshot SHA-256：
  `a9788045789fe114eaf1f7fd66144b7a33dd9f94b7bef659c173eb9f6d4cc2bf`；该字段是评分脚本
  对 task 目录快照的哈希，不冒充 Harbor task checksum；
- B200 hidden verifier：`single-block/varied/shuffled/random-values` 全部通过，
  correctness `1`；
- Candidate `0.019621 ms`，Reference `0.074611 ms`，raw speedup
  `3.8025876391x`，capped reward `1`；
- 评分证据：
  `jobs/candidate-verification-block-sparse-generation-only-3xYXwxe-20260823-043945/`。

Candidate 明显快于当前固定 Reference，因此这次通过不能证明性能门槛足够强。下一步：

### Codex/high repeat2 生成证据（2026-08-23）

同一 generation-only checksum `6f63902e…` 又完成两次独立生成，均正常退出并通过
`ptxas`；生成阶段没有 GPU、Verifier 被禁用，所以目前不能把它们记为 correctness pass：

| Trial | Agent execution | Candidate SHA | online-docs 实际使用 | 静态实现特征 | B200 状态 |
|---|---:|---|---:|---|---|
| `3XGLujB` | `719.3 s` | `20c3ebe8…` | Web Search `0` | 32 registers、576-byte shared | correctness `1`；raw `3.527634x` |
| `hiyedih` | `358.0 s` | `3cb6e2b7…` | NVIDIA 官方文档搜索 `1` 次 | 32 registers、544-byte shared | correctness `1`；raw `5.482798x` |

生成证据：
`jobs/ptx-block-sparse-generation-only-repeat2-codex-gpt56-high-online-docs-20260823/`。
评分证据：
`jobs/candidate-verification-block-repeat-3XGLujB-20260823-075930/` 与
`jobs/candidate-verification-block-repeat-hiyedih-20260823-080006/`。

两次评分都使用同一 canonical task snapshot `a9788045…`，四个 hidden case 全过。
连同首轮，本题 Codex/high correctness 为 `3/3`，raw speedup 范围
`3.527634–5.482798x`；当前 Reference 明显不足以形成性能区分。

下一步：

1. 加强同 ABI Reference，再决定是否需要增加 hidden CSR case；
2. 用开放权重模型做独立校准。
