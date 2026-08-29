# `ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16` 验证记录

**状态：规格、独立 C truth、固定 PTX Reference、Oracle 和 verifier 已写入；四种
context/page case 已由 B200 Harbor Oracle 与固定 Codex 候选通过，两个定向负例已拒绝，等待更强
Reference。**

任务固定 Q=128、Hq=8、Hkv=2、D=128、page size=16；hidden case 改变 context
length 与 page permutation。Correctness 对完整输出、输入不可变性和 guard 硬门控，
之后对 context 128/512/1024 的 speedup 取几何平均。

固定 CUDA 13.2.1 v2 SIF 汇编结果：32 registers、1 barrier、4096-byte static
shared memory，无 stack/spill。

`ignore-page-table.ptx` 与 `non-causal.ptx` 两个同 entry 负例均已通过 `ptxas`，并由
B200 verifier 在 KV length 128 hidden case 拒绝：前者 index 0 得到 `32.0`，后者得到
`0.03125`，truth 均为 `-2.0`；correctness/reward 均为 `0`。证据位于
`jobs/direct-ptx-negative-fixtures-20260823-032914/ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16/`。

## B200 Harbor Oracle（2026-08-23）

- task checksum：`00e540e005c330de3c72ef3a91c1fbe255cdff51a142a76fe4af6f7eaa22109e`；
- 正常结束、无异常，总运行时间约 29 秒，GPU 为 `NVIDIA B200`；
- correctness：`1`，KV length `128/256/512/1024` 全部通过；
- performance case `128/512/1024` 的 speedup 分别为
  `1.000023/0.999944/0.999968`；
- 几何平均 raw speedup / reward：`0.9999781823`；
- Job：
  `jobs/ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16-b200-oracle-20260823-025926/`。

## Codex/high 生成与 canonical B200 评分

- 无 GPU 临时副本只移除硬件 healthcheck 并禁用 verifier；canonical task 未修改；
- Agent：Codex 0.148 / GPT-5.6 / high / online-docs；正常退出，约 `572.6 s`；
- 生成侧诊断 checksum：`de3d144af267cf7afb9f157e8a148b875fe7d18b4f6fa84bc78c4bf02dc98a7b`；
- trajectory 为直接 PTX，实际 Web Search `0`；
- 固定 candidate SHA-256：
  `7e55bf2a2ebf491477103146ca53ad6c9000c20a92bfa00620cf31019a99abec`；
- `sm_100` 汇编通过，无 spill；最终静态报告为 30 registers、4112-byte static
  shared、1 次 block-wide barrier；
- 生成 Job：
  `jobs/ptx-paged-prefill-attention-generation-only-codex-gpt56-high-online-docs-20260823/`。

- canonical task snapshot SHA-256：
  `a278ff75dc749912bf11258184aae207b704d69a3f47244f19e0f15302e00898`；该字段是评分脚本
  对 task 目录快照的哈希，不冒充 Harbor task checksum；
- B200 hidden verifier：KV length `128/256/512/1024` 全部通过，correctness `1`；
- performance case `128/512/1024` 的 speedup 分别为
  `2.721006/3.603968/3.758730`；几何平均 raw speedup `3.3280042530x`，
  capped reward `1`；
- 评分证据：
  `jobs/candidate-verification-paged-prefill-generation-only-yFD8j5o-20260823-051822/`。

Candidate 明显快于固定 Reference，因此当前性能门槛仍偏弱。待完成：

## Codex/high repeat2（2026-08-23）

同一 generation-only checksum `de3d144a…` 又完成两次独立生成，两次均正常退出、
直接提交 PTX、实际 Web Search `0`：

| Trial | Agent execution | Candidate SHA | 静态资源 | canonical B200 结果 |
|---|---:|---|---|---|
| `w6HDVeX` | `362.4 s` | `9dc82ba2…` | 25 registers、4640-byte shared | correctness `1`；128/512/1024 raw `1.623676/2.167932/2.242588x`；geomean `1.991123x` |
| `u257Gsp` | `507.2 s` | `ca4f7560…` | 28 registers、2080-byte shared | correctness `1`；128/512/1024 raw `3.321180/5.323319/5.699271x`；geomean `4.653339x` |

生成证据：
`jobs/ptx-paged-prefill-generation-only-repeat2-codex-gpt56-high-online-docs-20260823/`；
评分证据：
`jobs/candidate-verification-paged-repeat-w6HDVeX-20260823-070628/` 与
`jobs/candidate-verification-paged-repeat-u257Gsp-20260823-070732/`。

连同首轮，本题同一 canonical task snapshot 的 Codex/high 候选 correctness 为
`3/3`；三次 raw geomean 范围 `1.991123–4.653339x`。这进一步证明当前
Reference 只适合 correctness/基础性能链路验证，不能冻结性能排名。

待完成：

1. 加强同 ABI Reference；
2. 用开放权重模型做独立校准。
