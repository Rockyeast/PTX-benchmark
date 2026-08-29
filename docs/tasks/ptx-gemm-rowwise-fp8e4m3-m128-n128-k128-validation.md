# `ptx-gemm-rowwise-fp8e4m3-m128-n128-k128` 验证记录

**状态：规格、独立 FP32 GEMM truth、固定 PTX Reference、Oracle 和 verifier 已写入；
四种数值 case 已由 B200 Harbor Oracle 与固定 Codex 候选通过，定向负例已拒绝，等待更强
Reference。**

该题固定 M=N=K=128，BF16 A/B，FP32 accumulate；每行计算 `amax/448` scale，
输出有限 E4M3 bytes。Verifier 解码每个 FP8 值并验证 scale，不把 Oracle 当正确性真值。

固定 CUDA 13.2.1 v2 SIF 汇编结果：30 registers、1 barrier、516-byte static
shared memory，无 stack/spill。

`wrong-scale.ptx` 同 entry 负例已通过 `ptxas`，并由 B200 verifier 在 `identity` case
拒绝：row scale 期望 `0.0030691964`，错误候选得到 `0.0108267721`；
correctness/reward 均为 `0`。证据位于
`jobs/direct-ptx-negative-fixtures-20260823-032914/ptx-gemm-rowwise-fp8e4m3-m128-n128-k128/`。

## B200 Harbor Oracle（2026-08-23）

- task checksum：`5c88a688d0f36cd45b44acf26393417569dd9d08fe61cd50d799ed02da868d85`；
- 正常结束、无异常，总运行时间约 24 秒，GPU 为 `NVIDIA B200`；
- correctness：`1`，`zeros/identity/random/wide-range` 全部通过；
- Candidate median `0.011810 ms`，Reference median `0.011811 ms`；
- raw speedup：`1.0000867236`；capped reward：`1`；
- Job：
  `jobs/ptx-gemm-rowwise-fp8e4m3-m128-n128-k128-b200-oracle-20260823-030017/`。

## Codex/high 生成与 canonical B200 评分

- 无 GPU 临时副本只移除硬件 healthcheck 并禁用 verifier；canonical task 未修改；
- Agent：Codex 0.148 / GPT-5.6 / high / online-docs；正常退出，约 `606.3 s`；
- 生成侧诊断 checksum：`9c2e567bb288e9ee70e31f0a8ad192cdae58f61176740284d410124642123a12`；
- trajectory 为直接 PTX；为确认官方 MMA fragment layout 实际使用 Web Search `5` 次，
  无 Direct-PTX 违规；
- 固定 candidate SHA-256：
  `2539bbf5eed41039d9186f28837c2aece514ac69c1aa186f578a71512a520bf3`；
- `sm_100a` 汇编通过：56 registers、788-byte declared static shared、无 spill；
- SASS 确认原生 BF16 HMMA 与 saturating finite E4M3 转换；
- 生成 Job：
  `jobs/ptx-gemm-rowwise-fp8-generation-only-codex-gpt56-high-online-docs-20260823/`。

- canonical task snapshot SHA-256：
  `f10de262a20cb083c95e9fc6dfae962e7538f90f38bd89286b910c2da3b596d7`；该字段是评分脚本
  对 task 目录快照的哈希，不冒充 Harbor task checksum；
- B200 hidden verifier：`zeros/identity/random/wide-range` 全部通过，correctness `1`；
- Candidate median `0.007614 ms`，Reference median `0.011814 ms`，raw speedup
  `1.5516613767x`，capped reward `1`；
- 评分证据：
  `jobs/candidate-verification-fp8-generation-only-HXJcPzJ-20260823-052311/`。

### 两次独立重复（同一 canonical snapshot）

两个全新 generation Trial 均正常退出并只提交 PTX；online-docs setting 下分别使用
官方文档搜索 `4` 次和 `6` 次：

| Trial | Agent execution | Candidate SHA | 实现特征 | raw speedup | correctness |
|---|---:|---|---|---:|---:|
| `rSudjwX` | `619.36 s` | `f86e83f2…` | scalar BF16 FMA；36 regs；无 shared/barrier | `1.547658x` | `1` |
| `CabD5MG` | `455.40 s` | `89a69775…` | HMMA；46 regs；35,616 B shared | `1.679253x` | `1` |

- 生成 Job：
  `jobs/ptx-gemm-rowwise-fp8-generation-only-repeat2-codex-gpt56-high-online-docs-20260823/`；
- 两次评分均使用 task snapshot SHA-256
  `f10de262a20cb083c95e9fc6dfae962e7538f90f38bd89286b910c2da3b596d7`；
- 评分证据：
  `jobs/candidate-verification-fp8-repeat-rSudjwX-20260823-070401/`、
  `jobs/candidate-verification-fp8-repeat-CabD5MG-20260823-070530/`。

合并首个 pilot 后，Codex/high correctness 为 `3/3`，raw speedup 范围
`1.547658–1.679253x`、均值 `1.592857x`。HMMA 路线略快，但当前 Reference 仍不足以
形成高性能门槛。

这些结果实际验证了 HMMA fragment 映射、row scale 和 finite E4M3 bytes，但当前性能
Reference 仍偏弱。待完成：

1. 加强同 ABI Reference；
2. 用开放权重模型做独立校准。
