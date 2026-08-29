# `ptx-flash-attention-backward-bf16-s128-d128` 验证记录

**状态：B200 Oracle、2/2 定向负例与 Codex/high pilot 均已通过；当前 Reference
明显偏慢。**

## 任务边界

- 固定 `S=128, D=128`，单 head、非 causal；
- BF16 `Q/K/V/dO`，FP32 `dQ/dK/dV`；
- 一条 entry、一个 128-thread block、dynamic shared memory 为 0；
- kernel 必须在一次 launch 内写完整三份梯度，不能依赖预先清零；
- correctness 先由独立 C truth 硬门控，之后才比较同 ABI Reference 性能。

## 作者侧 provenance

```text
CUDA toolkit: CUDA 13.2.1 / nvcc 13.2.78
command:      nvcc -ptx -O3 --use_fast_math -arch=compute_100 -code=compute_100
reference.cu:  5b73666c7cd617cb45d46b0e89321dcae75c5dc6b88d7f4802da9c2630ffb14e
reference.ptx: d32e26dfbaf6e4277c0f134d9f9a1a4d51207655bc537d4c686cf6a0f320c3ad
CPU truth:     82e978ea757d9471f492b1c74063f60319d3695d33819cb68e8c28c1c55a531f
```

## B200 Harbor Oracle（2026-08-23）

- task checksum：`f36175d5d4fb18dd41d6229dd1e0936b2700a612f45dac912cf645384120804d`；
- Agent：Harbor Oracle；正常结束，无异常；总运行时间约 23 秒；
- GPU：`NVIDIA B200`；
- correctness：`1`，`zeros/random/structured/dominant-key` 四个 case 全部通过；
- Candidate median：`4.668785 ms`；Reference median：`4.667241 ms`；
- raw speedup / reward：`0.9996691633`；
- Job：
  `jobs/ptx-flash-attention-backward-bf16-s128-d128-b200-oracle-20260823-025754/`。

该 Job 使用修正后的 `dominant-key` 生成器：所有 query 明确与固定 key row 7 对齐，
不是旧版没有实际形成主导 key 的弱 case。

## 定向负例（2026-08-23）

固定 CUDA 13.2.1 v2 SIF 已把 Reference 汇编为 `sm_100a` cubin：188 registers、
1 barrier、每线程 1024-byte stack，无额外 spill store/load。

- `only-dq.ptx`：实际进入 hidden verifier，因 `grad_k` 未写而拒绝；
- `depends-on-prezero.ptx`：实际进入 zeros case，因依赖输出初值产生 NaN 而拒绝；
- 两项 reward/correctness 均为 `0`；
- 证据：
  `jobs/direct-ptx-negative-fixtures-20260823-032914/ptx-flash-attention-backward-bf16-s128-d128/`。

## Codex/high/online-docs pilot（2026-08-23）

本次同样采用“无 GPU 生成 + canonical B200 hidden verifier”两阶段诊断，原因是共享
B200 当时均被占用。临时副本只把 `required_gpus` 改为 `0`、移除硬件 healthcheck
并禁用 verifier；instruction、Agent budget、Direct-PTX 工具和 online-docs setting
保持不变。

生成阶段：

- Agent：Codex 0.148 / GPT-5.6 / high / `web_search=live`；
- 正常退出；Agent execution 约 `1019.4 s`；
- 生成侧诊断 checksum：`9f2d1a1b4e10d142965ccab457a898a94adbbf7917f0b18f7bdd90a9c268b9d4`，
  不与 canonical checksum 混用；
- trajectory 为直接 PTX；没有 CUDA/Triton 编译；实际调用 NVIDIA 官方在线 PTX
  文档搜索 `2` 次；
- 最终 candidate SHA-256：
  `f796b6077c3afae73f5a74a445c0866009602740cae3693b3f0d3148d8a0e6ae`；
- `ptxas` 通过、无 stack/spill，`96` registers、`1` barrier、32 KiB static shared；
- Job：
  `jobs/ptx-flash-attention-backward-generation-only-codex-gpt56-high-online-docs-20260823/`。

canonical B200 评分阶段：

- canonical checksum 仍为
  `f36175d5d4fb18dd41d6229dd1e0936b2700a612f45dac912cf645384120804d`；
- physical GPU `5`；评分进程正常结束；
- `zeros/random/structured/dominant-key` 四个 hidden case 全过，correctness `1`；
- Candidate median `1.812026 ms`；Reference median `4.666638 ms`；
- raw speedup `2.5753704286x`；capped reward `1`；
- 评分证据：
  `jobs/candidate-verification-flash-bwd-generation-only-HBi4g6L-20260823-042942/`。

该结果说明本题 correctness 对一次 GPT-5.6/high pilot 可解，但生成过程用了约 17 分钟，
且经历 shared-memory 上限、PTX WMMA 语法和 BF16 中间量容差迭代。它同时暴露固定
Reference 明显偏弱：reward 被截成 `1` 后掩盖了 `2.575x` 的实际差异，因此当前性能
分数不能冻结。

### 第二份独立生成候选与 B200 评分

- Trial `8Dj8RLV` 正常退出，Agent execution 约 `604.5 s`；
- generation-only checksum 与首轮相同，为 `9f2d1a1b…`；
- candidate SHA-256：
  `90071bb7a94bb72bf524a7d94810be4b7e4e2c2eec2dec48b31657052c44aecb`；
- trajectory 仍是直接 PTX，调用 NVIDIA 官方 PTX 文档搜索 `2` 次；
- `ptxas` 对 `sm_100/sm_100a` 均成功：38 registers、33280-byte static shared、
  1 barrier、无 stack/spill；候选使用 BF16 `mma.sync`；
- Job：
  `jobs/ptx-flash-attention-backward-generation-only-repeat1-codex-gpt56-high-online-docs-20260823/`。
- canonical 评分使用与首轮相同的 task snapshot `66b5870b…`，physical GPU `3`；
- `zeros/random/structured/dominant-key` 全过，correctness `1`；
- Candidate median `0.358630 ms`，Reference median `4.681722 ms`；raw speedup
  `13.054470x`，capped reward `1`；
- 评分证据：
  `jobs/candidate-verification-flash-bwd-repeat-8Dj8RLV-20260823-080041/`。

连同首轮，本题 Codex/high correctness 为 `2/2`，raw speedup 范围
`2.575370–13.054470x`。第二份候选的 PTX-level MMA 显著优于 CUDA 编译 Reference，
进一步确认性能 baseline 必须加强。

## 待完成

1. 加强同 ABI Reference，再重跑性能对照；
2. 补一次 canonical Harbor Agent+Verifier 单 Trial；
3. 用开放权重模型与多次独立 attempt 校准通过率和时间预算。
