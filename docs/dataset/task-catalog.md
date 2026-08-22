# PTX-Benchmark 题目目录

**状态：Draft v0.2（2026-08-21）。候选题不等于已经进入正式数据集。**

## 1. 状态说明

| 状态 | 含义 |
|---|---|
| infrastructure | 只验证 Harbor/GPU，不计入 benchmark |
| calibration | 校准 correctness/performance 管线，不计入最终总分 |
| selected | 已选入 pilot，但尚未写规格 |
| specified | 规格、ABI 和测试边界已有草案，但尚未在目标 GPU 跑通 |
| feasibility | ABI 和最小 Oracle 已在目标 GPU 跑通 |
| validated | 正负例、Reference 和重复测量完成 |
| calibrated | 多个真实 Agent/模型已运行 |
| frozen | 可进入 release manifest |

## 2. 已有任务

| Task | 家族/难度 | 当前状态 | 是否在 dev manifest | 是否可进 release |
|---|---|---|---:|---:|
| `harbor-smoke` | infrastructure | 已完成 | 否 | 否 |
| `ptx-add-one` | infrastructure | 已完成 | 否 | 否 |
| `ptx-vector-add` | C0 calibration | calibrated | 是 | 否 |
| `ptx-rmsnorm-forward-bf16-n4096` | A / D2 | feasibility | 是 | 暂不 |

RMSNorm 是第一道正式题候选，但仍缺 INT21 外部性能对照、负例和多 Agent 校准。

## 3. 第一批十道 pilot 候选

第一版先用 10 道题形成较稳定的正确率分母。当前临时难度目标是：在统一的
`GPT-5.6 Sol + Codex + reasoning_effort=high` 配置下，整组 task 的 correctness
通过率约为 30%～50%。这是选题假设，不是已经得到的实验结论；正式冻结仍要运行
Kimi K3、GLM 等开放权重模型。

| 顺序 | 暂定 Task | 专门化边界 | 家族/难度 | 正确性真值 | 性能来源候选 | 状态 |
|---:|---|---|---|---|---|---|
| P1 | `rmsnorm-forward-bf16-n4096` | plain forward，BF16，FP32 weight，N=4096 | A / D2 | 独立 CPU/PyTorch 公式 | INT21 RMSNorm-B200 | feasibility |
| P2 | `rope-forward-bf16-d128` | NeoX RoPE，BF16，head_dim=128，Q/K 融合 | C / D2 | 独立 CPU/Python FP32/BF16 公式 | FlashInfer RoPE | feasibility |
| P3 | `silu-gate-forward-bf16-n4096` | `silu(gate) * up`，BF16，N=4096 | B / D1 | 独立 CPU/PyTorch 公式 | FlashInfer fused activation | selected |
| P4 | `softmax-forward-bf16-n4096` | row-wise forward，BF16，N=4096 | A / D2 | 稳定 FP32 softmax 公式 | FlashInfer / CUDA Reference | selected |
| P5 | `kv-cache-append-bf16-d128` | paged K/V append，BF16，head_dim=128 | C / D2 | 独立索引与拷贝公式 | FlashInfer paged KV API | selected |
| P6 | `fp8-rowwise-quant-e4m3` | BF16 → FP8 E4M3 + row scale，shape 待冻结 | D / D2-D3 | 独立量化与反量化规则 | FlashInfer / CUTLASS | selected |
| P7 | `moe-top2-routing-bf16-e128` | 每 token 选 2 个 expert，E=128 | F / D3 | 独立 stable top-2 + weight 公式 | FlashInfer / serving routing | selected |
| P8 | `gemm-bf16-fixed` | BF16×BF16→FP32/BF16，固定 B200 shape | D / D3 | 独立 CPU/PyTorch matmul | NVIDIA CUTLASS Blackwell GEMM | selected |
| P9 | `flash-attention-forward-bf16-d128` | causal forward，BF16，head_dim=128 | E / D4 | 独立 FP32 attention 公式 | FlashInfer / FlashAttention | selected |
| P10 | `paged-decode-attention-bf16-d128` | GQA decode，paged KV，BF16，head_dim=128 | E / D4 | 独立 FP32 attention + page gather | FlashInfer / CUTLASS GQA | selected |

家族数量为 `A:2, B:1, C:2, D:2, E:2, F:1`。P3 提供相对容易但仍有融合和特殊函数
意义的锚点；P8～P10 则故意保留 Tensor Core、online softmax 和 paged metadata 等
高难能力。预期分布必须由真实 Agent 重复运行验证，不能只凭 D1～D4 标签认定。

## 4. 第二批候选池

这些题只进入候选池，尚未选入 pilot：

| 候选 | 家族 | 主要价值 | 当前阻塞/注意点 |
|---|---|---|---|
| LayerNorm forward/backward | A | reduction、均值与方差 | 与 RMSNorm 重复度要控制 |
| FP8 / NVFP4 GEMM | D | Blackwell 低精度 Tensor Core | scale 格式、TMA descriptor 和 Reference 较复杂 |
| KDA recurrent decode | E | 现代 state-space/attention hybrid | 官方 B200 实现有 host dispatch 和七条 kernel 路径，不适合直接塞进当前单-entry格式 |
| grouped GEMM / fused MoE | F | expert routing 与不规则工作量 | 需要 workspace 和多输出契约 |
| Top-K / Top-P sampling | F | 非规则选择与概率语义 | 随机性、确定性和统计正确性需单独设计 |

## 5. 上游覆盖依据

- INT21 目前公开的第一组实现覆盖 RMSNorm 与 Kimi Delta Attention，并分别有 H100、
  B200 版本；这提供两类 workload，不是一套可直接转成任意 PTX task 的答案库。
- FlashInfer 官方覆盖 attention、BF16/FP8/FP4 GEMM、MoE、sampling、RoPE、normalization
  和 fused activation，适合作为候选题来源与外部实现对照。
- FlashInfer-Bench 提供真实部署 kernel/workload 数据集，可用于后续选择 shape；它的
  proposal/workflow 系统与 Harbor task 不是同一层，不直接替换当前 verifier。
- CUTLASS 官方覆盖 Blackwell BF16、FP8、NVFP4/MXFP 等 GEMM，可作为线性代数题的
  NVIDIA 性能基线；使用 architecture-accelerated 指令时需要明确 `sm_100a` 边界。

## 6. 下一步

P2 RoPE 已完成 B200 Oracle feasibility 和两个安全负例检查。下一步为 P3 SiLU-gate
做来源审阅和规格草案，暂不同时 scaffold P4～P10；同时保留 P1 RMSNorm、P2 RoPE 的
剩余负例、重复计时和 Agent 校准。等前三道正式候选暴露出共同的 ABI、Reference 和
verifier 问题后，再冻结模板并继续扩展。
