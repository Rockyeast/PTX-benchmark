# PTX-Benchmark 题目目录

**状态：Draft v0.1（2026-08-20）。候选题不等于已经进入正式数据集。**

## 1. 状态说明

| 状态 | 含义 |
|---|---|
| infrastructure | 只验证 Harbor/GPU，不计入 benchmark |
| calibration | 校准 correctness/performance 管线，不计入最终总分 |
| selected | 已选入 pilot，但尚未写规格 |
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

## 3. 第一批五道 pilot

| 顺序 | 暂定 Task | 专门化边界 | 家族/难度 | 正确性真值 | 性能来源候选 | 状态 |
|---:|---|---|---|---|---|---|
| P1 | `rmsnorm-forward-bf16-n4096` | plain forward，BF16，FP32 weight，N=4096 | A / D2 | 独立 PyTorch/CPU 公式 | INT21 RMSNorm-B200 + 同 ABI Reference | feasibility |
| P2 | `rope-forward-bf16-d128` | LLaMA-style RoPE，BF16，head_dim=128 | C / D2 | PyTorch sin/cos 公式 | FlashInfer RoPE | selected |
| P3 | `fp8-rowwise-quant-e4m3` | BF16 → FP8 E4M3 + row scale，shape 待冻结 | D / D2-D3 | CPU/PyTorch quant 规则 | FlashInfer/CUTLASS 低精度实现 | selected |
| P4 | `gemm-bf16-fixed` | BF16×BF16→FP32/BF16，固定 B200 shape family | D / D3 | CPU/PyTorch matmul | NVIDIA CUTLASS BF16 GEMM | selected |
| P5 | `flash-attention-forward-bf16-d128` | causal forward，BF16，head_dim=128，shape 待冻结 | E / D4 | PyTorch SDPA 数学参考 | FlashInfer / FlashAttention | selected |

这个顺序故意逐步增加风险：先做索引/向量化，再做低精度，再进入 Tensor Core 和
attention。P2 是下一道最合适的 task；它能验证“从现有 task 模板复制新题”的能力，
但不会立即陷入 GEMM/Attention 的复杂 host ABI。

## 4. 第二批候选池

这些题只进入候选池，尚未选入 pilot：

| 候选 | 家族 | 主要价值 | 当前阻塞/注意点 |
|---|---|---|---|
| LayerNorm forward/backward | A | reduction、均值与方差 | 与 RMSNorm 重复度要控制 |
| fused SiLU-gate / GELU | B | 融合与特殊函数 | 可能过易，需看模型分布 |
| KV-cache append / layout transform | C | 真实 serving 数据移动 | layout 与 paged metadata 需冻结 |
| FP8 / NVFP4 GEMM | D | Blackwell 低精度 Tensor Core | scale 格式、TMA descriptor 和 Reference 较复杂 |
| paged decode attention | E | 真实 serving 核心路径 | 单 entry ABI 很大，先在 P5 后评估 |
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

只推进 P2 RoPE 的“来源审阅 + 规格文档”，暂不 scaffold。RMSNorm 保持 feasibility
状态，等五道 pilot 的共同结构清楚后，再统一补齐负例和真实 Agent 校准。

