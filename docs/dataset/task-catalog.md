# PTX-Benchmark 题目目录

**状态：Draft v0.6（2026-08-28）。当前均为开发任务，尚无 frozen release task。**

- 每题规格与验证档案：[`../tasks/README.md`](../tasks/README.md)
- 跨任务实验汇总：[`../experiments/summary.md`](../experiments/summary.md)
- 开发数据集 manifest：[`../../tasks/dataset.toml`](../../tasks/dataset.toml)

本文件只维护全局题目索引、状态和候选池，不记录单次实验过程。

## 状态

| 状态 | 含义 |
|---|---|
| `infrastructure` | 只验证 Harbor、Agent 或 GPU 链路 |
| `calibration` | 校准 correctness/performance 管线，不计正式总分 |
| `selected` | 已选入候选，但尚未完成规格 |
| `specified` | operation、ABI 和测试边界已有草案 |
| `feasibility` | 最小作者标准提交已在目标 GPU 跑通 |
| `validated` | 正负例、真值和评分流程完成 |
| `calibrated` | 已运行真实 Agent；必须注明模型与配置 |
| `frozen` | task digest、环境、评分和发布证据均冻结 |

## Infrastructure and calibration

| Task | 类型 | 状态 | Release |
|---|---|---|---:|
| `harbor-smoke` | Harbor smoke | `infrastructure` | 否 |
| `ptx-add-one` | Direct-PTX correctness smoke | `infrastructure` | 否 |
| `ptx-add-one-l40` | L40/Qwen smoke | `calibration` | 否 |
| `ptx-vector-add` | 基础性能链路 | `calibration` | 否 |
| `ptx-vector-add-l40` | L40 性能链路 | `calibration` | 否 |
| `ptx-silu-gate-forward-bf16-n4096-l40` | L40 硬件迁移 | `calibration` | 否 |

## Pilot tasks

| ID | Task | 家族/结构 | 当前状态 | 关键结论 |
|---:|---|---|---|---|
| P1 | `ptx-rmsnorm-forward-bf16-n4096` | A / single-entry | `feasibility` | 缺 INT21/SOTA 性能锚点和 Agent 校准 |
| P2 | `ptx-rope-forward-bf16-d128` | C / single-entry | `feasibility` | Oracle 与安全负例已验证 |
| P3 | `ptx-silu-gate-forward-bf16-n4096` | B / single-entry | `calibrated` | Codex 5/5，容易锚点 |
| P4 | `ptx-softmax-forward-fp32-n4096` | A / single-entry | `calibrated` | Codex 5/5；性能 Reference 待替换 |
| P5 | `ptx-kv-cache-append-bf16-d128` | C / selected | `selected` | 尚未 scaffold |
| P6 | `ptx-fp8-rowwise-quant-e4m3` | D / selected | `selected` | shape 尚未冻结 |
| P7 | `ptx-moe-top2-routing-bf16-e128` | F / selected | `selected` | correctness 与稳定排序待规格化 |
| P8 | `ptx-gemm-bf16-fixed` | D / selected | `selected` | SOTA、Baseline 和 shape 待冻结 |
| P9 | `ptx-flash-attention-forward-bf16-s128-d128` | E / single-entry | `calibrated` | Codex 5/5；Reference 偏弱 |
| P10 | `ptx-paged-decode-attention-bf16-hq32-hkv8-d128-p16` | E / single-entry | `calibrated` | Codex correctness 5/5；Reference 偏弱 |
| P11 | `ptx-kda-recurrent-decode-bf16-t1-h16-hv32-d128-n32` | E / single-entry | `calibrated` | Codex 3/3，偏容易 |
| P12 | `ptx-kda-speculative-decode-bf16-t6-n8-h16-hv32-d128` | E / single-entry | `calibrated` | Codex 3/3；性能存在差异 |
| P13 | `ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128` | E / multi-entry | `calibrated` | Codex 3/3；联合性能 reward 有差异 |
| P14 | `ptx-flash-attention-backward-bf16-s128-d128` | E / single-entry | `calibrated` | Codex 2/2；Reference 明显偏弱 |
| P15 | `ptx-block-sparse-attention-forward-bf16-s1024-d128-b16` | E / single-entry | `calibrated` | Codex 3/3；Reference 偏弱 |
| P16 | `ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16` | E / single-entry | `calibrated` | Codex 3/3；Reference 偏弱 |
| P17 | `ptx-gemm-rowwise-fp8e4m3-m128-n128-k128` | D / single-entry | `calibrated` | Codex 3/3；Reference 偏弱 |
| P18 | `ptx-kda-full-dispatch-bf16-d128` | E / multi-entry | `calibrated` | Codex 3/5；当前 correctness upper probe |
| P19 | `ptx-paged-sparse-attention-backward-bf16-q128-hq8-hkv2-d128-p16` | E / multi-entry | `validated` | correctness-only；Agent 尚未校准 |

当前任务中，只有 P18 对 GPT-5.6/Codex/high 形成明显 correctness 区分；该结果不能替代开放权重模型校准。多数性能题仍使用 task-owned 弱 Reference，不能冻结正式性能排名。

## Candidate pool

| 候选 | 家族 | 主要价值 | 主要前置条件 |
|---|---|---|---|
| LayerNorm forward/backward | A | reduction、均值与方差 | 控制与 RMSNorm 的重复度 |
| FP8/NVFP4 GEMM | D | Blackwell Tensor Core | 固定 scale 格式、Baseline 和开源 SOTA |
| General multi-shape GEMM | D | 隐藏多 shape 通用性 | 固定统一 entry/ABI 和 SOTA shape regime |
| Grouped GEMM/fused MoE | F | expert routing、不规则 workload | 固定 workspace、输出和真实 Baseline |
| Top-K/Top-P sampling | F | 非规则选择与概率语义 | 设计确定性与统计正确性 |
| General multi-shape attention | E | 防止单 shape 过拟合 | 复用开源 SOTA 的接口和 shape regime |

## Selection rule

正式 performance task 应优先从真实开源 SOTA 反推 operation、dtype/layout、shape regime 和 ABI，并同时具备：

1. 独立 correctness truth；
2. 现实、可复现的性能 Baseline；
3. 固定 revision、许可证和构建记录的 SOTA 性能满分线；
4. 隐藏多 shape、负例和目标模型校准。

没有合格 SOTA 的任务只能保留为 correctness、infrastructure 或 feasibility task。
