# PTX-Benchmark task documentation

本目录保存每道 task 的详细档案。数据集全局状态见
[`../dataset/task-catalog.md`](../dataset/task-catalog.md)，跨任务实验汇总见
[`../experiments/summary.md`](../experiments/summary.md)。

- `*-spec.md`：operation、dtype/layout、shape、ABI、launch 和评分设计。
- `*-validation.md`：Oracle、负例、Agent、性能与环境验证证据。
- `tasks/<task>/` 只保留 Harbor 执行所需文件，不再维护 task-level README。

## Infrastructure and calibration

| Task | Spec | Validation |
|---|---|---|
| `harbor-smoke` | — | [`harbor-task-lifecycle.md`](../harbor-task-lifecycle.md) |
| `ptx-add-one` | — | [`validation`](ptx-add-one-validation.md) |
| `ptx-add-one-l40` | — | [`validation`](ptx-add-one-l40-validation.md) |
| `ptx-vector-add` | [`spec`](ptx-vector-add-spec.md) | [`validation`](ptx-vector-add-validation.md) |
| `ptx-vector-add-l40` | — | [`validation`](ptx-vector-add-l40-validation.md) |
| `ptx-silu-gate-forward-bf16-n4096-l40` | — | [`validation`](ptx-silu-gate-forward-bf16-n4096-l40-validation.md) |

## Formal candidates and upper probes

| Task | Spec | Validation |
|---|---|---|
| `ptx-rmsnorm-forward-bf16-n4096` | [`spec`](ptx-rmsnorm-forward-bf16-n4096-spec.md) | [`validation`](ptx-rmsnorm-forward-bf16-n4096-validation.md) |
| `ptx-rope-forward-bf16-d128` | [`spec`](ptx-rope-forward-bf16-d128-spec.md) | [`validation`](ptx-rope-forward-bf16-d128-validation.md) |
| `ptx-silu-gate-forward-bf16-n4096` | [`spec`](ptx-silu-gate-forward-bf16-n4096-spec.md) | [`validation`](ptx-silu-gate-forward-bf16-n4096-validation.md) |
| `ptx-softmax-forward-fp32-n4096` | [`spec`](ptx-softmax-forward-fp32-n4096-spec.md) | [`validation`](ptx-softmax-forward-fp32-n4096-validation.md) |
| `ptx-flash-attention-forward-bf16-s128-d128` | [`spec`](ptx-flash-attention-forward-bf16-s128-d128-spec.md) | [`validation`](ptx-flash-attention-forward-bf16-s128-d128-validation.md) |
| `ptx-paged-decode-attention-bf16-hq32-hkv8-d128-p16` | [`spec`](ptx-paged-decode-attention-bf16-hq32-hkv8-d128-p16-spec.md) | [`validation`](ptx-paged-decode-attention-bf16-hq32-hkv8-d128-p16-validation.md) |
| `ptx-kda-recurrent-decode-bf16-t1-h16-hv32-d128-n32` | [`spec`](ptx-kda-recurrent-decode-bf16-t1-h16-hv32-d128-n32-spec.md) | [`validation`](ptx-kda-recurrent-decode-bf16-t1-h16-hv32-d128-n32-validation.md) |
| `ptx-kda-speculative-decode-bf16-t6-n8-h16-hv32-d128` | [`spec`](ptx-kda-speculative-decode-bf16-t6-n8-h16-hv32-d128-spec.md) | [`validation`](ptx-kda-speculative-decode-bf16-t6-n8-h16-hv32-d128-validation.md) |
| `ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128` | — | [`validation`](ptx-kda-dispatch-bf16-t1-t6-h16-hv32-d128-validation.md) |
| `ptx-flash-attention-backward-bf16-s128-d128` | — | [`validation`](ptx-flash-attention-backward-bf16-s128-d128-validation.md) |
| `ptx-block-sparse-attention-forward-bf16-s1024-d128-b16` | — | [`validation`](ptx-block-sparse-attention-forward-bf16-s1024-d128-b16-validation.md) |
| `ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16` | — | [`validation`](ptx-paged-prefill-attention-bf16-q128-hq8-hkv2-d128-p16-validation.md) |
| `ptx-gemm-rowwise-fp8e4m3-m128-n128-k128` | — | [`validation`](ptx-gemm-rowwise-fp8e4m3-m128-n128-k128-validation.md) |
| `ptx-kda-full-dispatch-bf16-d128` | — | [`validation`](ptx-kda-full-dispatch-bf16-d128-validation.md) |
| `ptx-paged-sparse-attention-backward-bf16-q128-hq8-hkv2-d128-p16` | [`spec`](ptx-paged-sparse-attention-backward-bf16-q128-hq8-hkv2-d128-p16-spec.md) | [`validation`](ptx-paged-sparse-attention-backward-bf16-q128-hq8-hkv2-d128-p16-validation.md) |
