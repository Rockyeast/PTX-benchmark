# `ptx-paged-decode-attention-bf16-hq32-hkv8-d128-p16` 任务规格

**状态：Agent-calibrated draft v0.1（2026-08-22）。这是第一批 pilot 中当前结构最难的
单-entry Direct-PTX 候选；B200 Oracle、负例和 5 次 Codex/high 初步校准已完成，
尚未完成开放权重模型和 release 验证。**

## 1. 固定 specialization

```text
single-request paged GQA decode attention
BF16 query / paged K / paged V / output
HQ=32 query heads
HKV=8 KV heads
GQA group size=4
D=128
page size=16 tokens
maximum logical KV length=4096
320 physical cache pages
NHD-within-page contiguous layout
NVIDIA B200 / sm_100
```

这比固定 dense MHA 同时增加：logical-to-physical page table、非连续物理页、最后一页、
运行时 KV 长度和 `q_head -> kv_head` GQA 映射。第一版不加入 batch、RoPE、ALiBi、
sliding window、FP8、LSE、workspace、split-KV 或多 kernel merge；这些会改变 ABI 或
拓扑，应另拆 task。

## 2. FlashInfer 依据

固定官方仓库 revision：

```text
repository: https://github.com/flashinfer-ai/flashinfer
commit:     fb28d7242b3506a2348265962041acc1fb56cca4
license:    Apache-2.0
```

主要依据：

- `flashinfer/trace/templates/attention.py` 的 `_gqa_paged_decode_reference`：先按 page ID
  gather，再 flatten page/token，`kv_head = q_head // gqa_ratio`，FP32 QK/Softmax/PV；
- `tests/attention/test_batch_decode_kernels.py`：paged NHD、page size 16、GQA、D=128、
  跨页 KV 长度和独立 contiguous reference；
- `include/flashinfer/attention/decode.cuh`：paged offset、GQA dispatch、长 KV partition、
  online Softmax 和输出路径；
- `csrc/batch_decode.cu`：page indices/indptr/last-page metadata、layout stride、plan/run
  dispatch。

本题把 batch/indptr/last-page metadata 收缩为单请求的 `page_table + kv_len`，从而保留
分页和 GQA 核心能力，同时保持一个普通 launch 的 raw-PTX entry。FlashInfer 实现不是
本题可直接提交的 PTX 答案。

## 3. Agent ABI

Agent 创建 `/app/kernel.ptx`，包含 visible entry：

```text
paged_decode_fwd_bf16_hq32_hkv8_d128_p16(
    const bf16* q,
    const bf16* k_cache,
    const bf16* v_cache,
    const uint32_t* page_table,
    bf16* output,
    uint32_t kv_len)
```

五个 pointer 为 64-bit，`kv_len` 为 u32，且 `0 <= kv_len <= 4096`。

固定 tensor：

```text
q          : [32, 128], bf16
k_cache    : [320, 16, 8, 128], bf16
v_cache    : [320, 16, 8, 128], bf16
page_table : [256], uint32
output     : [32, 128], bf16
```

对 logical token `t`：

```text
logical_page  = t / 16
page_offset   = t % 16
physical_page = page_table[logical_page]
```

对 KV head `hk`、feature `d`，cache 线性索引：

```text
((((physical_page * 16) + page_offset) * 8 + hk) * 128 + d)
```

Hidden cases 使用乱序且唯一的物理 page ID，范围 `[0,320)`；未使用的 page table entry
含非法 sentinel，最后一页中 `t >= kv_len` 的 slot 含 poison，因此实现不能假设物理页
连续，也不能把 KV 长度向上取整后读取 padding。

## 4. 数学语义

对 `query_head hq`：

```text
hk = hq / 4
scale = 1 / sqrt(128)

score[t] = scale * sum_d(float(q[hq,d]) * float(K[t,hk,d]))
m        = max_t(score[t])
p[t]     = exp(score[t]-m) / sum_j(exp(score[j]-m))
acc[d]   = sum_t(p[t] * float(V[t,hk,d]))
output[hq,d] = bf16(acc[d])
```

QK、Softmax、PV 使用至少 FP32。`kv_len=0` 时必须把全部 output 写成 BF16 `+0`，且不
读取 page table 或 cache。输入有限；近似指数只要满足最终 BF16 容差即可。

## 5. 固定 launch

```text
grid:                    (32,1,1)
block:                   (128,1,1)
dynamic shared memory:   0
programmatic dependency: disabled
```

每个 block 对应一个 query head，`blockIdx.x` 即 `hq`，`threadIdx.x` 可自然映射到
feature。允许静态 shared memory；保存 4096 个 FP32 score 需要 16 KiB。

## 6. Correctness

CPU truth 使用 double/FP32 稳定公式，独立 gather page table，并执行 BF16
round-to-nearest-even。Candidate 与固定 Reference 分别比较 CPU truth：

```text
abs(got-want) <= 2e-2 + 2e-2*abs(want)
```

初始 cases：

| kv_len | 目的 |
|---:|---|
| 0 | 零输出，不得读取 metadata/cache |
| 1 | 单 token 与 32->8 GQA 映射 |
| 15 | page 内边界 |
| 16 | 完整一页 |
| 17 | 跨页与最后一页 1 token |
| 97 | 乱序物理页、非整页 |
| 512 | 多页随机 logits、head 隔离 |
| 4095 | 最大长度前一位、padding poison |
| 4096 | 满上下文与数值稳定性 |

检查 Q/K/V/page table 不变、output 全写、allocation guards、合法 PTX/module/entry、
launch/sync 和 CUDA 错误。错误 PTX correctness 为 0，不进入性能评分。

## 7. Performance 与 reward

性能规模：

```text
kv_len = 128, 1024, 4096
```

固定同 ABI Reference 与 Candidate 在同 GPU/context、相同输入下交替计时：

```text
r_L = reference_median_time_L / candidate_median_time_L
speedup = geometric_mean(r_128, r_1024, r_4096)
performance = clamp(speedup, 0, 1)
reward = correctness * performance
```

逻辑 FLOPs（32 query heads，每个 token 同时参与 QK 与 PV）：

```text
logical_flops(L) = 4 * 32 * L * 128
```

第一版 Reference 若被真实 Agent 普遍大幅超过，只能作为 feasibility 基线，不能冻结
leaderboard。

## 8. 验证门槛

- [x] 固定上游 revision、paged/GQA 语义和许可证；
- [x] 固定 entry、layout、shape、page table 和 launch ABI；
- [x] 生成 fail-closed scaffold；
- [x] 实现 Agent instruction、CPU truth、Reference/Oracle 和 verifier；
- [x] 添加 contiguous-page 错误、错误 GQA、忽略 last-page 和慢正确 fixtures；
- [x] B200 Oracle、负例、重复性能和 Harbor Job；
- [x] 固定配置真实 Agent feasibility（1 次；verifier pass，但 Agent timeout）；
- [x] 固定配置真实 Agent 重复校准（总计 5/5 verifier pass，4/5 正常退出）；
- [ ] 开放权重模型与严格网络隔离校准。

未完成以上验证前保持 `release_ready=false`。
