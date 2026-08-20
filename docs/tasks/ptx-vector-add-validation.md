# `ptx-vector-add` B200 验证记录

日期：2026-08-20

## 环境

- 主机：`catalyst-b200`
- GPU：NVIDIA B200；每次由 Harbor Singularity GPU allocator 独占一张卡
- CUDA：13.2，`nvcc` 13.2.78
- SIF：`ptx-bench-cuda13.2.1-ubuntu24.04-v1.sif`
- SIF SHA-256：`841897a25dd7b05cf387c3b7a580c6bfcf20ae15b3a8858303d49139b15f9ebd`
- 远端 Harbor：0.13.2，个人隔离副本
  `/home/zhuominc/rockyeast-work/harbor-ptx`

## Direct-PTX 迁移验证

2026-08-20 又完成了一组受限环境验证。新 SIF
`ptx-bench-direct-cuda13.2.1-codex0.148-ubuntu24.04-v1.sif` 移除了 `nvcc`、
NVRTC、`cudafe++` 和 NVVM，保留 `ptxas`；Verifier 改为直接加载出题阶段
预编译且固定 digest 的 `tests/reference.ptx`。

| Trial | Agent | correctness | reward | 结果 |
|---|---|---:|---:|---|
| `ptx-vector-add-direct-oracle-final-image` | Oracle | 1 | 0.991443 | 通过 |
| `ptx-rmsnorm-direct-oracle-final-image` | Oracle | 1 | 1.000000 | 通过 |
| `ptx-vector-add-direct-codex-final-image` | `gpt-5.6-sol` | 1 | 1.000000 | 通过；raw speedup 1.007243 |

真实 Agent trajectory 显示：它直接创建 `/app/kernel.ptx`，随后只调用
`ptxas` 和 `cuobjdump` 检查，没有调用或下载 CUDA/Triton 等高层编译器。
该结果验证了本次运行的 direct-PTX 路径；由于模型 API 仍需要网络，正式发布
仍应保留 trajectory 审计，并在基础设施允许时增加 egress allowlist。

## Oracle 重复性

| Trial | 物理 GPU | Task checksum | correctness | reward |
|---|---:|---|---:|---:|
| `ptx-vector-add-oracle-v1` | 0 | `d2c5c1b0...0c408cae` | 1 | 0.990683 |
| `ptx-vector-add-oracle-v2` | 0 | `7a54bd86...e365ad6` | 1 | 0.997697 |
| `ptx-vector-add-oracle-v3` | 4 | `7a54bd86...e365ad6` | 1 | 0.987482 |

三次都通过 20 个 correctness case。reward 范围为 `0.987482–0.997697`；这个
reward 是同一次 trial 内 Reference/Candidate 中位时延比的几何平均，不是裸时延。
v1 与 v2/v3 的 checksum 差异来自新增的任务作者验证 fixtures，核心 Oracle、
verifier 和性能 reference 未改变。

## 负例与慢实现

| Trial | 预期故障/性质 | 实际结果 |
|---|---|---|
| `ptx-vector-add-negative-missing-entry-v1` | 无 `vector_add` | `CUDA_ERROR_NOT_FOUND`，reward 0 |
| `ptx-vector-add-negative-overrun-v1` | 写入 `output[n]` | `output suffix guard changed for n=0`，reward 0 |
| `ptx-vector-add-slow-v1` | 正确但每个 block 多执行同步循环 | correctness 1，reward 0.267804 |

慢实现三个性能规模的 Candidate/Reference 中位时延比分别约为
`0.3850 / 0.2271 / 0.2197`，几何平均为 `0.267804`。这证明当前 reward 能区分
“错误”“正确但慢”和“正确且接近 Reference”三种结果。

## 最终 Oracle 与真实 Agent

最终文档版之前的同一执行逻辑使用 task checksum
`6b683aa38a83d9b3e9463af61a9f68f2ec8104789dd6590b2102451ff98f5e70`：

| Trial | Agent | 物理 GPU | correctness | raw speedup | reward |
|---|---|---:|---:|---:|---:|
| `ptx-vector-add-oracle-final-v1` | Oracle | 0 | 1 | 0.991764 | 0.991764 |
| `ptx-vector-add-codex-v1` | Codex 0.148.0 / `gpt-5.6-sol` | 0 | 1 | 1.000609 | 1.000000 |

Codex Agent 执行约 60.35 秒，完整 trial 约 79.89 秒。轨迹显示它直接创建
1003-byte `/app/kernel.ptx`，使用 PTX 8.7 和 `sm_100`，随后调用 `ptxas` 和
`cuobjdump` 做语法与资源检查；它没有调用 `nvcc` 生成提交。最终 kernel 使用
12 个寄存器、无 shared/local memory。

三个性能规模上，Codex Candidate/Reference 的中位时延分别为：

| elements | Candidate ms | Reference ms | ratio |
|---:|---:|---:|---:|
| 1,048,576 | 0.006569 | 0.006576 | 1.001062 |
| 16,777,216 | 0.054898 | 0.054898 | 1.000001 |
| 67,108,864 | 0.210337 | 0.210497 | 1.000764 |

这个结果证明 verifier 与真实 Agent 链路有效，但不能证明 vector add 有足够模型
区分度。该题保留为基础设施和评分校准题，不直接代表最终 PTX-Bench 难度。

完成说明性 README 更新并移走远端遗留的 scaffold `test_outputs.py` 后，本地与
远端 13 个 task 源文件的逐文件 SHA-256 完全一致。最终发布版 task checksum 为
`db9fb7560698ef725caf522c76495cd46418683e7a3ba2676302e215e1b2f2a2`。
对应 `ptx-vector-add-oracle-release-v2` 在物理 GPU 0 上再次通过：correctness
`1.0`、raw speedup/reward `0.990869`。这些整理没有改变任何执行逻辑。

## 证据位置

远端 trial 均保存在：

```text
/home/zhuominc/rockyeast-work/PTX-benchmark/trials/<trial-name>/
```

每个目录中的 `result.json` 保存 Harbor 最终多指标 reward，
`verifier/metrics.json` 保存全部 correctness sizes、`nvcc` 版本、每个规模的五个
原始 timing batch、中位数、离散度和同卡 speedup。
