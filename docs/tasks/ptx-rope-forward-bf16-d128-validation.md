# `ptx-rope-forward-bf16-d128` 验证记录

**状态：B200 feasibility 通过；尚未 verifier-validated 或 agent-calibrated。**

## 当前固定内容

```text
Harbor manifest digest:
sha256:937b3c37ca4b60df262a869fbd1dfa0b92a4f9ab306d3c4f4d6b83aa596861ea

oracle.ptx / reference.ptx SHA-256:
95c08244bb4358a0ef2502717610903de9df6c27464105e616f2968ecddb0a02

verify_and_benchmark.py SHA-256:
cb11760b6c3e22ee9beadd053092170f6800bc75ca5c0d09a2286a20041c9baa

verify_hardware.py SHA-256:
af99e3c2d8eb9d7bfc10300c8c258bd2d4e978995c266b58d677c91a3a426429

test_verifier_contract.py SHA-256:
66d6abaaf3da5eb16a132c5b63f65a5fcef7d634e33e0a1978581bbcc061a761

hidden direct-ptx-policy-check SHA-256:
2140384f3457e66cb36040d441c2ecebb2bca9aab79842dbd77c781d45ff1f4a

hidden PTX documentation manifest SHA-256:
1b306871a9d5674cb5fccb003c7b54f3dd2ed25341a38f642fe5fe513f0ca28e
```

Oracle 和性能 Reference 当前是同一份手写 PTX，所以二者速度比应接近 `1`。这只验证
计时和 reward 管线是否自洽，不证明 Reference 已达到高性能。

## Harbor Oracle：通过

2026-08-21 在 `catalyst-b200` 的 GPU 4 上运行：

```text
job:   jobs/ptx-rope-b200-oracle-error-contract-r7-20260821/
trial: ptx-rope-harbor-oracle-20260821__NvjZJTZ
GPU:   NVIDIA B200
```

结果：

| 指标 | 数值 |
|---|---:|
| trials / exceptions | 1 / 0 |
| correctness | 1.0 |
| speedup_vs_reference | 0.9962823588316692 |
| performance | 0.9962823588316692 |
| reward | 0.9962823588316692 |
| 总 Job 时间 | 14 秒 |

正确性覆盖 `num_tokens = 0, 1, 2, 3, 7, 31, 257`。性能覆盖
`num_tokens = 256, 4096, 32768`。硬件检查记录 `NVIDIA B200`、compute capability
`10.0`、`sm_100`；环境 policy checker 也通过，GPU 由 Harbor allocator 锁定并在
结束后释放。

GPU 4 上 `num_tokens=256` 的 Candidate dispersion 约为 `2.2%`，仍高于两个较大规模；
本次结果只能作为 feasibility 证据，不能冻结性能分数。此前 GPU 4/GPU 5 的几次开发
运行也都得到接近 `1` 的速度比，但不能替代固定 digest 下的正式重复测量。

## 隐藏 policy checker：通过

Harbor 0.21.0 源码确认单步 trial 的顺序是 Agent 结束、收集 artifact，然后才由
verifier 把 task 的 `tests/` 上传到容器 `/tests` 并执行 `/tests/test.sh`。当前 verifier
因此调用 `/tests/direct-ptx-policy-check` 和 `/tests/ptx-docs-manifest.toml`，不再信任
Agent 可修改的 `/usr/local/bin/direct-ptx-policy-check`。

2026-08-21 在现有 Direct-PTX SIF 中完成篡改负例：正常 runtime 文档返回 `0`；修改
`/opt/ptx-docs/manifest.toml` 后，隐藏 checker 返回 `1`；即使同时把公开 checker
替换为永远成功的脚本，完整 `/tests/test.sh` 仍返回 `1`。错误日志明确报告 runtime
manifest 与 verifier 的可信 manifest 不一致。

随后在 `catalyst-b200` GPU 1 上运行真实 Harbor Oracle：

```text
job:   jobs/ptx-rope-b200-oracle-hidden-policy-r10-20260821/
trial: ptx-rope-hidden-policy-r10__qsZwbkG
```

该 Job 用时 16 秒，`correctness=1.0`、`reward=0.9994813813564517`、
`speedup_vs_reference=0.9994813813564517`，无 exception；
`verifier/environment-policy.log` 记录隐藏 policy checker 通过。和前一次 Oracle 一样，
它使用了仅供 Oracle 的临时 public-network task 副本，不构成不可信 Agent 网络隔离证据。
临时副本的 Harbor legacy `task_checksum` 为
`b309c28bd5080870a850df3b989d25d584f829c7d1c735ead3fb138a3cfce6d5`。

当前 task 只在仓库中提供并固定一份 PTX ISA 官方来源记录 README，未重新分发 NVIDIA
PDF。它可以验证文档接口与 manifest 防篡改，但不等同于为 B200 Agent 提供了完整离线
PTX ISA 手册。

## Oracle 专用网络差异

Canonical task 中 Agent 策略是：

```toml
[agent]
network_mode = "no-network"
```

当前 Harbor Singularity backend 无法在容器启动后从 public setup 动态切换到
no-network Agent。为运行 Oracle，使用了 `/tmp` 临时 task 副本，并且只删除上述一行，
让 Oracle trial 全程继承 environment 的 public 网络。临时 trial 的 Harbor legacy
`task_checksum` 是：

```text
173fa36f294fa038c122b1f90afd1dbb9f76cda2bd68319ffc343020fe50c1e1
```

该 trial 没有运行不可信 Agent；instruction、Oracle、verifier、Reference、SIF 和 GPU
执行路径均与 canonical task 相同。因此它足以支持 task feasibility，但不能作为
Singularity 上正式 Direct-PTX Agent 隔离已经完成的证据。

## 安全负例：通过

两个作者侧负例在 B200 GPU 5 上运行，均保留 fail-closed 初始 reward：

| 负例 | verifier 结果 | correctness / reward |
|---|---|---:|
| 将 required entry 改名 | 缺少 `rope_forward_bf16_d128`，module load 前拒绝 | 0 / 0 |
| PTX 可汇编，但把 sine 项强制清零 | `num_tokens=2` 数值检查失败 | 0 / 0 |

第二个负例的首个报告差异是：

```text
candidate q_out tokens=2: wrong output at index=4096:
expected 1.71875, got 1.0703125
```

## 错误分类契约：通过

Verifier 和 `test.sh` 现在使用以下约定：

```text
0  Candidate 通过
1  Candidate 被拒绝；test.sh 保留 reward 0 后正常结束
2+ Reference、CUDA 环境或 verifier 自身错误；test.sh 非零退出
```

本地两个无 GPU 回归测试验证了“缺失 Candidate 返回 1”和“缺失隐藏 Reference 返回
2”。B200 上可汇编但 sine 错误的 Candidate 返回 `candidate_rejected`，reward `0`，而
`test.sh` 正常结束，因此 Harbor 会将它计为模型答案错误，不会伪装成基础设施异常。

## 尚未完成

- split-half 写成 interleaved、忽略 positions、只写 Q、Q/K head 混淆等剩余负例；
- 输入修改和越界写的隔离性负例；
- 慢但正确实现与多次重复计时；
- Codex、Qwen、K3、GLM 等真实 Agent 校准；
- Singularity 上可审计的 Agent 网络隔离或允许列表方案；
- 独立高性能 Reference 与 FlashInfer 外部对照。

完成这些项目之前，catalog 中应保持 `feasibility`，不能升级为 `validated`、
`calibrated` 或 `frozen`。
