# Direct-PTX task authoring template v0

本项目使用一个未冻结的 v0 模板约束新 PTX task，防止不同作者或 Agent 为每道题
临时发明目录、工具策略、reward 字段和验收流程。模板源码位于：

```text
templates/direct-ptx-task-v0/
```

生成器位于：

```text
scripts/scaffold_direct_ptx_task.py
```

## 创建新任务

以下命令只生成 fail-closed scaffold，不会生成一道已经可运行的题：

```bash
python3 scripts/scaffold_direct_ptx_task.py \
  ptx-layernorm-forward-bf16-n4096 \
  --title "Implement BF16 LayerNorm forward in direct PTX" \
  --description "Implement and benchmark a BF16 LayerNorm PTX kernel on NVIDIA B200." \
  --entry layernorm_fwd_bf16_n4096 \
  --network-mode no-network
```

默认目标是 NVIDIA B200 / `sm_100`，Agent timeout 为 600 秒，Verifier timeout
为 180 秒。其他硬件必须显式传入，例如：

```bash
python3 scripts/scaffold_direct_ptx_task.py \
  ptx-example-l40 \
  --description "Example direct-PTX task for NVIDIA L40." \
  --entry example_kernel \
  --hardware-target "NVIDIA L40" \
  --gpu-type L40 \
  --sm 89 \
  --network-mode allowlist \
  --allowed-host lovelace.ece.local.cmu.edu
```

生成器不提供 Agent 网络默认值，作者必须显式选择 Agent 作答阶段的策略。正常的
served-model setting 使用模型 API `allowlist`；不需要外部模型连接时使用
`no-network`。模板保留 `[environment] public` 供可信的构建、setup 和 verifier
阶段获取固定且经过哈希校验的依赖，Harbor 在 Agent 开始作答时应用更严格的阶段覆盖：

```bash
python3 scripts/scaffold_direct_ptx_task.py \
  ptx-example-b200 \
  --description "Example allowlisted task." \
  --entry example_kernel \
  --network-mode allowlist \
  --allowed-host model-api.internal.example
```

阶段级配置示例：

```toml
[agent]
network_mode = "allowlist"
allowed_hosts = ["model-api.internal.example"]

[environment]
network_mode = "public"
```

如果 Agent 需要访问多个模型 API 主机，可以重复传入 `--allowed-host`。固定 PTX ISA
文档不通过公网访问，而是放在 task 的 `environment/ptx-docs/` 中并复制到容器内的
`/opt/ptx-docs`。

## 模板固定什么

- Harbor task 基本目录和 schema；
- 只评分 `/app/kernel.ptx`；
- correctness 是 performance 的硬门槛；
- reward 初始值与失败路径；
- 1 MiB PTX 文件上限；
- B200 direct-PTX 基础环境：保留 `ptxas`/`cuobjdump`，移除 `nvcc`、NVRTC、
  NVVM；
- 必须显式选择 environment baseline，并保证 release 的有效 Agent 策略不是
  `public`；
- `/opt/ptx-docs` 固定只读文档接口及来源、版本、许可证、SHA-256 manifest；
- 构建期与 verifier 前都会执行基础环境工具策略检查；
- Oracle、`test.sh` 和 verifier 的 fail-closed 行为；
- `template_version=direct-ptx-v0` 和 `release_ready=false` 元数据。

## 每道题必须填写什么

- operation、direction/fusion、dtype/layout、hardware target、shape specialization；
- 完整 entry、参数顺序与类型；
- grid、block、dynamic shared memory；
- tensor shape/layout、边界行为和 aliasing；
- 独立 correctness truth、容差、隐藏 case 和内存 canary；
- 同 ABI 性能 Reference、计时规模、原始样本和 reward 公式；
- 可审计 Oracle、负例、许可证和来源；
- 实际硬件上的重复性与多个真实 Agent 校准。

生成后必须清除所有 `TODO`。模板 verifier 会主动抛出
`NotImplementedError` 并写 reward 0，防止空 scaffold 被误认为已经完成。

## 检查 scaffold 与 release

开发过程中运行结构检查：

```bash
python3 scripts/check_direct_ptx_task.py tasks/<task-name>
```

冻结前把 `release_ready` 改为 `true`，并运行严格检查：

```bash
python3 scripts/check_direct_ptx_task.py tasks/<task-name> --release
```

严格检查会按 Harbor 0.21 的覆盖规则解析 Agent 的有效网络策略：显式 `[agent]`
策略优先，否则继承 `[environment]` baseline。它会拒绝有效 Agent 策略为 `public`、
未清除的 `TODO`、缺失或哈希不匹配的 PTX 文档，以及未显式声明为 release-ready 的
任务。它只覆盖静态 release 条件；Oracle、负例、真实 GPU 性能和多模型校准仍需通过
实际 Job 提供证据。

## 环境边界

v0 复用已经在 B200 验证过的“CUDA devel 镜像后移除高层编译前端”方案。它是可审计
的开发基线，不代表最终最小镜像已经冻结。完成三到五道不同家族 pilot 后，再决定
是否改为多阶段构建、固定预构建镜像 digest，并冻结 v1。

不要为单道 task 随意增加编译器、Python 包或网络权限。确有必要时，应把差异写入
规格文档，并判断它是否已经构成新的 benchmark setting。`ptxas` 和 `cuobjdump` 是
允许的 PTX 级验证工具；允许 `nvcc`、Triton 等高层到 PTX 工具时，必须另列为
tool-assisted setting。
