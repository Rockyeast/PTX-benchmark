# PTX-Bench Task 使用说明

PTX-Bench 使用 Harbor 运行和评分要求 Agent 直接编写 NVIDIA PTX 的任务。发布任务见
[`CATALOG.md`](CATALOG.md)，设计与评分原则见 [`DESIGN.md`](DESIGN.md)，正式结果见
[`RESULTS.md`](RESULTS.md)。

新建和发布任务只支持 Docker-only `direct-ptx-v1`。旧 pilot 仅作为实验记录保留，
生成器和 checker 不提供旧模板 fallback；有价值的旧题必须逐题迁移到 v1。

## 1. 任务模式

Agent 修改 `/app/kernel.ptx`，可以查看公开的数学参考和自测工具。隐藏 verifier 在
Agent 结束后检查正确性；只有全部正确才计算性能。允许使用 `ptxas` 和 `cuobjdump`，
不允许通过 CUDA C++、Triton、NVRTC、NVVM 或其他高层编译器生成提交。

## 2. 目录结构

```text
tasks/
├── README.md                            # 全部 task 的创建、运行和验证说明
├── DESIGN.md                            # 全部 task 的设计与评分原则
├── CATALOG.md                           # 正式发布的 task 清单
├── RESULTS.md                           # 正式发布的实验结果
└── <task-name>/
    ├── task.toml                        # Harbor 时间、网络、硬件要求和产物配置
    ├── instruction.md                   # Agent 可见的完整任务合同
    ├── environment/
    │   ├── Dockerfile                   # 构建纯 PTX 作答环境
    │   ├── docker-compose.yaml          # 将指定 GPU 分配给容器
    │   ├── kernel.ptx                   # Agent 修改的 PTX 起始骨架
    │   ├── reference.py                 # 公开、正确但较慢的数学参考
    │   └── measure.py                   # 公开正确性与性能自测
    ├── solution/
    │   ├── solve.sh                     # 将性能 Reference 放到 /app/kernel.ptx
    │   └── performance_reference.ptx    # 发布前验证使用的高性能 PTX
    └── tests/
        ├── test.sh                      # Harbor 隐藏评分入口
        ├── verifier.py                  # 隐藏真值、cases、测速和 reward
        ├── performance_baseline.ptx     # 性能零分线
        └── performance_reference.ptx    # 性能满分线
```

Docker 构建只把 `environment/` 中的公开文件复制到 `/app`。普通 Agent 看不到
`solution/`；Harbor 在 Agent 结束后才提供 `tests/`。运行结束后，Candidate 被归档到
`jobs/<job>/<trial>/artifacts/kernel.ptx`。

## 3. 创建任务

从仓库根目录运行：

```bash
python3 scripts/scaffold_direct_ptx_task.py \
  <task-name> \
  --title "<title>" \
  --description "<description>" \
  --entry <entry-name> \
  --agent-timeout <seconds> \
  --network-mode <no-network-or-allowlist>
```

生成的起始骨架故意不能发布。任务作者必须补齐任务合同、公开参考、自测、隐藏 verifier、
性能 Baseline 和性能 Reference，然后运行静态检查：

```bash
python3 scripts/check_direct_ptx_task.py tasks/<task-name> --release
```

## 4. 在 GPU 上验证任务

选择空闲 GPU 后，必须分别验证性能 Reference 和未修改的起始骨架：

```bash
export PTX_BENCH_GPU_DEVICE=<gpu-index>

.venv/bin/harbor run \
  --path tasks/<task-name> \
  --agent oracle \
  --env docker \
  --n-attempts 1 \
  --n-concurrent 1 \
  --job-name <task-name>-reference-check \
  --yes

.venv/bin/harbor run \
  --path tasks/<task-name> \
  --agent nop \
  --env docker \
  --n-attempts 1 \
  --n-concurrent 1 \
  --job-name <task-name>-starter-check \
  --yes
```

性能 Reference 应通过全部正确性检查并获得接近满分的 reward；起始骨架应得到 `0`。
两次运行都必须经过真实的 `tests/test.sh`，不能只单独调用 verifier 中的某个函数。

## 5. 运行真实 Agent

```bash
export PTX_BENCH_GPU_DEVICE=<gpu-index>

.venv/bin/harbor run \
  --path tasks/<task-name> \
  --agent <agent-name> \
  --model <model-name> \
  --env docker \
  --n-attempts <attempts> \
  --n-concurrent 1 \
  --job-name <job-name> \
  --yes
```

比较模型时必须固定 task 内容、硬件、Agent、模型配置、网络/文档设置、时间预算和独立
attempt 数量。Correctness、是否正常结束、原始性能和归一化 reward 分开报告。

## 6. 运行边界

- Agent 网络策略由 `task.toml` 的 `[agent]` 配置；正式任务不能使用不受限制的公网。
- 公开 `measure.py` 不得泄露隐藏 cases、性能 Baseline、性能 Reference 或最终 reward。
- `tests/` 中的正确性真值和评分代码只在评分阶段提供，不能信任 Agent 可修改的公开文件。
- 高分结果发布前必须审计 trajectory，确认没有高层源码生成、禁用工具、答案下载或评分篡改。
