# PTX Benchmark

本项目计划使用 Harbor 构建 PTX 编程 Agent benchmark。

Harbor 基础链路和第一个 B200 PTX smoke task 已跑通。第一道带正确性与性能评分
的校准题 `ptx-vector-add` 已完成实现、B200 正负例和真实 Agent 校准。

## 目录结构

```text
PTX-benchmark/
├── README.md
├── docs/                       # 我们自己的项目文档
│   ├── 任务.md                  # 当前计划和验收标准
│   ├── harbor-job-output.md    # Harbor jobs 输出目录结构和阅读方法
│   ├── harbor-task-lifecycle.md # Task 从 instruction 到 reward 的运行链路
│   ├── dataset-design.md       # 数据集分层、task 粒度、评分与 release gate
│   ├── task-catalog.md         # 已有任务、五道 pilot 和候选池
│   ├── 新任务阅读清单.md         # 每个新 task 固定看什么、按什么顺序看
│   ├── tasks/
│   │   └── ptx-vector-add-spec.md # 第一题的接口、正确性、计时和评分规格
│   ├── source/
│   │   └── 录音.txt              # 卓明会议录音转写
│   └── setup/
│       ├── harbor-cli.md       # Harbor 安装与 CLI 实测记录
│       ├── docker-wsl.md       # Docker Desktop / WSL 配置记录
│       └── harbor-singularity-gpu-allocation.md # 远端并行 GPU 分配
├── images/                     # 可复现的 PTX/Codex SIF 定义
├── patches/harbor/             # 远端 Harbor Singularity 补丁
├── tasks/                      # Harbor dev dataset manifest 与 task 源码
│   ├── dataset.toml            # ptx-bench/ptx-bench-dev，固定 task digest
│   ├── harbor-smoke/           # 最小文本任务
│   ├── ptx-add-one/            # B200 PTX 正确性 smoke task
│   ├── ptx-vector-add/         # CPU 真值门控 + nvcc 性能基线校准题
│   └── ptx-rmsnorm-forward-bf16-n4096/ # 第一件正式算子候选题
├── jobs/                       # Harbor 自动生成的运行结果
└── .venv/                      # 本地 Python/Harbor 环境
```

## 哪些属于 Harbor，哪些属于本项目

- Harbor 框架安装在 `.venv/lib/python3.12/site-packages/harbor/`，CLI 入口是 `.venv/bin/harbor`，不直接修改。
- `tasks/harbor-smoke/` 使用 Harbor 规定的目录格式，但题目、环境、参考答案和测试由本项目维护。
- `tasks/ptx-add-one/` 和之后的 PTX tasks 也属于本项目；Harbor 只负责运行生命周期。
- `patches/harbor/` 保存可复现补丁；实际生效的个人 Harbor 副本位于远端
  `/home/zhuominc/rockyeast-work/harbor-ptx/`，共享 `~/harbor` 未修改。
- `jobs/` 由 Harbor 每次运行时自动生成，不是任务源码。
- `docs/` 中的需求、计划和环境记录完全属于本项目。

## 建议阅读顺序

1. [`docs/source/录音.txt`](docs/source/%E5%BD%95%E9%9F%B3.txt)：了解项目来源和目标。
2. [`docs/任务.md`](docs/%E4%BB%BB%E5%8A%A1.md)：查看任务进度与验收标准。
3. [`docs/dataset-design.md`](docs/dataset-design.md)：先看整个数据集的分层、家族和 release gate。
4. [`docs/task-catalog.md`](docs/task-catalog.md)：查看已有任务、五道 pilot 和候选池。
5. [`tasks/harbor-smoke/`](tasks/harbor-smoke/)：查看最小 Harbor task。
6. [`tasks/ptx-add-one/`](tasks/ptx-add-one/)：查看已跑通的 PTX 正确性链路。
7. [`docs/tasks/ptx-vector-add-spec.md`](docs/tasks/ptx-vector-add-spec.md)：查看第一道性能校准题。
8. [`docs/新任务阅读清单.md`](docs/%E6%96%B0%E4%BB%BB%E5%8A%A1%E9%98%85%E8%AF%BB%E6%B8%85%E5%8D%95.md)：以后每个新 task 都按这份清单阅读和验收。
9. [`docs/harbor-job-output.md`](docs/harbor-job-output.md)：理解一次运行生成的 Job、Trial、Agent 和 Verifier 文件。
10. [`docs/harbor-task-lifecycle.md`](docs/harbor-task-lifecycle.md)：查看 instruction、Agent、Verifier 和 reward 的完整实证链路。

## 当前进度

- Harbor CLI `0.21.0` 已安装并检查。
- Docker Desktop 与 Ubuntu WSL2 已连通。
- `harbor-smoke` 的 Oracle 正向验证得到 reward `1`。
- `nop` 负向验证得到 reward `0`。
- 临时错误内容 `harbor-wrong\n` 的 Oracle 负向验证得到 reward `0`，且只有内容测试失败。
- 真实 Codex Agent 验证得到 reward `1`。
- Harbor task 生命周期文档已经完成，并以实际 task、运行日志和 Harbor `0.21.0` 源码为依据。
- 远端 B200 上的 `ptx-add-one` Oracle、负例和真实 Codex 均已完成验证。
- 通用 CUDA SIF、Codex 专用 SIF 和 Singularity GPU 自动分配补丁已经完成。
- `ptx-vector-add` 已实现 Oracle、CPU 独立正确性门控、内存 canary、隐藏
  `nvcc` reference、CUDA Event 计时和多指标 reward。B200 上 Oracle 三次 reward
  为 `0.990683 / 0.997697 / 0.987482`，慢速正确实现为 `0.267804`，缺失 entry
  和越界写负例均为 `0`。真实 Codex Agent correctness 为 `1`，原始
  `speedup_vs_reference=1.000609`，封顶 reward 为 `1`。
- `ptx-rmsnorm-forward-bf16-n4096` 已完成 B200 feasibility Oracle 首跑：六个
  correctness case 和三个 performance shape 通过；它尚未完成 INT21 外部性能对照、
  负例和多 Agent 校准。
- `tasks/dataset.toml` 已建立 `ptx-bench/ptx-bench-dev`，目前只包含 vector-add
  校准题和 RMSNorm pilot，并以 SHA-256 digest 固定具体 task 内容。

原始运行日志保存在本机 `jobs/`，该目录默认不纳入版本控制。
