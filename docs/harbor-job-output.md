# Harbor `jobs/` 输出目录结构

`jobs/` 是 Harbor 每次运行自动生成的结果目录。它保存配置快照、Agent 轨迹、Verifier 输出和最终得分，不是 task 源码，通常不应手工修改。

下面以真实运行 `jobs/harbor-smoke-codex/` 为例。

## 总体结构图

```text
jobs/
└── harbor-smoke-codex/                    # 一个 Job：一次 harbor run
    ├── config.json                        # 用户提交的 Job 级运行配置
    ├── lock.json                          # Harbor 解析并锁定后的完整 Job 配置
    ├── result.json                        # 整个 Job 的汇总成绩
    ├── job.log                            # Job 调度和汇总过程日志
    │
    └── harbor-smoke__Ukp94kC/             # 一个 Trial：一次具体任务尝试
        ├── config.json                    # 该 Trial 的实际启动参数
        ├── lock.json                      # 该 Trial 锁定的任务、Agent、环境和评分配置
        ├── result.json                    # 该 Trial 的详细结果、耗时、Token 和 reward
        ├── trial.log                      # 环境、Agent、Verifier 各阶段的生命周期日志
        │
        ├── agent/                         # Agent 执行记录
        │   ├── codex.txt                  # Codex CLI 的原始事件输出
        │   ├── trajectory.json            # Harbor 统一格式的结构化 Agent 轨迹
        │   └── sessions/                  # Codex 自身的原始会话归档（可选）
        │       └── YYYY/MM/DD/
        │           └── rollout-*.jsonl
        │
        ├── verifier/                      # 正式评分记录
        │   ├── test-stdout.txt            # tests/test.sh 和 pytest 的标准输出
        │   ├── reward.txt                 # Verifier 交给 Harbor 的最终原始分数
        │   └── ctrf.json                  # 机器可读的标准测试报告
        │
        └── artifacts/
            └── manifest.json              # 额外产物的收集清单和状态
```

## 两层目录分别代表什么

```text
Job（一次 harbor run）
└── Trial 1（任务 A 的第一次尝试）
└── Trial 2（任务 A 的第二次尝试）
└── Trial 3（任务 B 的一次尝试）
```

- Job 层对应一次 `harbor run`，可能包含一个或多个 trial。
- Trial 层对应某个 Agent 对某个 task 的一次具体尝试。
- `harbor-smoke__Ukp94kC` 中，前半部分来自 task 名称，后半部分是 Harbor 生成的随机后缀。

本次命令只运行一个 task、一个 Agent、一次尝试，因此 Job 下只有一个 Trial。

## Job 层文件

### `config.json`

记录用户交给 Harbor 的 Job 级配置，例如：

- Job 名称；
- 并发 trial 数；
- Agent 和模型；
- task 路径。

它更接近原始运行请求。

### `lock.json`

记录 Harbor 解析默认值后锁定的完整配置，例如：

- Harbor 版本；
- task 版本和内容摘要；
- Agent、模型和环境参数；
- Verifier 模式；
- 重试策略。

它比 `config.json` 更完整，主要用于确认“这次实验究竟按什么配置运行”。认证值会被隐藏。

### `result.json`

汇总整个 Job，包括：

- trial 总数和完成状态；
- 各 Agent/模型组合的平均 reward；
- 异常数量；
- Token 总量和总成本。

这是查看一次运行总体成绩时的第一个入口。

### `job.log`

记录 Job 调度过程。Job 没有正常产生 trial、并发调度异常或汇总失败时，优先查看它。

## Trial 层文件

### `config.json` 与 `lock.json`

- `config.json` 是本 Trial 的启动参数，并通过 `job_id` 指回所属 Job。
- `lock.json` 是本 Trial 最终锁定的 task、Agent、Docker 环境和 Verifier 配置。

平时看成绩不需要先读它们；需要复现或核对配置时再看。

### `result.json`

这是单次 Trial 的详细成绩单，包括：

- task、Trial 和 Job 的关联；
- task checksum；
- 实际 Agent、版本和模型；
- Agent Token 与费用；
- Verifier reward；
- 异常信息；
- 环境准备、Agent 安装、Agent 执行和 Verifier 的时间。

### `trial.log`

记录一个 Trial 从创建环境到清理环境的生命周期。遇到 Docker 构建、Agent 安装、超时或 Verifier 启动问题时查看。

## `agent/`：Agent 做了什么

### `codex.txt`

Codex CLI 的原始事件流。本次记录显示 Codex：

1. 收到任务；
2. 创建 `/app/answer.txt`；
3. 使用 `od` 检查文件字节；
4. 返回完成消息。

不同 Agent 的文件名可能不同，例如 Oracle Trial 中是 `oracle.txt`。

### `trajectory.json`

Harbor 将 Agent 行为整理成统一的结构化轨迹，便于 Viewer、上传和后续分析。它与 `codex.txt` 内容有重叠，但更适合程序读取。

### `sessions/.../rollout-*.jsonl`

Codex CLI 自己保存的完整会话事件。Harbor 在 Agent 结束后将其从容器中的 `CODEX_HOME/sessions` 复制到 Trial 目录，再据此生成 `trajectory.json`。这是更底层的调试材料，普通结果检查不必阅读；不同 Agent 也不一定产生该目录。

## `verifier/`：为什么得到这个分数

### `test-stdout.txt`

保存 `/tests/test.sh` 的输出。本次可以看到 pytest 收集两个测试，并且两个都通过。

### `reward.txt`

Verifier 与 Harbor 之间的最终分数接口。本次内容只有：

```text
1
```

Harbor 读取后，在 Trial 和 Job 的 `result.json` 中记录为 `1.0`。

### `ctrf.json`

pytest 插件生成的 CTRF 标准 JSON 报告，包含测试名称、状态和耗时，主要供程序、报表工具或 Harbor Viewer 使用。人工排查通常先看 `test-stdout.txt`。

## `artifacts/`：额外产物

`manifest.json` 记录 Harbor 尝试收集哪些额外文件、来源、目标位置以及收集状态。

当前 smoke task 没有产生额外 artifact，所以清单状态为 `empty`。以后 PTX 任务可能在这里收集编译日志、性能结果或其他明确配置的产物。

## 一次运行的数据流

```text
harbor run
    │
    ├── 写入 Job config.json / lock.json
    │
    └── 创建 Trial
         ├── 准备 Docker 环境
         ├── Agent 执行 ───────────→ agent/
         ├── Verifier 测试 ────────→ verifier/
         ├── 汇总单次结果 ─────────→ Trial result.json
         └── 汇总所有 Trial ───────→ Job result.json
```

## 推荐阅读顺序

只想确认成绩：

```text
Job result.json
→ Trial result.json
→ verifier/test-stdout.txt
→ verifier/reward.txt
```

想知道 Agent 做了什么：

```text
Trial result.json
→ agent/codex.txt
→ agent/trajectory.json（需要结构化细节时）
```

运行失败时：

```text
Job result.json
→ Trial result.json 的 exception_info
→ trial.log
→ job.log
→ Agent 或 Verifier 对应日志
```

需要复现实验配置时：

```text
Job config.json
→ Job lock.json
→ Trial lock.json
```

## 特殊情况

- `--install-only` 只验证 Agent 安装，不执行 Agent 和 Verifier，因此对应 Trial 可能没有 `agent/`、`verifier/` 和 reward；这不等于任务得了零分。
- NOP Agent 不执行任务，因此可能没有 Agent 轨迹，但 Verifier 仍会运行并产生 reward。
- 文件是否存在取决于 Agent 类型、任务配置以及运行是否执行到相应阶段。
- `jobs/` 是可重新生成的本地运行证据，已被项目 `.gitignore` 忽略；不要把它当作 task 源码手工维护。
