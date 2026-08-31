# Harbor Task 运行生命周期

本文使用本机 Harbor `0.21.0`、`tasks/harbor-smoke/` 和真实运行 `jobs/harbor-smoke-codex/` 说明一个单阶段 task 如何从题目变成 reward。

相关目录的逐文件说明见 [`harbor-job-output.md`](harbor-job-output.md)。

## 总览

```text
宿主机 task 源码
tasks/harbor-smoke/
├── instruction.md
├── task.toml
├── environment/Dockerfile
├── solution/solve.sh
└── tests/
    ├── test.sh
    └── test_outputs.py
          │
          │ harbor run
          ▼
Harbor 创建 Docker 容器
├── 工作目录 /app
├── 将 instruction 文本交给 Agent
├── Agent 修改 /app
├── Agent 结束后复制 tests/ 到 /tests
├── Verifier 在同一容器检查 /app
└── Verifier 写入 /logs/verifier/reward.txt
          │
          ▼
宿主机运行结果
jobs/<job-name>/<trial-name>/
├── agent/
├── verifier/
├── result.json
└── trial.log
```

本次 `task.toml` 没有配置 `[[steps]]`，因此这是一个 single-step trial：一份 instruction、一次 Agent 执行和一次 Verifier 评分。

## 1. Harbor 从哪个文件读取 Agent 指令？

Harbor 从 task 根目录的 `instruction.md` 读取指令：

```text
tasks/harbor-smoke/instruction.md
```

本次要求 Agent 创建 `/app/answer.txt`，内容必须恰好为 `harbor-ok\n`。

Harbor 源码路径：

```text
.venv/lib/python3.12/site-packages/harbor/models/task/task.py
└── Task.__init__()
    └── self.paths.instruction_path.read_text()
```

随后 single-step trial 将读取到的字符串交给 Agent：

```text
.venv/lib/python3.12/site-packages/harbor/trial/single_step.py
└── SingleStepTrial._run_agent()
    └── instruction=self.task.instruction
```

默认 Harbor 中，指令不是把 `instruction.md` 文件挂进容器，而是作为 `codex exec`
的最后一个命令行参数传入。真实证据在：

```text
jobs/harbor-smoke-codex/harbor-smoke__Ukp94kC/trial.log
```

该日志完整显示了 `codex exec ... -- '# Create the output file ...'`，同时 stdin 被重定向为 `/dev/null`。

本项目维护的 Harbor 分支
`Rockyeast/harbor:ptx-benchmark/harbor-0.21` 在这个默认行为上增加了一个可重读镜像：
Harbor 将同一 Agent phase 的有效 instruction 写入
`/opt/task/instruction.md`，设置 `HARBOR_INSTRUCTION_PATH`，并继续把 instruction 文本
作为 prompt 传给 Agent。镜像不替代 prompt，也不属于 `/opt/ptx-docs`。仓库中的
`patches/harbor/0006-agent-instruction-mirror.patch` 只作为离线恢复副本。

## 2. Agent 启动时能看到哪些目录和文件？

对于本次真实 Codex Trial，容器内主要视图是：

```text
/
├── app/                    # Task 工作目录；Agent 在这里创建 answer.txt
├── opt/task/
│   └── instruction.md      # PTX Harbor 分支提供的有效 instruction 只读镜像
├── logs/
│   ├── agent/             # Agent 日志挂载目录
│   ├── verifier/          # Verifier 日志挂载目录
│   └── artifacts/         # Agent 可发布的额外产物目录
├── tmp/
│   ├── codex-home/        # 本次 Codex 临时状态
│   └── codex-secrets/     # 临时认证文件
└── Ubuntu 基础系统和 Harbor 安装的 Agent 工具
```

本 task 的 Dockerfile 只有 Ubuntu 24.04 和 `WORKDIR /app`，没有把项目源码复制到 `/app`，因此 Codex 启动时 `/app` 中没有 task 源码；它只收到了 instruction 文本。

两个重要的隔离边界：

- `/solution` 只由特殊的 Oracle Agent 复制和执行。真实 Codex Agent 不会获得 `tasks/harbor-smoke/solution/solve.sh`。
- `/tests` 由 Verifier 在 Agent 结束后复制。Codex 执行阶段不会获得 `tasks/harbor-smoke/tests/`。

Harbor 源码对此有直接说明：

```text
.venv/lib/python3.12/site-packages/harbor/models/trial/paths.py
└── EnvironmentPaths
    ├── /solution  # Copied over by the OracleAgent only
    └── /tests     # Copied over by the Verifier after the agent runs
```

额外说明：如果未来在 Dockerfile 中 `COPY` 文件、在运行配置中增加 mounts，或者注入 skills，Agent 可见内容会随任务配置变化；上述结论针对当前 `harbor-smoke`。

## 3. Agent 的工作目录是什么？

本次工作目录是：

```text
/app
```

来源是：

```dockerfile
# tasks/harbor-smoke/environment/Dockerfile
WORKDIR /app
```

本 task 没有在 `task.toml` 中用 `environment.workdir` 覆盖目录，因此容器沿用 Dockerfile 的 `WORKDIR /app`。真实 Codex session 也将 `cwd` 记录为 `/app`。

Codex 最终创建：

```text
/app/answer.txt
```

对应 Agent 原始事件：

```text
jobs/harbor-smoke-codex/harbor-smoke__Ukp94kC/agent/codex.txt
```

## 4. Agent 完成后，Harbor 如何启动 Verifier？

本次 Trial 的 `verifier_environment_mode` 是 `shared`，所以 Agent 和 Verifier 使用同一个 Docker 容器，Verifier 能检查 Agent 留在 `/app` 中的文件。

执行顺序是：

```text
Agent 执行结束
→ Harbor 保存 Agent 日志
→ Harbor 收集 artifacts
→ 保持同一容器运行
→ 将 tasks/harbor-smoke/tests/ 上传到容器 /tests
→ chmod +x /tests/test.sh
→ 执行 /tests/test.sh
→ Verifier 完成后停止容器
```

Harbor 源码路径：

```text
.venv/lib/python3.12/site-packages/harbor/trial/single_step.py
└── SingleStepTrial._run()
    ├── _run_agent()
    ├── _upload_agent_logs()
    ├── _collect_artifacts()
    ├── _run_verifier()
    └── _stop_agent_environment()

.venv/lib/python3.12/site-packages/harbor/verifier/verifier.py
└── Verifier.verify()
    ├── environment.upload_dir(..., target_dir="/tests")
    └── environment.exec(test_script)
```

本 task 的 `/tests/test.sh` 使用 pytest 运行 `/tests/test_outputs.py`。真实输出在：

```text
jobs/harbor-smoke-codex/harbor-smoke__Ukp94kC/verifier/test-stdout.txt
```

其中两个检查分别确认 `/app/answer.txt` 存在，以及原始字节等于 `b"harbor-ok\n"`。

## 5. Verifier 如何把分数返回给 Harbor？

`tasks/harbor-smoke/tests/test.sh` 根据 pytest 退出码写分数：

```text
pytest 全部通过
→ /logs/verifier/reward.txt 写入 1

任一测试失败
→ /logs/verifier/reward.txt 写入 0
```

`/logs/verifier` 是挂载目录，因此容器中的 reward 会出现在宿主机 Trial 目录：

```text
jobs/<job-name>/<trial-name>/verifier/reward.txt
```

Harbor 的 `Verifier.verify()` 优先读取 `reward.json`；本 task 没有该文件，因此读取 `reward.txt`，将文本解析为浮点数，再写入 Trial 的：

```json
"verifier_result": {
  "rewards": {
    "reward": 1.0
  }
}
```

Job 层最后对所有 Trial 的 reward 做汇总，生成 Job 根目录的 `result.json`。

## 6. Agent 轨迹、stdout/stderr 和 reward 保存在哪里？

以真实 Codex Trial 为例：

| 内容 | 实际路径 | 用途 |
|---|---|---|
| Codex 原始事件 | `jobs/harbor-smoke-codex/harbor-smoke__Ukp94kC/agent/codex.txt` | 看 Agent 说了什么、改了什么、执行了什么 |
| Codex 原生 session | `.../agent/sessions/YYYY/MM/DD/rollout-*.jsonl` | Codex 更底层的完整会话记录 |
| Harbor 统一轨迹 | `.../agent/trajectory.json` | Viewer、上传和跨 Agent 统一分析 |
| Trial 编排日志 | `.../trial.log` | 看环境准备、Agent 命令、日志收集和清理 |
| Verifier stdout | `.../verifier/test-stdout.txt` | 看依赖安装、pytest 通过或失败详情 |
| Verifier stderr | `.../verifier/test-stderr.txt` | 仅在产生独立 stderr 文件时出现；本次没有 |
| 原始 reward | `.../verifier/reward.txt` | Verifier 交给 Harbor 的最终数字 |
| 单次详细结果 | `.../result.json` | Agent、耗时、Token、异常和 reward |
| Job 汇总结果 | `jobs/harbor-smoke-codex/result.json` | Trial 数量、平均分、异常和总成本 |

文件并非每次都全部出现。例如 `--install-only` 不执行 Agent 和 Verifier，NOP 也可能没有 Agent 轨迹。

## 正向和负向验证证据

### 正确答案

真实 Codex 创建了：

```text
/app/answer.txt = b"harbor-ok\n"
```

两个测试均通过，结果为：

```text
jobs/harbor-smoke-codex/harbor-smoke__Ukp94kC/verifier/reward.txt
→ 1
```

### 错误内容实验：具体改了什么？

为了验证测试不只是检查“文件存在”，我们没有修改正式 task，而是将其复制到临时目录：

```text
/tmp/harbor-smoke-wrong.u65i6a
```

只修改临时副本的一个文件、一行内容：

```diff
 # /tmp/harbor-smoke-wrong.u65i6a/solution/solve.sh
-printf 'harbor-ok\n' > /app/answer.txt
+printf 'harbor-wrong\n' > /app/answer.txt
```

`instruction.md` 和 `tests/test_outputs.py` 没有改，测试仍要求 `b"harbor-ok\n"`。随后用 Oracle Agent 执行临时错误解：

```bash
harbor run \
  --path /tmp/harbor-smoke-wrong.u65i6a \
  --agent oracle \
  --n-concurrent 1 \
  --job-name harbor-smoke-wrong-content \
  --yes
```

使用 Oracle 是为了让它确定性地执行临时 `solution/solve.sh`；真实模型可能自行纠正错误，不能稳定构造负例。

结果为：

```text
test_output_file_exists   PASSED
test_output_file_contents FAILED

实际：b"harbor-wrong\n"
期望：b"harbor-ok\n"
reward：0
Exceptions：0
```

证据路径：

```text
jobs/harbor-smoke-wrong-content/result.json
jobs/harbor-smoke-wrong-content/harbor-smoke-wrong.u65i6a__3Mosoac/verifier/test-stdout.txt
jobs/harbor-smoke-wrong-content/harbor-smoke-wrong.u65i6a__3Mosoac/verifier/reward.txt
```

这说明 Harbor 流程本身正常完成，零分来自内容断言失败，而不是 Docker、Oracle 或 Verifier 异常。正式文件 `tasks/harbor-smoke/solution/solve.sh` 始终保持正确的 `harbor-ok\n`。

## 最终闭环

```text
instruction.md 定义任务要求
        ↓
Dockerfile 提供隔离环境和 /app 工作目录
        ↓
Agent 只按 instruction 修改 /app
        ↓
Verifier 在 Agent 结束后获得 tests/
        ↓
test.sh 运行 test_outputs.py
        ↓
reward.txt 返回分数
        ↓
Trial result.json 记录单次结果
        ↓
Job result.json 汇总整次运行
```
