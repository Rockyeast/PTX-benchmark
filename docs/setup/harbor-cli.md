# Harbor CLI 安装与检查记录

首次检查日期：2026-08-18；定制分支核验日期：2026-08-31

## 安装结果

- Python：`3.12.3`
- Harbor：`0.21.0`
- 安装位置：`/home/rockyeast/PTX-benchmark/.venv`
- CLI 路径：`/home/rockyeast/PTX-benchmark/.venv/bin/harbor`
- 当前安装来源：`Rockyeast/harbor` 的 `ptx-benchmark/harbor-0.21` 分支，
  commit `8527f653aa866bd56ee5f96d2e93a0ff72dee01a`

当前仍显示版本 `0.21.0`，因为定制分支以官方 `v0.21.0` 为底座，并没有伪造新的
Harbor 发布版本。精确来源可通过以下命令核对：

```bash
python -c "from importlib.metadata import distribution; print(distribution('harbor').read_text('direct_url.json'))"
```

重新安装当前固定 commit：

```bash
.venv/bin/pip install --no-deps --force-reinstall \
  'git+https://github.com/Rockyeast/harbor.git@8527f653aa866bd56ee5f96d2e93a0ff72dee01a'
```

分支和历史补丁状态见 [`../../patches/harbor/README.md`](../../patches/harbor/README.md)。

使用前先激活环境：

```bash
cd /home/rockyeast/PTX-benchmark
source .venv/bin/activate
harbor --version
```

实际版本输出：

```text
0.21.0
```

## 指定命令检查

以下三个命令均执行成功，exit code 均为 `0`：

```bash
harbor --help
harbor run --help
harbor task --help
```

### `harbor --help`

顶层 CLI 当前包含：

```text
check, analyze, init, run, exec, publish, upload, add, download,
remove, sync, view, adapter, task, dataset, job, hub, trial,
cache, plugins, auth, version
```

结论：当前版本确实存在 `harbor task`，不需要改用猜测的旧命令。

### `harbor run --help`

`harbor run` 是 `harbor job start` 的别名。与本项目最相关的参数为：

```text
-p, --path       本地 task 或 dataset 目录
-a, --agent      Agent；当前列表包含 oracle、codex、claude-code 等
-m, --model      Agent 使用的模型
-e, --env        环境类型；默认是 docker
-o, --jobs-dir   trial 结果目录；默认是 jobs
```

因此后续运行本地 Oracle task 的命令形式是：

```bash
harbor run -p tasks/harbor-smoke -a oracle
```

### `harbor task --help`

当前子命令为：

```text
init, download, start-env, update, annotate, visibility, migrate
```

继续检查 `harbor task init --help` 后，确认当前创建任务的实际命令是：

```bash
harbor task init [name] --tasks-dir <path>
```

例如：

```bash
harbor task init ptx-bench/harbor-smoke --tasks-dir tasks
```

## Docker/WSL 前置条件

Docker Desktop 与 Ubuntu WSL2 的检查、修复和验证过程单独记录在
[`docker-wsl.md`](docker-wsl.md)，避免在 CLI 文档中重复维护同一份信息。
