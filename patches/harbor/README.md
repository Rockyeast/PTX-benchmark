# Harbor 定制状态

当前 Direct-PTX 实验以官方 Harbor `v0.21.0` 为固定底座。可安装的维护版本位于：

```text
repository: https://github.com/Rockyeast/harbor.git
branch:     ptx-benchmark/harbor-0.21
base:       harbor-framework/harbor v0.21.0
base SHA:   64afbbcb62165950301e1a6407c729aa26d844ff
patch SHA:  8527f65
```

## 当前实验真正使用的改动

- Codex `web_search` 模式：Harbor 0.21 已原生支持，不再维护 `0005` 的重复实现；严格离线实验使用 `--ak web_search=disabled`。
- instruction 镜像：维护分支把每个 Agent phase 的有效题目写到 `/opt/task/instruction.md`，并设置 `HARBOR_INSTRUCTION_PATH`。`0006-agent-instruction-mirror.patch` 是该改动的离线恢复副本。

可复现实验安装固定 commit（分支继续用于后续开发）：

```bash
python -m pip install --no-deps --force-reinstall \
  'git+https://github.com/Rockyeast/harbor.git@8527f653aa866bd56ee5f96d2e93a0ff72dee01a'
```

## 历史补丁

`0001`--`0004` 是 Catalyst/Singularity 环境的历史开发补丁。当前 Modal split-GPU 流程不使用它们，也没有把它们合入上述维护分支。

`0005` 是 Harbor 旧版本缺少 Codex Web Search 配置时的兼容补丁；在 0.21 上已过时，不应再次应用。

不要再把 `0001`--`0006` 当作一条需要逐层应用的当前安装链。Singularity 工作如需恢复，应另开分支，以对应版本重新测试。

## 已完成验证

- instruction 镜像与 Harbor 0.21 Codex 配置相关测试：`68 passed`；
- Ruff lint 通过；
- `git diff --check` 通过。

这些是源码级单元测试结果，不等同于完整 Docker/Modal 端到端实验。
