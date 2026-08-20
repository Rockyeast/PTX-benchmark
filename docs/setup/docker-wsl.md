# Docker Desktop 与 Ubuntu WSL2 检查记录

日期：2026-08-19

## 结论

Docker Desktop 对 `Ubuntu` 的 WSL2 integration 已正常工作。最初失败的直接原因是 Docker Desktop 未启动，不是 integration 配置缺失。

## 修复前

```text
Ubuntu          Running  WSL 2
docker-desktop  Stopped  WSL 2
com.docker.service: Stopped
```

Docker Desktop 的目标设置实际已经正确：

```text
EnableIntegrationWithDefaultWslDistro=true
IntegratedWslDistros="Ubuntu"
```

## 执行的修复

从 Windows 启动：

```text
C:\Program Files\Docker\Docker\Docker Desktop.exe
```

等待 Docker Desktop 状态由 `starting` 变为 `running`。

## 验证

在新的 Ubuntu WSL 进程中执行 `docker version` 和 `docker info`，确认：

```text
Docker Desktop 4.46.0
Client 28.4.0
Server 28.4.0
Linux engine
overlayfs
16 CPUs
```

随后执行：

```bash
docker run --rm hello-world
```

镜像拉取和容器执行成功，输出 `Hello from Docker!`。

## Codex 工具环境说明

普通 Ubuntu WSL shell 可以直接使用 Docker。当前 Codex 的受控命令沙箱会限制宿主 Docker socket，因此沙箱内直接运行 `docker` 可能显示 `permission denied`。这不代表机器配置再次损坏；后续自动化验证应通过获批的宿主 WSL 进程执行。
