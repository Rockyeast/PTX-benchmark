# Harbor Singularity 网络隔离

## 目标

让 Harbor 在 Singularity/Apptainer 后端中真正执行 Agent 阶段网络策略，而不是只在
`task.toml` 里记录一个没有运行效果的字段。

实现补丁：

```text
patches/harbor/0003-singularity-network-policy.patch
patches/harbor/0005-codex-web-search-policy.patch
```

所有补丁按编号顺序应用：

```text
0001-singularity-nvidia-workdir.patch
0002-singularity-gpu-allocation.patch
0003-singularity-network-policy.patch
0004-singularity-long-command-timeout.patch
0005-codex-web-search-policy.patch
```

补丁只应用于个人隔离 Harbor 副本：

```text
/home/zhuominc/rockyeast-work/harbor-ptx
```

共享的 `/home/zhuominc/harbor` 不修改。

## 为什么不能直接关闭整个容器网络

Singularity server 必须继续接收 Harbor 的本机控制请求；云端 Codex 还必须访问模型
API。如果把整个容器一次性断网，Harbor 无法切换 setup、Agent、verifier 三个阶段，
云端 Agent 也无法工作。

因此补丁按每条 Agent 命令实施策略：

```text
Harbor 控制进程（保留网络）
        │
        ├── public：按原方式执行
        │
        ├── no-network：unshare 网络命名空间，只有 loopback
        │
        └── allowlist：unshare 网络命名空间
                         │
                         └── localhost CONNECT bridge
                                  │ Unix socket
                                  ▼
                           host-side allowlist proxy
                                  │
                                  └── 只连接允许的 hostname
```

restricted 命令本身没有外部路由。因此，即使 Agent 删除代理环境变量，或者尝试用
IP 地址绕过 hostname 检查，也不能直接连接公网。

## 两种策略怎么写

完全断网的本地 Agent 或 Oracle：

```toml
[agent]
network_mode = "no-network"
```

需要访问模型 API 的云端 Agent，task 仍可默认写成 `no-network`，运行时只追加本次
必要的模型域名：

```bash
--allow-agent-host chatgpt.com \
--allow-agent-host '*.chatgpt.com'
```

Harbor 会把本次 Agent 阶段的有效策略合并成 `allowlist`。setup 阶段安装 Agent CLI
和 verifier 阶段运行隐藏测试不继承这份 Agent allowlist。

不要把 `pypi.org`、GitHub、CUDA 下载站点等开发工具域名加入正式 Direct-PTX
Agent allowlist；否则 Agent 又可能在作答阶段下载高层编译工具。

## 应用补丁

在一个干净的 Harbor 上游 revision 上依次执行：

```bash
git apply /path/to/PTX-benchmark/patches/harbor/0001-singularity-nvidia-workdir.patch
git apply /path/to/PTX-benchmark/patches/harbor/0002-singularity-gpu-allocation.patch
git apply /path/to/PTX-benchmark/patches/harbor/0003-singularity-network-policy.patch
git apply /path/to/PTX-benchmark/patches/harbor/0004-singularity-long-command-timeout.patch
git apply /path/to/PTX-benchmark/patches/harbor/0005-codex-web-search-policy.patch
```

`0004` 去掉 Singularity HTTP transport 隐藏的 600 秒默认读超时。Agent phase
仍由 Harbor 的 `task.toml [agent].timeout_sec` 和外层 `asyncio.wait_for` 控制；显式
传给单条命令的 timeout 仍保留 10 秒 transport 收尾余量。否则 3600 秒或 2--8 小时
任务会在约 600 秒被内部 HTTP 层提前截断，错误文本却仍报告外层任务预算。

第三个补丁增加：

- `network_guard.py`：网络命名空间、loopback bridge 和 hostname 代理；
- Singularity server 的逐命令网络包装；
- `no-network`、`allowlist` 和运行期动态切换能力声明；
- hostname 解析、通配符和代理拒绝路径的单元测试。

`0005` 处理另一层网络能力：Codex 自带的模型侧 Web Search。它不经过 task
容器，所以 `no-network` 或 Agent hostname allowlist 无法单独控制它。按照当前
PTX-Bench 的在线文档主 setting，个人 Harbor 副本默认把 Codex 启动为：

```text
-c web_search=live
```

这样 Agent 可以查询在线 PTX 文档，但容器命令仍受 allowlist 限制，不能下载或安装
Triton、NVCC 等高层编译工具。若运行严格离线文档对照，必须显式运行：

```bash
harbor run ... --agent codex --agent-kwarg web_search=disabled
```

两种 setting 的结果不得合并校准。每个正式 Job 还应运行轨迹审计，因为配置存在与
Agent 实际行为是两类证据：

```bash
python3 scripts/audit_direct_ptx_trajectory.py \
  --policy online-docs jobs/<job-name>

python3 scripts/audit_direct_ptx_trajectory.py \
  --policy strict-offline jobs/<job-name>
```

## 2026-08-22 验证证据

静态和单元测试：

- 网络策略与 GPU allocator 合计 `16 passed`；
- ruff lint、ruff format 和 Python bytecode compile 通过；
- 在 Harbor 上游基线依次应用三个补丁通过。

B200 运行验证：

1. standalone 探针中，allowlist 允许 `example.com:443`、拒绝
   `pypi.org:443`，直接连接 `1.1.1.1` 失败；`ptxas` 在两种 restricted 模式下
   都能运行。
2. Harbor Oracle Job
   `ptx-add-one-b200-singularity-no-network-probe-20260822` 中，直接 TCP 探针返回
   `Network is unreachable`，随后 PTX verifier 仍通过，reward 为 `1.0`。
3. Harbor Codex Job
   `ptx-add-one-b200-singularity-codex-allowlist-20260822` 中，Codex 经
   `chatgpt.com`/`ab.chatgpt.com` 完成任务；未列入 allowlist 的
   `*.oaiusercontent.com` 被代理拒绝；总运行 40 秒，reward 为 `1.0`。

这些结果证明网络策略既会阻断直接公网连接，也不会破坏 B200 PTX 执行或必要的
模型 API 通信。

## 2026-08-23 Codex Web Search 修复

- `0005` 已应用于远端个人 Harbor 副本；Codex flag descriptor 测试
  `53 passed`；
- 默认生成的 Agent 参数包含 `-c web_search=live`；严格离线文档 setting 显式覆盖为
  `disabled`；
- `scripts/audit_direct_ptx_trajectory.py` 扫描当前 46 份 trajectory：43 份没有
  模型侧 Web Search，2 份 KDA trajectory 共出现 5 次调用，另 1 份旧 Qwen
  trajectory 不是合法 JSON、无法据此完成审计；
- 这只是配置与回归测试证据。下一次真实 Agent Job 仍须再次审计 trajectory，才能
  形成端到端生效证据。

## 边界

- 该实现依赖 Linux unprivileged user/network namespaces；启动前应确认
  `unshare --user --map-root-user --net true` 可运行。
- allowlist 约束的是目标 hostname，不是“允许任意联网”。只应加入 Agent 工作所需
  的最小域名集合。
- 这是个人 Harbor 后端补丁；在正式冻结 benchmark 前，应固定 Harbor revision、
  全部补丁的内容以及运行命令。
- 容器命令网络策略与 Codex 模型侧 Web Search 是两个独立控制面，不能用其中一个的
  通过结果推断另一个也已受限。
