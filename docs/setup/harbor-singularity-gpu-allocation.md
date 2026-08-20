# Harbor Singularity 并行 GPU 分配

## 目标

同一个 Harbor 进程并行启动多个 GPU trial 时，每个
`SingularityEnvironment` 自动租用不同的物理 GPU，并让对应容器只能通过
`CUDA_VISIBLE_DEVICES` 使用自己的租约。

实现补丁：

```text
patches/harbor/0002-singularity-gpu-allocation.patch
```

该补丁应在 `0001-singularity-nvidia-workdir.patch` 之后应用。

## 分配流程

每个需要 GPU 的 trial 在启动 Singularity server 前依次执行：

1. 用 `nvidia-smi` 枚举物理 GPU 和 `memory.used`；
2. 尝试获取 `/tmp/harbor-singularity-gpu-<id>.lock` 的非阻塞文件锁；
3. 拒绝显存超过阈值的 GPU；
4. 拒绝存在计算进程的 GPU；
5. 在独立 Python 子进程中创建最小 CUDA Driver context；
6. 把通过检查的物理编号写入容器内的 `CUDA_VISIBLE_DEVICES`；
7. trial 完成、启动失败或异常退出时释放文件锁。

锁文件本身会留在 `/tmp`，但文件描述符关闭后租约已经释放。真正表示占用的是
`flock`，不是文件是否存在。

## 并行命令

下面示例对同一道题运行两个 attempt，并允许 Harbor 同时启动两个 trial：

```bash
cd /home/zhuominc/rockyeast-work/harbor-ptx

timeout 7200 .venv/bin/harbor run \
  --path /home/zhuominc/rockyeast-work/PTX-benchmark/tasks/ptx-add-one \
  --agent oracle \
  --env singularity \
  --environment-kwarg singularity_nv=true \
  --environment-kwarg singularity_gpu_ids=0,1,2,3,4,5,6,7 \
  --environment-kwarg singularity_image_path=/home/zhuominc/rockyeast-work/containers/ptx-bench-cuda13.2.1-ubuntu24.04-v1.sif \
  --n-attempts 2 \
  --n-concurrent 2 \
  --job-name ptx-add-one-parallel-2 \
  --yes
```

运行真实 Codex 时，将 `--agent oracle` 换成：

```bash
--agent codex \
--model gpt-5.6-sol \
--agent-env CODEX_FORCE_AUTH_JSON=true
```

并将 `singularity_image_path` 换成预装 Codex 的 SIF。

可选参数：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `singularity_gpu_ids` | 全部物理 GPU | 限制允许分配的物理编号 |
| `singularity_gpu_memory_threshold_mb` | `256` | 当前显存超过该值就拒绝 |
| `singularity_gpu_context_probe` | `true` | 分配前必须成功创建 CUDA context |
| `singularity_gpu_lock_dir` | `/tmp` | 保存跨 trial 文件锁的位置 |

## 当前验证

2026-08-20 已完成：

- GPU 分配器单元测试 `7 passed`；
- ruff lint 和 format 检查通过；
- 真实 Harbor Oracle trial `ptx-add-one-auto-gpu-allocator` 自动租用 GPU 3；
- verifier 输出 `PTX_VERIFY_OK: 8 outputs matched`，reward 为 `1.0`；
- trial 日志记录了 GPU 3 的租用和释放；
- 当 GPU 5 的 CUDA context 返回 error 999、GPU 3 被外部进程占用时，分配器
  安全返回 `GpuAllocationError`，没有误用这两张卡。

尚未完成两张真实 B200 同时成功的运行验证：当时共享服务器没有两张同时满足
条件的 GPU。并发的互斥分配已由单元测试覆盖，但在出现两张健康空闲卡后仍应跑
上面的 `--n-attempts 2 --n-concurrent 2` 作为运行时验收。

## 边界

这是直接 SSH 使用共享节点时的 best-effort 分配器，不是集群调度器。文件锁能
阻止 Harbor trial 互相抢卡，但实验室其他用户的程序不会遵守这个锁，所以每次
租用仍执行实时检查。外部进程可能在检查后启动；需要严格排他保证时仍应由
SLURM 等调度系统分配 GPU。
