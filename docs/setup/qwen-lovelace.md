# Lovelace Qwen 服务

本页记录 PTX-Bench 原型阶段使用的自托管模型服务。它不属于 Harbor task
environment：模型常驻 Lovelace 的一张 L40，Harbor Agent 通过 OpenAI-compatible
HTTP API 调用它。

## 固定配置

| 项目 | 值 |
|---|---|
| 模型 | `Qwen/Qwen3.8-27B-FP8` |
| Hugging Face revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| vLLM image | `vllm/vllm-openai:v0.27.1` |
| vLLM image digest | `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| Served model name | `qwen3.8-27b-fp8` |
| GPU | Lovelace GPU 2（启动时必须重新检查） |
| Context | 131072 tokens |
| Concurrent sequences | 1 |
| Default thinking mode | disabled |
| Host port | 8022 |
| Tool parser | `qwen3_xml` |
| Reasoning parser | `qwen3` |

选择 `--language-model-only`，因为 PTX Agent 只处理文本；不加载视觉模块可以给
权重和 KV cache 留出更多显存。模型缓存位于 Docker named volume
`ptx-bench-qwen38-hf-cache`，vLLM 编译缓存位于
`ptx-bench-qwen38-vllm-cache`。停止服务会释放显存，但不会删除下载的模型或
编译缓存。

最初的 32768 context 配置在 API smoke 中被否决：Qwen Code 0.21.15 默认请求
`max_tokens=64000`。单卡实测的 KV cache 容量为 142722 tokens，因此将服务提高到
131072 context，同时保持单并发；这既满足 Agent 请求，也没有使用模型声明之外的
RoPE scaling。

这是原型配置，不是已经冻结的 benchmark setting。正式模型比较前仍需决定统一的
context、Agent scaffold、采样参数、时间预算和量化策略。

服务默认通过 vLLM 的 `--default-chat-template-kwargs` 设置
`enable_thinking=false`。这只控制 Qwen 的请求级推理模式，不修改 Harbor task 的
`instruction.md`；客户端仍可在请求中显式覆盖该默认值。该设置用于避免简单 PTX
校准任务在首次工具调用前消耗完整 Agent 时间预算。

## 管理服务

仓库脚本：

```bash
scripts/lovelace/qwen38-vllm.sh start
scripts/lovelace/qwen38-vllm.sh status
scripts/lovelace/qwen38-vllm.sh health
scripts/lovelace/qwen38-vllm.sh logs
scripts/lovelace/qwen38-vllm.sh stop
```

脚本第一次启动时会：

1. 确认目标 GPU 没有计算进程且显存占用不超过 1 GiB；
2. 确认 8022 端口空闲；
3. 拉取固定版本 vLLM image；
4. 创建持久 Docker volume 并下载固定 revision 的模型；
5. 启动不启用 API 鉴权的 OpenAI-compatible API。

该原型服务不设置 API key。任何能够访问 Lovelace 8022 端口的用户都可以调用它，
因此仍需依靠 CMU 内网/VPN 边界和团队资源协调，避免并发请求影响实验。

## Harbor 接口

服务通过健康检查以后，Harbor 原型配置为：

```text
agent: qwen-coder
model: openai/qwen3.8-27b-fp8
OPENAI_BASE_URL: http://lovelace.ece.local.cmu.edu:8022/v1
OPENAI_API_KEY: unused
```

`qwen-coder` 客户端要求 `OPENAI_API_KEY` 环境变量存在，因此使用固定占位值
`unused`；vLLM 不验证它，它不是密码。

先运行 `tasks/harbor-smoke/`。该运行只证明“Harbor -> Qwen Agent -> task output ->
verifier”链路可用，不产生正式 PTX benchmark 分数。

## 2026-08-21 验证结果

部署验证使用 Lovelace GPU 2（NVIDIA L40）：

- Qwen checkpoint 大小为 28.75 GiB；加载权重实际占用 27.64 GiB；
- 131072 context、单并发运行后总显存约 41123 MiB，剩余约 4336 MiB；
- `/v1/models` 健康检查返回 `max_model_len=131072`；
- 最小文本请求返回 `QWEN_OK`；
- 受控工具请求被解析为 `report_status({"status":"ready"})`；
- Qwen Code 版本为 0.21.15；
- Harbor job `jobs/harbor-smoke-qwen38-131k/` 完成，reward 为 1.0，异常数为 0。

成功 trajectory 中，Qwen 依次调用 `write_file` 创建 `/app/answer.txt`，再调用
`run_shell_command` 通过 `od -c` 检查换行符，最后由 Harbor verifier 通过两项测试。

保留 `jobs/harbor-smoke-qwen38/` 作为失败校准记录：其 32768 context 在首个 API
请求处被 Qwen Code 的 `max_tokens=64000` 拒绝，没有执行任何 Agent 工具。

`jobs/ptx-add-one-l40-qwen38-no-thinking-20260821/` 验证了关闭 thinking 的实际
效果：第一次模型响应包含 461 个输出 token、0 个 thought token，并成功调用
`write_file` 创建 35 行的 `/app/kernel.ptx`。但后续模型响应未及时结束，Agent 在
600 秒处超时；保留的 PTX 又在 verifier 的 `cuModuleLoad` 阶段返回 CUDA error
218，因此 reward 为 0。这说明下一项服务校准应限制每次 completion 的最大输出，
而不是先增加 TP 卡数。

后续 `jobs/ptx-add-one-l40-qwen38-no-thinking-max1024-20260821/` 将
`QWEN_CODE_MAX_OUTPUT_TOKENS` 设为 1024。实际九次响应最长只有 461 tokens，说明
单次输出长度不是该次超时的原因。Agent 反而通过 `apt-get` 安装了 CUDA toolkit，
约耗时 197 秒，随后才用 `ptxas` 发现候选 PTX 的寄存器与 predicate 错误，并在修复
前达到 600 秒上限。该证据说明下一步不是 TP=2，而是限制正式 Agent 阶段只能访问
模型 API，并在任务 contract 中禁止下载或安装编译工具链。

采用该限制后的
`jobs/ptx-add-one-l40-qwen38-allowlist-max1024-20260821/` 实际显示 Harbor
egress sidecar 处于 controlled 模式，allowlist 只有 Lovelace。Agent 没有再尝试安装
软件，并在约 355 秒后正常结束，证明单卡服务和 600 秒预算足以完成本次交付流程。
但最终 PTX 将 32 位 `%r4` 用作 `shl.b64` 的源操作数；作者侧 `ptxas` 精确报告该
指令参数不匹配，verifier 因此仍在 `cuModuleLoad` 返回 CUDA error 218，reward 为
0。当前剩余问题已经从基础设施超时收敛为模型的 PTX 类型正确性，而不是 TP 卡数。

随后 `jobs/ptx-add-one-l40-qwen38-direct-v0-20260821/` 对齐完整
Direct-PTX v0 环境：保留 `ptxas`/`cuobjdump`，删除 NVCC、NVRTC、NVVM、
Triton，并提供固定只读的 PTX ISA 9.2 离线文档。相同环境下 Oracle 得到 reward 1，
证明环境与隐藏 verifier 正常。真实 Qwen Agent 没有创建高层源码或安装工具，但直到
约第 507 秒才写出第一版 PTX，第 560 秒才调用 `ptxas`，最终在 600 秒达到
`AgentTimeoutError`。该次累计输入 337818 tokens、输出 2428 tokens，单次输出最长
731 tokens，没有触及 1024 上限。保留候选存在多处汇编和地址计算错误，无法通过
`ptxas`，`cuModuleLoad` 返回 CUDA error 218，reward 为 0。它被记录为正式的
Qwen-600s 有效失败样本，不据此单独冻结任务难度，也不针对错误修改 instruction。

## 资源边界

- Lovelace 是共享服务器，空闲 GPU 快照不代表长期分配；启动前应告知其他使用者。
- 单卡 OOM 时优先改为两张卡 `TP=2`，不要把 Agent context 压缩到失去代表性。
- L40 用于 task 定义与 example；录音中的主实验硬件是 H200。
- 模型服务 GPU 与 PTX verifier GPU 可以位于不同服务器。
