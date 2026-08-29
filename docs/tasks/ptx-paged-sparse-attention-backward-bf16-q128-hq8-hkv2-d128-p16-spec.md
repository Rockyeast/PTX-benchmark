# Paged sparse attention backward BF16 Q128/HQ8/HKV2/D128/P16 规格

状态：correctness upper-probe v1，尚未 release。

## 目标

评测 Agent 能否在一份 Direct-PTX module 中实现两阶段 sparse attention
backward：先完整清零梯度，再对 paged KV、CSR 稀疏页、causal mask 和 GQA head
sharing 计算 FP32 `dQ/dK/dV`。

本题不是把三道旧题机械合并。两个 entry 对应真实的跨 block 梯度生命周期：先清零
共享 KV 梯度，主 kernel 再用 FP32 atomic 累加多个 query-head 的贡献。

## 固定问题

- `Q_LEN=128, Hq=8, Hkv=2, D=128`；
- 32 个逻辑页、32 个物理页、page size 16；
- query block size 16，共 8 个 CSR row，每行 2--8 个逻辑页；
- query 是全局 token 384--511，并施加 causal mask；
- 前向 `output` 为 BF16，`LSE` 为自然对数 FP32；
- `dQ/dK/dV` 全部为 FP32，未被选中的物理 cache 位置梯度必须为零。

## 两个 launch ABI

```text
zero: grid=(3072,1,1), block=(128,1,1), shared=0
main: grid=(1024,1,1), block=(128,1,1), shared=0
```

完整参数顺序和公式以 task 的 `instruction.md` 为唯一 Agent-facing contract。

## 真值与隐藏检查

- 独立 C truth 自行执行 paged gather、稀疏 causal forward 和 backward；
- `output/LSE` 由 C truth 生成，不依赖 CUDA Reference；
- hidden cases 改变 page permutation、CSR、数值和 dominant logits；
- 检查两个 symbol、全部三份梯度、输入不可变、allocation guard 和 CUDA error；
- 容差：`atol=3e-2, rtol=5e-2`；
- 当前只计 binary correctness：全过得 1，否则 0。

三个安全负例分别忽略 page table、只写 dQ、缺少 main entry。只有 Oracle 通过且三者
均被拒绝后，任务才进入 Agent pilot。

## 来源边界

FlashInfer `103fcf862b902e100775ac781a0de36d067de3a4` 提供 paged NHD cache、GQA、
output/LSE 和 block-sparse schema 的语义依据。它不提供本题的 exact backward kernel；
反向公式、shape、ABI、truth 和 Oracle 均为 PTX-Benchmark 自有实现。
