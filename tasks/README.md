# ptx-bench/ptx-bench-dev

Harbor development dataset for PTX-Bench on NVIDIA B200.

This manifest is intentionally mutable and currently contains:

- `ptx-vector-add`: infrastructure/performance calibration only;
- `ptx-rmsnorm-forward-bf16-n4096`: first formal task candidate, currently
  feasibility-validated.

It was generated and synced with local Harbor 0.21.0, then successfully parsed
without modification by the remote patched Harbor 0.13.2 environment. A release
dataset must still pin one Harbor version rather than rely on cross-version
compatibility.

`harbor-smoke` and `ptx-add-one` remain in this directory because they are
useful infrastructure examples, but they are deliberately excluded from
`dataset.toml` and from any future benchmark score.

The dataset architecture and candidate backlog are documented in:

- [`../docs/dataset/dataset-design.md`](../docs/dataset/dataset-design.md)
- [`../docs/dataset/task-catalog.md`](../docs/dataset/task-catalog.md)

After a local task changes, update its pinned content digest with:

```bash
harbor sync tasks
```

Do not publish this development manifest as a frozen benchmark release.
