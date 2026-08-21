# @@TITLE@@

**Status: scaffolded from `direct-ptx-v0`; not runnable or release-ready.**

This Harbor task targets @@HARDWARE_TARGET@@ (`sm_@@SM@@`) and requires the
Agent to submit `/app/kernel.ptx` with visible entry `@@ENTRY_NAME@@`.

The generated scaffold is intentionally fail-closed. Its Oracle exits with an
error and its verifier writes reward zero until the task author completes all
task-specific work.

Before running this task:

- replace every `TODO` in the task directory;
- replace the placeholder under `environment/ptx-docs/` with a pinned,
  license-reviewed PTX ISA snapshot and complete its manifest;
- align the logical signature and launch ABI across instruction, Oracle, and
  verifier;
- implement an independent correctness truth and memory-safety checks;
- add a fixed same-ABI performance Reference if this is a performance task;
- document source, license, compiler revision, shapes, tolerances, and reward;
- run Oracle, at least two meaningful negative fixtures, repeat timing, and a
  real Agent;
- keep `release_ready=false` until every release gate passes.

Run the repository checker while authoring, and use its stricter release mode
before adding the task to a frozen dataset:

```bash
python3 scripts/check_direct_ptx_task.py tasks/@@TASK_SLUG@@
python3 scripts/check_direct_ptx_task.py tasks/@@TASK_SLUG@@ --release
```

The environment follows the direct-PTX policy: `ptxas` and `cuobjdump` are
available, while `nvcc`, NVRTC, NVVM, and Triton are not provided. Every
generated task must explicitly choose its Agent-phase network policy. The
trusted environment baseline remains public for setup and pinned artifact
fetches; Harbor overrides it with the generated `allowlist` or `no-network`
policy while the untrusted Agent runs. The release checker evaluates this
effective Agent-phase policy. Use a model-API allowlist for the normal
served-model setting.
