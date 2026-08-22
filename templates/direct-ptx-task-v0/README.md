# @@TITLE@@

**Status: scaffolded from `direct-ptx-v0.1`; not runnable or release-ready.**

This Harbor task targets @@HARDWARE_TARGET@@ (`sm_@@SM@@`) and requires the
Agent to submit `/app/kernel.ptx` with visible entry `@@ENTRY_NAME@@`.

The generated scaffold is intentionally fail-closed. Its Oracle exits with an
error and its verifier writes reward zero until the task author completes all
task-specific work.

Before running this task:

- replace every `TODO` in the task directory;
- pin the CUDA base image by digest;
- replace the placeholder under `environment/files/ptx-docs/` with a pinned,
  license-reviewed PTX ISA snapshot and complete its manifest;
- align the logical signature and launch ABI across instruction, Oracle, and
  verifier;
- implement an independent correctness truth and memory-safety checks;
- retain the generated hardware identity check for the target GPU and SM;
- add a fixed same-ABI performance Reference if this is a performance task;
- document source, license, compiler revision, shapes, tolerances, and reward;
- run Oracle, at least two meaningful negative fixtures, repeat timing, and a
  real Agent;
- keep `release_ready=false` until every release gate passes.

Before release, change `implementation_status` from `scaffold` to `implemented`
only after replacing the fail-closed Oracle/verifier bodies and retaining the
corresponding runtime Jobs. The static checker cannot substitute for those
runtime results.

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

## Runtime backend

The task contract is backend-neutral, but GPU selection is runtime-specific.
The task metadata records that one @@GPU_TYPE@@ GPU is required; Harbor does
not schedule that metadata by itself. The verifier independently checks the
actual CUDA device name and compute capability before grading the Candidate.

For Docker, select an idle device for the generated Compose override:

```bash
export PTX_BENCH_GPU_DEVICE=<gpu-index>
harbor run --path tasks/@@TASK_SLUG@@ --env docker ...
```

For the patched Singularity backend, request one allocated GPU and provide the
pinned SIF explicitly:

```bash
harbor run --path tasks/@@TASK_SLUG@@ \
  --env singularity \
  --override-gpus 1 \
  --environment-kwarg singularity_nv=true \
  --environment-kwarg singularity_image_path=/absolute/path/to/ptx-bench-direct-v2.sif \
  ...
```

`environment/files/` is the shared source of policy and documentation assets.
Docker copies it during image construction; the Singularity bootstrap injects
it for each trial and runs `environment/files/setup.sh`.

The environment copy of `direct-ptx-policy-check` runs during trusted setup.
Harbor uploads an identical checker and manifest from `tests/` only after the
Agent phase, and `/tests/test.sh` uses those hidden copies for grading. The
repository checker rejects drift between the public setup assets and hidden
verifier assets.
