---
name: create-direct-ptx-task
description: Create, refine, or release-audit a Harbor task that evaluates an agent writing NVIDIA PTX directly. Use for PTX-Bench task selection, specification, scaffolding, Oracle and hidden-verifier design, negative fixtures, performance scoring, anti-cheat policy, or calibration. Do not use for generic Harbor tasks or compiler-assisted CUDA/Triton kernel tasks.
---

# Create a Direct-PTX task

Build a trustworthy measurement instrument, not merely a runnable Harbor directory. Reuse the repository's template and deterministic scripts; use judgment for the operation contract, independent truth, adversarial cases, and calibration.

## Read first

Before changing files, inspect the current Git status and the relevant portions of:

- `docs/task-authoring-template.md` for the maintained Direct-PTX environment and backend contract;
- `docs/dataset/dataset-design.md` and `docs/dataset/task-catalog.md` for task granularity, families, and current coverage;
- `docs/experiments/summary.md` and any existing `docs/tasks/<task>-validation.md` before extending an existing task;
- one current task with the same verifier shape, if one exists.

Preserve unrelated worktree changes. Distinguish proposed design, static checks, Oracle evidence, negative-fixture evidence, real GPU runs, Agent calibration, and frozen release status.

## 1. Select and specify the task

Prefer an operation used by a real ML system or maintained OSS kernel library. Calibration-only toy tasks may exist, but label them as infrastructure/calibration and exclude them from release scoring.

Define one stable task contract:

- operation and forward/backward/fusion boundary;
- dtype and physical tensor layout;
- target GPU and PTX ISA/SM;
- shape specialization and which varying shapes are verifier cases rather than new tasks;
- required PTX entries and exact parameter order/types;
- grid, block, dynamic shared memory, aliasing, and boundary behavior;
- correctness tolerances and performance objective;
- time budget and documentation/network setting.

Create or update `docs/tasks/<task>-spec.md` when the contract needs review. Do not create the Harbor task until the ABI and independent correctness definition are concrete.

## 2. Scaffold with the maintained generator

Use `scripts/scaffold_direct_ptx_task.py`; do not hand-copy an old task. Explicitly choose the Agent timeout and network mode. The generated task must remain fail-closed until its task-specific implementation is complete.

```bash
python3 scripts/scaffold_direct_ptx_task.py \
  <task-slug> \
  --title "<title>" \
  --description "<description>" \
  --entry <entry> \
  --agent-timeout <seconds> \
  --network-mode <no-network-or-allowlist>
```

For multiple required entries, scaffold once and then declare every entry in the task metadata and instruction. Do not create separate tasks merely because verifier cases vary a size parameter under the same operation and launch ABI.

## 3. Write a clean Agent contract

Replace every placeholder in `instruction.md`. Tell the solving Agent exactly what artifact and behavior to deliver, but do not reveal the Oracle, hidden inputs, Reference implementation, or task-specific negative fixtures.

Include:

- `/app/kernel.ptx` and every required visible `.entry`;
- logical tensor shapes, dtypes, layouts, formulas, and state-update semantics;
- exact launch ABI;
- observable correctness requirements and the fact that correctness gates performance;
- allowed PTX-level tools and documentation access;
- the public Candidate-only measurement command when the task has a performance objective.

Describe the contract, not a step-by-step implementation strategy.

## 4. Preserve the Direct-PTX boundary

The Agent may write PTX and use `ptxas` or `cuobjdump`. It must not compile CUDA C++, Triton, Python, NVRTC, or NVVM into the submission. Keep the pinned PTX documentation contract and policy checks shared by Docker and Singularity.

Treat these as distinct settings and never merge their results:

- Direct-PTX with online official documentation;
- Direct-PTX with only the pinned local documentation;
- compiler/tool-assisted PTX delivery.

The task files define the artifact and environment contract; the Harbor launch selects Docker or Singularity. Do not fork task semantics by backend.

## 5. Build independent truth, Reference, and Oracle

Correctness truth must be independent of both the Agent PTX and the performance Reference. Implement it in trusted verifier code using a straightforward CPU/C/Python mathematical definition with fixed seeds.

For a performance task, provide a fixed same-ABI, same-launch Reference and record its source, license, revision, build command, and hashes. A weak Reference is acceptable only for feasibility and must be documented as such; it is not release-quality performance evidence.

`solution/solve.sh` must install only the auditable Oracle artifact. The Oracle must not fetch an answer at runtime.

## 6. Make the hidden verifier difficult to fake

Keep trusted grading code under `tests/`, which Harbor uploads at verifier time. Fail closed before loading the Candidate. Check, as applicable:

- PTX size, assembly/module loading, and every required entry;
- independent numerical truth across fixed hidden cases;
- input immutability, output completeness, allocation guards/canaries, CUDA errors, and synchronization;
- state/checkpoint semantics, metadata permutations, edge shapes, and plausible implementation mistakes;
- correctness before performance;
- stable GPU timing against the same-ABI Reference, with raw measurements retained separately from capped reward.

Do not expose hidden truth, Reference, cases, or final reward through the public Candidate-only benchmark command.

## 7. Add adversarial fixtures

At minimum exercise these failure classes when meaningful:

- missing or misnamed entry;
- no-op, constant output, or partial output;
- one plausible-but-wrong semantic implementation chosen for this operation;
- out-of-bounds write or input mutation;
- a slow correct implementation for a performance task.

Each wrong-but-loadable fixture must receive reward `0` for the intended reason. A slow correct fixture must pass correctness while receiving appropriately low performance credit.

## 8. Validate before Agent calibration

Run static checks first:

```bash
python3 scripts/check_direct_ptx_task.py tasks/<task-slug>
```

Then validate on the declared target GPU:

1. untouched/missing Candidate receives `0` for the expected reason;
2. Oracle receives correctness `1` and the expected reward;
3. every negative fixture receives `0` for its intended reason;
4. timing repeats are stable enough that thresholds have margin;
5. Docker or Singularity environment checks confirm the actual GPU/SM and tool policy.

Do not describe static checker success as verifier validation. Do not describe an Oracle-only run as Agent calibration.

## 9. Calibrate and audit

Run independent attempts under a fixed task digest, model, agent harness, reasoning effort, time budget, GPU, and documentation setting. Record correctness, normal exit versus timeout, raw performance, capped reward, artifact hash, and trajectory policy evidence.

Audit trajectories for high-level source generation, compiler installation, forbidden network/tool use, verifier tampering, and genuine work on the PTX artifact.

The desired release signal is a useful spread among target models. All-pass and all-fail tasks may remain anchors or probes, but do not claim they discriminate the target model tier. Calibration on one proprietary model does not substitute for the planned open-weight model grid.

## 10. Record status and freeze deliberately

Update the task README, `docs/tasks/<task>-validation.md`, `docs/dataset/task-catalog.md`, and `docs/experiments/summary.md` with evidence-bounded status. Never overwrite or merge results from different digests or settings.

Only set `implementation_status = "implemented"` and `release_ready = true` after the Direct-PTX shipping checklist is evidenced, then run:

```bash
python3 scripts/check_direct_ptx_task.py tasks/<task-slug> --release
```

The release checker covers static conditions only. A task is not frozen until Oracle, baseline, adversarial fixtures, target-GPU repeatability, trajectory audit, and the intended multi-model calibration are documented.
