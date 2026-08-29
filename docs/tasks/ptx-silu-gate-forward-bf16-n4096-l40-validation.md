# BF16 SiLU-gate L40 calibration validation

This is an L40 (`sm_89`) calibration variant of the B200 pilot task
`ptx-silu-gate-forward-bf16-n4096`. It keeps the same operation, launch ABI,
correctness cases, fixed Reference, and Direct-PTX policy while changing only
the hardware specialization needed for Lovelace.

The Agent must directly author `/app/kernel.ptx` with visible entry
`silu_gate_fwd_bf16_n4096`. Correctness is a hard gate before performance.

This variant is deliberately marked `calibration_only=true` and
`release_ready=false`. It does not replace the B200 task and must not be mixed
into a frozen B200 benchmark score.

Select an idle L40 at runtime:

```bash
export PTX_BENCH_GPU_DEVICE=<gpu-index>
harbor run \
  --path tasks/ptx-silu-gate-forward-bf16-n4096-l40 \
  --env docker \
  ...
```

Run the repository checker while authoring:

```bash
python3 scripts/check_direct_ptx_task.py \
  tasks/ptx-silu-gate-forward-bf16-n4096-l40
```
