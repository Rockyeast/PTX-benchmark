# PTX vector-add L40 calibration validation

This task is the NVIDIA L40 (`sm_89`) calibration variant of
`tasks/ptx-vector-add/`. It keeps the same `vector_add` ABI, correctness cases,
guard checks, performance sizes, and reward calculation while changing the
hardware target and PTX baseline from B200 to L40.

It is deliberately excluded from `tasks/dataset.toml`. Its purpose is to test
the complete local pipeline with the Qwen model service on Lovelace GPU 2 and a
separate L40 verifier GPU. Passing this task is infrastructure and Agent
calibration evidence, not a formal PTX-Bench score.

The task uses the same Direct-PTX environment policy as
`tasks/ptx-add-one-l40`: `ptxas` and `cuobjdump` are available, while `nvcc`,
NVRTC, NVVM, and Triton are removed. The Agent phase may reach only the
Lovelace model API host.

Before a local Docker run on Lovelace, set `PTX_BENCH_GPU_DEVICE` to an idle L40
index. Harbor's Docker backend reads it through
`tasks/ptx-vector-add-l40/environment/docker-compose.yaml`.
