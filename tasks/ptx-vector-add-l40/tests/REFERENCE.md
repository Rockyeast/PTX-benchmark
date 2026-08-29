# Performance reference provenance

`reference.cu` adapts the elementary vector-add kernel pattern from NVIDIA's
CUDA Samples `vectorAdd` example to this task's fixed `vector_add` ABI.

Upstream:

- Repository: <https://github.com/NVIDIA/cuda-samples>
- Sample index: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/README.md#vectoradd>
- Upstream license: <https://github.com/NVIDIA/cuda-samples/blob/master/LICENSE>

`reference.ptx` was compiled once during task authoring; the verifier loads the
pinned PTX and does not contain or invoke `nvcc`. Build provenance:

- source: `reference.cu`, SHA-256
  `f1c3645f6f2ae051f2650fe1ee9c67f8e51b880efd5338f43db136d2515e6a8f`;
- compiler: NVIDIA `nvcc` 13.2.78, NVVM 22.0.0;
- command: `nvcc --ptx --optimize=3 --gpu-architecture=compute_100`;
- artifact: `reference.ptx`, SHA-256
  `57009bdda50759fddc102f68bc46a1ea8ef1c68f4592bcedcf4aec8609268762`.

The pinned artifact is a performance baseline only. Correctness is decided
independently from the mathematical definition using CPU `float32` results.
