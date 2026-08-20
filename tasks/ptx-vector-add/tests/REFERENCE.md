# Performance reference provenance

`reference.cu` adapts the elementary vector-add kernel pattern from NVIDIA's
CUDA Samples `vectorAdd` example to this task's fixed `vector_add` ABI.

Upstream:

- Repository: <https://github.com/NVIDIA/cuda-samples>
- Sample index: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/README.md#vectoradd>
- Upstream license: <https://github.com/NVIDIA/cuda-samples/blob/master/LICENSE>

The verifier compiles this CUDA source with NVIDIA `nvcc` before loading either
implementation. Compilation and JIT time are excluded from performance timing.
The compiled reference is a performance baseline only. Correctness is decided
independently from the mathematical definition using CPU `float32` results.
