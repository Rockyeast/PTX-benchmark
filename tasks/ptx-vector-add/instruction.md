# Implement a PTX vector-add kernel

Create the file `/app/kernel.ptx` containing NVIDIA PTX assembly.

Write or edit the PTX directly. Do not generate the submission by compiling
CUDA C++, Triton, Python, or another higher-level language. `ptxas` may be used
to validate PTX syntax, but `nvcc` and NVRTC are not available.

The file must define a visible kernel entry named `vector_add` with this logical
signature:

```text
vector_add(const float* a, const float* b, float* output, uint32_t n)
```

For every index `i` where `0 <= i < n`, the kernel must write:

```text
output[i] = a[i] + b[i]
```

The input and output arrays contain 32-bit floating-point values in global
memory. The input arrays must not be modified, and the kernel must not access
elements outside the range `[0, n)`.

The verifier launches the kernel with the following fixed configuration:

```text
block dimensions: (256, 1, 1)
grid dimensions:  (max(1, ceil(n / 256)), 1, 1)
dynamic shared memory: 0 bytes
```

The kernel must handle every valid `uint32_t n`, including `n = 0` and sizes
that are not multiples of the block size.

The verifier will load `/app/kernel.ptx` with the CUDA Driver API and execute it
on an NVIDIA B200 GPU. The submission must be valid PTX for that device;
targeting `sm_100` is recommended.

Correctness is required before performance is measured. Kernel execution time
will be compared under the same launch configuration, so avoid unnecessary
work and global-memory traffic.

Only `/app/kernel.ptx` is graded. Do not submit CUDA source, a cubin, a host
program, or a separate launch configuration.
