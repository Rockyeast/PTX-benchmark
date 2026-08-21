# Implement an add-one kernel in pure PTX for NVIDIA L40

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. Only this file is
graded. Do not create CUDA, Triton, or other higher-level source code.

Work directly in PTX. Do not download or install compilers, CUDA toolkits,
Python packages, documentation, or other software. `ptxas` and `cuobjdump` are
available for validating and inspecting PTX written directly by you. `nvcc`,
NVRTC, NVVM, and Triton are not available.

The NVIDIA PTX ISA 9.2 manual is available offline at the read-only path
`/opt/ptx-docs/ptx-isa-9.2.pdf`. `pdftotext` is available for searching or
reading it without network access.

The file must define a visible kernel entry named `add_one` with this logical
signature:

```text
add_one(const int32_t* input, int32_t* output, uint32_t n)
```

For every index `i` where `0 <= i < n`, the kernel must write:

```text
output[i] = input[i] + 1
```

The verifier launches multiple one-dimensional grids with 128 threads per
block. The implementation must compute a global thread index and must not read
or write outside `[0, n)`.

The verifier loads `/app/kernel.ptx` with the CUDA Driver API and executes it on
an NVIDIA L40 GPU with compute capability 8.9. Target `sm_89`.
