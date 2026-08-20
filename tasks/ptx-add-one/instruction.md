# Implement an add-one PTX kernel

Create the file `/app/kernel.ptx` containing NVIDIA PTX assembly.

The file must define a visible kernel entry named `add_one` with this logical
signature:

```text
add_one(const int32_t* input, int32_t* output, uint32_t n)
```

For every index `i` where `0 <= i < n`, the kernel must write
`output[i] = input[i] + 1`.

The verifier will load `/app/kernel.ptx` with the CUDA Driver API and execute it
on an NVIDIA B200 GPU. Only `/app/kernel.ptx` is graded.
