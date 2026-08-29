// Performance reference adapted to PTX-Bench's ABI from NVIDIA CUDA Samples'
// vectorAdd example. This source is hidden from the evaluated agent and is
// precompiled and pinned as reference.ptx during task authoring.

extern "C" __global__ void vector_add(
    const float* a,
    const float* b,
    float* output,
    unsigned int n) {
    const unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        output[i] = a[i] + b[i];
    }
}
