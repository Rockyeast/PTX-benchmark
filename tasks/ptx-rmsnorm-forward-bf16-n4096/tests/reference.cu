// Simple same-ABI performance reference for the first feasibility run.
// This is not copied from INT21 and is not intended to represent its optimized
// B200 performance. See REFERENCE.md.

#include <cuda_bf16.h>

extern "C" __global__ void rmsnorm_fwd_bf16_n4096(
    const __nv_bfloat16* x,
    const float* weight,
    __nv_bfloat16* output,
    unsigned int M,
    float eps) {
  constexpr int N = 4096;
  __shared__ float warp_sums[4];

  const unsigned int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= M) {
    return;
  }

  const unsigned long long row_offset =
      static_cast<unsigned long long>(row) * N;
  float sum_sq = 0.0f;
  for (int col = tid; col < N; col += blockDim.x) {
    const float value = __bfloat162float(x[row_offset + col]);
    sum_sq = fmaf(value, value, sum_sq);
  }

  for (int delta = 16; delta > 0; delta >>= 1) {
    sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, delta);
  }
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) {
    warp_sums[warp] = sum_sq;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = lane < 4 ? warp_sums[lane] : 0.0f;
    for (int delta = 16; delta > 0; delta >>= 1) {
      block_sum += __shfl_down_sync(0xffffffffu, block_sum, delta);
    }
    if (lane == 0) {
      warp_sums[0] = block_sum;
    }
  }
  __syncthreads();

  const float rstd = rsqrtf(warp_sums[0] * (1.0f / float(N)) + eps);
  for (int col = tid; col < N; col += blockDim.x) {
    const float value = __bfloat162float(x[row_offset + col]);
    output[row_offset + col] =
        __float2bfloat16_rn(value * rstd * weight[col]);
  }
}
