#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

// Negative fixture: quantizes consistently, but uses int8's 127 range instead
// of finite E4M3's required 448 range.
extern "C" __global__ __launch_bounds__(128)
void gemm_rowwise_fp8e4m3_m128_n128_k128(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    uint8_t* output_e4m3, float* row_scale) {
  constexpr int N = 128;
  const int m = blockIdx.x, n = threadIdx.x;
  __shared__ float values[N];
  __shared__ float scale;
  float accumulator = 0.0f;
  for (int k = 0; k < N; ++k) {
    accumulator = fmaf(__bfloat162float(a[m * N + k]),
                       __bfloat162float(b[n * N + k]), accumulator);
  }
  values[n] = accumulator;
  __syncthreads();
  if (n == 0) {
    float maximum = 0.0f;
    for (int column = 0; column < N; ++column) maximum = fmaxf(maximum, fabsf(values[column]));
    scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
    row_scale[m] = scale;
  }
  __syncthreads();
  reinterpret_cast<__nv_fp8_e4m3*>(output_e4m3)[m * N + n] = __nv_fp8_e4m3(values[n] / scale);
}
