// Fixed same-ABI performance Reference for PTX-Bench FP32 Softmax N=4096.
//
// Repository-authored CUDA adaptation informed by FlashInfer's stable online
// softmax structure at revision 103fcf862b902e100775ac781a0de36d067de3a4
// (Apache-2.0). This file is compiled only during trusted task authoring; nvcc
// is not present while an Agent solves the Direct-PTX task.

#include <cuda_runtime.h>

extern "C" __global__ __launch_bounds__(1024)
void softmax_fwd_fp32_n4096(const float* input, float* output,
                            unsigned int num_rows) {
  const unsigned int row = blockIdx.x;
  if (row >= num_rows) {
    return;
  }

  constexpr unsigned int kWidth = 4096;
  constexpr unsigned int kBlock = 1024;
  const unsigned int lane = threadIdx.x;
  const float* row_input = input + row * kWidth;
  float* row_output = output + row * kWidth;

  const float x0 = row_input[lane];
  const float x1 = row_input[lane + kBlock];
  const float x2 = row_input[lane + 2 * kBlock];
  const float x3 = row_input[lane + 3 * kBlock];

  __shared__ float scratch[kBlock];
  float local_max = fmaxf(fmaxf(x0, x1), fmaxf(x2, x3));
  scratch[lane] = local_max;
  __syncthreads();

  for (unsigned int offset = kBlock / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      scratch[lane] = fmaxf(scratch[lane], scratch[lane + offset]);
    }
    __syncthreads();
  }

  const float row_max = scratch[0];
  const float e0 = __expf(x0 - row_max);
  const float e1 = __expf(x1 - row_max);
  const float e2 = __expf(x2 - row_max);
  const float e3 = __expf(x3 - row_max);
  scratch[lane] = (e0 + e1) + (e2 + e3);
  __syncthreads();

  for (unsigned int offset = kBlock / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      scratch[lane] += scratch[lane + offset];
    }
    __syncthreads();
  }

  const float inverse_sum = 1.0f / scratch[0];
  row_output[lane] = e0 * inverse_sum;
  row_output[lane + kBlock] = e1 * inverse_sum;
  row_output[lane + 2 * kBlock] = e2 * inverse_sum;
  row_output[lane + 3 * kBlock] = e3 * inverse_sum;
}
