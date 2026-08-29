// Trusted author-side source for the fixed performance Reference PTX.
// It is compiled outside the untrusted Direct-PTX Agent environment.
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(128)
void flash_attn_fwd_bf16_s128_d128(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* output,
    uint32_t num_heads) {
  if (num_heads == 0) return;

  const uint32_t block = blockIdx.x;
  const uint32_t query_pos = block / num_heads;
  const uint32_t head = block - query_pos * num_heads;
  if (query_pos >= 128 || head >= num_heads) return;

  const uint32_t tid = threadIdx.x;
  __shared__ float weights[128];

  float score = -INFINITY;
  if (tid <= query_pos) {
    float dot = 0.0f;
    const uint64_t q_base =
        (static_cast<uint64_t>(query_pos) * num_heads + head) * 128;
    const uint64_t k_base =
        (static_cast<uint64_t>(tid) * num_heads + head) * 128;
#pragma unroll 4
    for (uint32_t d = 0; d < 128; ++d) {
      dot = fmaf(__bfloat162float(q[q_base + d]),
                 __bfloat162float(k[k_base + d]), dot);
    }
    score = dot * 0.08838834764831845f;
  }
  weights[tid] = score;
  __syncthreads();

  if (tid == 0) {
    float maximum = -INFINITY;
    for (uint32_t j = 0; j <= query_pos; ++j) {
      maximum = fmaxf(maximum, weights[j]);
    }
    float denominator = 0.0f;
    for (uint32_t j = 0; j <= query_pos; ++j) {
      weights[j] = expf(weights[j] - maximum);
      denominator += weights[j];
    }
    for (uint32_t j = 0; j <= query_pos; ++j) {
      weights[j] /= denominator;
    }
    for (uint32_t j = query_pos + 1; j < 128; ++j) {
      weights[j] = 0.0f;
    }
  }
  __syncthreads();

  float accumulator = 0.0f;
  for (uint32_t key_pos = 0; key_pos <= query_pos; ++key_pos) {
    const uint64_t v_index =
        (static_cast<uint64_t>(key_pos) * num_heads + head) * 128 + tid;
    accumulator = fmaf(weights[key_pos], __bfloat162float(v[v_index]), accumulator);
  }
  const uint64_t output_index =
      (static_cast<uint64_t>(query_pos) * num_heads + head) * 128 + tid;
  output[output_index] = __float2bfloat16_rn(accumulator);
}
