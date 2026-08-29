// Author-only fixtures. Compile with exactly one FIXTURE_* macro.
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(128)
void flash_attn_fwd_bf16_s128_d128(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* output,
    uint32_t num_heads) {
  if (num_heads == 0) return;
  const uint32_t query_pos = blockIdx.x / num_heads;
  const uint32_t head = blockIdx.x - query_pos * num_heads;
  if (query_pos >= 128 || head >= num_heads) return;
  const uint32_t tid = threadIdx.x;

#if defined(FIXTURE_SLOW_CORRECT)
  if (tid != 0) return;
  float scores[128];
  float maximum = -INFINITY;
  const uint64_t q_base = ((uint64_t)query_pos * num_heads + head) * 128;
  for (uint32_t j = 0; j <= query_pos; ++j) {
    float dot = 0.0f;
    const uint64_t k_base = ((uint64_t)j * num_heads + head) * 128;
    for (uint32_t d = 0; d < 128; ++d) {
      dot = fmaf(__bfloat162float(q[q_base + d]),
                 __bfloat162float(k[k_base + d]), dot);
    }
    scores[j] = dot * 0.08838834764831845f;
    maximum = fmaxf(maximum, scores[j]);
  }
  float denominator = 0.0f;
  for (uint32_t j = 0; j <= query_pos; ++j) {
    scores[j] = expf(scores[j] - maximum);
    denominator += scores[j];
  }
  for (uint32_t d = 0; d < 128; ++d) {
    float accumulator = 0.0f;
    for (uint32_t j = 0; j <= query_pos; ++j) {
      const uint64_t index = ((uint64_t)j * num_heads + head) * 128 + d;
      accumulator = fmaf(scores[j] / denominator,
                         __bfloat162float(v[index]), accumulator);
    }
    output[q_base + d] = __float2bfloat16_rn(accumulator);
  }
#else
  __shared__ float weights[128];
  const uint32_t last_key =
#if defined(FIXTURE_NON_CAUSAL)
      127;
#else
      query_pos;
#endif
  float dot = 0.0f;
  const uint64_t q_base = ((uint64_t)query_pos * num_heads + head) * 128;
  const uint64_t k_base = ((uint64_t)tid * num_heads + head) * 128;
  for (uint32_t d = 0; d < 128; ++d) {
    dot = fmaf(__bfloat162float(q[q_base + d]),
               __bfloat162float(k[k_base + d]), dot);
  }
  weights[tid] = tid <= last_key ? dot * 0.08838834764831845f : -INFINITY;
  __syncthreads();
  if (tid == 0) {
    float maximum = -INFINITY;
    for (uint32_t j = 0; j <= last_key; ++j) maximum = fmaxf(maximum, weights[j]);
    float denominator = 0.0f;
    for (uint32_t j = 0; j <= last_key; ++j) {
      weights[j] = expf(weights[j] - maximum);
      denominator += weights[j];
    }
    for (uint32_t j = 0; j <= last_key; ++j) weights[j] /= denominator;
  }
  __syncthreads();
  float accumulator = 0.0f;
  for (uint32_t j = 0; j <= last_key; ++j) {
    const uint64_t index = ((uint64_t)j * num_heads + head) * 128 + tid;
    accumulator = fmaf(weights[j], __bfloat162float(v[index]), accumulator);
  }
#if defined(FIXTURE_PARTIAL_OUTPUT)
  if (tid < 64)
#endif
    output[q_base + tid] = __float2bfloat16_rn(accumulator);
#endif
}
