#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(128)
void paged_prefill_attn_bf16_q128_hq8_hkv2_d128_p16(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, const int32_t* page_table,
    __nv_bfloat16* output, uint32_t context_length) {
  constexpr int QLEN = 128;
  constexpr int HQ = 8;
  constexpr int HKV = 2;
  constexpr int D = 128;
  constexpr int PAGE = 16;
  constexpr float scale = 0.08838834764831845f;
  const int block = static_cast<int>(blockIdx.x);
  const int qi = block / HQ;
  const int hq = block - qi * HQ;
  const int tid = static_cast<int>(threadIdx.x);
  if (qi >= QLEN || context_length < QLEN || context_length > 1024) return;
  const int hkv = hq / (HQ / HKV);
  const int count = static_cast<int>(context_length) - QLEN + qi + 1;
  __shared__ float weights[1024];

  float local_max = -INFINITY;
  for (int logical = tid; logical < count; logical += 128) {
    const int physical = page_table[logical / PAGE];
    const int cache_token = physical * PAGE + logical % PAGE;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
      const uint64_t q_index = (static_cast<uint64_t>(qi) * HQ + hq) * D + d;
      const uint64_t k_index = (static_cast<uint64_t>(cache_token) * HKV + hkv) * D + d;
      dot = fmaf(__bfloat162float(q[q_index]), __bfloat162float(k_cache[k_index]), dot);
    }
    const float score = dot * scale;
    weights[logical] = score;
    local_max = fmaxf(local_max, score);
  }
  __syncthreads();

  if (tid == 0) {
    float maximum = -INFINITY;
    for (int logical = 0; logical < count; ++logical) maximum = fmaxf(maximum, weights[logical]);
    float denominator = 0.0f;
    for (int logical = 0; logical < count; ++logical) {
      weights[logical] = expf(weights[logical] - maximum);
      denominator += weights[logical];
    }
    for (int logical = 0; logical < count; ++logical) weights[logical] /= denominator;
  }
  __syncthreads();

  float accumulator = 0.0f;
  for (int logical = 0; logical < count; ++logical) {
    const int physical = page_table[logical / PAGE];
    const int cache_token = physical * PAGE + logical % PAGE;
    const uint64_t v_index = (static_cast<uint64_t>(cache_token) * HKV + hkv) * D + tid;
    accumulator = fmaf(weights[logical], __bfloat162float(v_cache[v_index]), accumulator);
  }
  const uint64_t output_index = (static_cast<uint64_t>(qi) * HQ + hq) * D + tid;
  output[output_index] = __float2bfloat16_rn(accumulator);
}
