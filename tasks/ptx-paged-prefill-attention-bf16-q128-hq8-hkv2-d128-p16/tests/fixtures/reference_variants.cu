#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

// Author-only negative fixtures selected with IGNORE_PAGE_TABLE or NON_CAUSAL.
extern "C" __global__ __launch_bounds__(128)
void paged_prefill_attn_bf16_q128_hq8_hkv2_d128_p16(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, const int32_t* page_table,
    __nv_bfloat16* output, uint32_t context_length) {
  constexpr int QLEN = 128, HQ = 8, HKV = 2, D = 128, PAGE = 16;
  constexpr float scale = 0.08838834764831845f;
  const int qi = blockIdx.x / HQ, hq = blockIdx.x % HQ, tid = threadIdx.x;
  const int hkv = hq / (HQ / HKV);
#if defined(NON_CAUSAL)
  const int count = static_cast<int>(context_length);
#else
  const int count = static_cast<int>(context_length) - QLEN + qi + 1;
#endif
  __shared__ float weights[1024];
  for (int logical = tid; logical < count; logical += 128) {
#if defined(IGNORE_PAGE_TABLE)
    const int physical = logical / PAGE;
#else
    const int physical = page_table[logical / PAGE];
#endif
    const int cache_token = physical * PAGE + logical % PAGE;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
      dot = fmaf(__bfloat162float(q[(qi * HQ + hq) * D + d]),
                 __bfloat162float(k_cache[(cache_token * HKV + hkv) * D + d]), dot);
    }
    weights[logical] = dot * scale;
  }
  __syncthreads();
  if (tid == 0) {
    float maximum = -INFINITY, denominator = 0.0f;
    for (int logical = 0; logical < count; ++logical) maximum = fmaxf(maximum, weights[logical]);
    for (int logical = 0; logical < count; ++logical) {
      weights[logical] = expf(weights[logical] - maximum);
      denominator += weights[logical];
    }
    for (int logical = 0; logical < count; ++logical) weights[logical] /= denominator;
  }
  __syncthreads();
  float accumulator = 0.0f;
  for (int logical = 0; logical < count; ++logical) {
#if defined(IGNORE_PAGE_TABLE)
    const int physical = logical / PAGE;
#else
    const int physical = page_table[logical / PAGE];
#endif
    const int cache_token = physical * PAGE + logical % PAGE;
    accumulator = fmaf(weights[logical],
        __bfloat162float(v_cache[(cache_token * HKV + hkv) * D + tid]), accumulator);
  }
  output[(qi * HQ + hq) * D + tid] = __float2bfloat16_rn(accumulator);
}
