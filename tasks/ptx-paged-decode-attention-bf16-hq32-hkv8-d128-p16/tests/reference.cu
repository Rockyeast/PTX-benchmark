// Trusted author-side source for the fixed same-ABI performance Reference.
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(128)
void paged_decode_fwd_bf16_hq32_hkv8_d128_p16(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, const uint32_t* page_table,
    __nv_bfloat16* output, uint32_t kv_len) {
  const uint32_t hq = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  if (hq >= 32) return;
  if (kv_len == 0) {
    output[hq * 128 + tid] = __float2bfloat16_rn(0.0f);
    return;
  }

  const uint32_t hk = hq >> 2;
  __shared__ float weights[4096];
  for (uint32_t token = tid; token < kv_len; token += 128) {
    const uint32_t physical_page = page_table[token >> 4];
    const uint32_t page_offset = token & 15;
    const uint64_t cache_base =
        (((uint64_t)physical_page * 16 + page_offset) * 8 + hk) * 128;
    float dot = 0.0f;
#pragma unroll 4
    for (uint32_t d = 0; d < 128; ++d) {
      dot = fmaf(__bfloat162float(q[hq * 128 + d]),
                 __bfloat162float(k_cache[cache_base + d]), dot);
    }
    weights[token] = dot * 0.08838834764831845f;
  }
  __syncthreads();

  if (tid == 0) {
    float maximum = -INFINITY;
    for (uint32_t token = 0; token < kv_len; ++token)
      maximum = fmaxf(maximum, weights[token]);
    float denominator = 0.0f;
    for (uint32_t token = 0; token < kv_len; ++token) {
      weights[token] = expf(weights[token] - maximum);
      denominator += weights[token];
    }
    for (uint32_t token = 0; token < kv_len; ++token)
      weights[token] /= denominator;
  }
  __syncthreads();

  float accumulator = 0.0f;
  for (uint32_t token = 0; token < kv_len; ++token) {
    const uint32_t physical_page = page_table[token >> 4];
    const uint32_t page_offset = token & 15;
    const uint64_t index =
        (((uint64_t)physical_page * 16 + page_offset) * 8 + hk) * 128 + tid;
    accumulator = fmaf(weights[token], __bfloat162float(v_cache[index]), accumulator);
  }
  output[hq * 128 + tid] = __float2bfloat16_rn(accumulator);
}
