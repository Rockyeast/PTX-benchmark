// Trusted author-side feasibility implementation. Never exposed to the Agent.
#include <cuda_bf16.h>
#include <math.h>

constexpr int QLEN = 128;
constexpr int HQ = 8;
constexpr int HKV = 2;
constexpr int D = 128;
constexpr int PAGE = 16;
constexpr int PAGES = 32;
constexpr int Q_ELEMENTS = QLEN * HQ * D;
constexpr int CACHE_ELEMENTS = PAGES * PAGE * HKV * D;

extern "C" __global__ __launch_bounds__(128)
void paged_sparse_attn_bwd_zero_bf16_q128_hq8_hkv2_d128_p16(
    float* grad_q, float* grad_k_cache, float* grad_v_cache) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < Q_ELEMENTS) grad_q[index] = 0.0f;
  else if (index < Q_ELEMENTS + CACHE_ELEMENTS) grad_k_cache[index - Q_ELEMENTS] = 0.0f;
  else if (index < Q_ELEMENTS + 2 * CACHE_ELEMENTS)
    grad_v_cache[index - Q_ELEMENTS - CACHE_ELEMENTS] = 0.0f;
}

__device__ __forceinline__ int cache_index(
    const int* page_table, int logical_token, int hkv, int d) {
#ifdef IGNORE_PAGE_TABLE
  const int physical_page = logical_token / PAGE;
#else
  const int physical_page = page_table[logical_token / PAGE];
#endif
  return (((physical_page * PAGE + logical_token % PAGE) * HKV + hkv) * D + d);
}

extern "C" __global__ __launch_bounds__(128)
void paged_sparse_attn_bwd_main_bf16_q128_hq8_hkv2_d128_p16(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, const __nv_bfloat16* output,
    const __nv_bfloat16* grad_output, const float* lse,
    const int* page_table, const int* row_offsets, const int* page_indices,
    float* grad_q, float* grad_k_cache, float* grad_v_cache) {
  if (threadIdx.x != 0 || blockIdx.x >= QLEN * HQ) return;
  constexpr float scale = 0.08838834764831845f;
  const int qi = static_cast<int>(blockIdx.x) / HQ;
  const int hq = static_cast<int>(blockIdx.x) % HQ;
  const int hkv = hq / 4;
  const int row = qi / 16;
  const int global_query = 384 + qi;
  const int q_base = (qi * HQ + hq) * D;

  float delta = 0.0f;
  for (int d = 0; d < D; ++d)
    delta = fmaf(__bfloat162float(grad_output[q_base + d]),
                 __bfloat162float(output[q_base + d]), delta);

  float dq[D];
  for (int d = 0; d < D; ++d) dq[d] = 0.0f;
  for (int slot = row_offsets[row]; slot < row_offsets[row + 1]; ++slot) {
    const int logical_page = page_indices[slot];
    for (int offset = 0; offset < PAGE; ++offset) {
      const int token = logical_page * PAGE + offset;
      if (token > global_query) continue;
      float score = 0.0f;
      float d_probability = 0.0f;
      for (int d = 0; d < D; ++d) {
        const int c = cache_index(page_table, token, hkv, d);
        score = fmaf(__bfloat162float(q[q_base + d]), __bfloat162float(k_cache[c]), score);
        d_probability = fmaf(__bfloat162float(grad_output[q_base + d]),
                             __bfloat162float(v_cache[c]), d_probability);
      }
      const float probability = expf(score * scale - lse[qi * HQ + hq]);
      const float ds = probability * (d_probability - delta);
      for (int d = 0; d < D; ++d) {
        const int c = cache_index(page_table, token, hkv, d);
        dq[d] = fmaf(scale * ds, __bfloat162float(k_cache[c]), dq[d]);
#ifndef ONLY_DQ
        atomicAdd(&grad_k_cache[c], scale * ds * __bfloat162float(q[q_base + d]));
        atomicAdd(&grad_v_cache[c], probability * __bfloat162float(grad_output[q_base + d]));
#endif
      }
    }
  }
  for (int d = 0; d < D; ++d) grad_q[q_base + d] = dq[d];
}
