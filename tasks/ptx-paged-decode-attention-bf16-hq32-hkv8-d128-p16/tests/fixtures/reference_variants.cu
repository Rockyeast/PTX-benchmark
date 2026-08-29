// Trusted author-only verifier fixtures. Define exactly one FIXTURE_* macro.
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(128)
void paged_decode_fwd_bf16_hq32_hkv8_d128_p16(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, const uint32_t* page_table,
    __nv_bfloat16* output, uint32_t kv_len) {
  const uint32_t hq=blockIdx.x, tid=threadIdx.x;
  if (hq>=32) return;
  if (kv_len==0) { output[hq*128+tid]=__float2bfloat16_rn(0.0f); return; }
  const uint32_t hk=
#if defined(FIXTURE_WRONG_GQA)
      hq & 7;
#else
      hq >> 2;
#endif
  const uint32_t length=
#if defined(FIXTURE_ROUND_LAST_PAGE)
      (kv_len+15)&~15u;
#else
      kv_len;
#endif

#if defined(FIXTURE_SLOW_CORRECT)
  if (tid!=0) return;
  float scores[4096]; float maximum=-INFINITY;
  for (uint32_t token=0;token<length;++token) {
    const uint32_t physical=page_table[token>>4], offset=token&15;
    const uint64_t base=(((uint64_t)physical*16+offset)*8+hk)*128;
    float dot=0.0f;
    for (uint32_t d=0;d<128;++d)
      dot=fmaf(__bfloat162float(q[hq*128+d]),__bfloat162float(k_cache[base+d]),dot);
    scores[token]=dot*0.08838834764831845f; maximum=fmaxf(maximum,scores[token]);
  }
  float denominator=0.0f;
  for (uint32_t token=0;token<length;++token) { scores[token]=expf(scores[token]-maximum); denominator+=scores[token]; }
  for (uint32_t d=0;d<128;++d) {
    float accumulator=0.0f;
    for (uint32_t token=0;token<length;++token) {
      const uint32_t physical=page_table[token>>4], offset=token&15;
      const uint64_t index=(((uint64_t)physical*16+offset)*8+hk)*128+d;
      accumulator=fmaf(scores[token]/denominator,__bfloat162float(v_cache[index]),accumulator);
    }
    output[hq*128+d]=__float2bfloat16_rn(accumulator);
  }
#else
  __shared__ float weights[4096];
  for (uint32_t token=tid;token<length;token+=128) {
    const uint32_t physical=
#if defined(FIXTURE_CONTIGUOUS_PAGES)
        token >> 4;
#else
        page_table[token >> 4];
#endif
    const uint32_t offset=token&15;
    const uint64_t base=(((uint64_t)physical*16+offset)*8+hk)*128;
    float dot=0.0f;
    for (uint32_t d=0;d<128;++d)
      dot=fmaf(__bfloat162float(q[hq*128+d]),__bfloat162float(k_cache[base+d]),dot);
    weights[token]=dot*0.08838834764831845f;
  }
  __syncthreads();
  if (tid==0) {
    float maximum=-INFINITY;
    for (uint32_t token=0;token<length;++token) maximum=fmaxf(maximum,weights[token]);
    float denominator=0.0f;
    for (uint32_t token=0;token<length;++token) { weights[token]=expf(weights[token]-maximum); denominator+=weights[token]; }
    for (uint32_t token=0;token<length;++token) weights[token]/=denominator;
  }
  __syncthreads();
  float accumulator=0.0f;
  for (uint32_t token=0;token<length;++token) {
    const uint32_t physical=
#if defined(FIXTURE_CONTIGUOUS_PAGES)
        token >> 4;
#else
        page_table[token >> 4];
#endif
    const uint32_t offset=token&15;
    const uint64_t index=(((uint64_t)physical*16+offset)*8+hk)*128+tid;
    accumulator=fmaf(weights[token],__bfloat162float(v_cache[index]),accumulator);
  }
  output[hq*128+tid]=__float2bfloat16_rn(accumulator);
#endif
}
