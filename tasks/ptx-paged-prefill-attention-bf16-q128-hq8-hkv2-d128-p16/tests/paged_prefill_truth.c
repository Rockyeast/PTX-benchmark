#include <math.h>
#include <stdint.h>

#define QLEN 128
#define HQ 8
#define HKV 2
#define D 128
#define PAGE 16

static float bf16_to_float(uint16_t value) {
  union { uint32_t u; float f; } bits;
  bits.u = ((uint32_t)value) << 16;
  return bits.f;
}

static uint16_t float_to_bf16(float value) {
  union { uint32_t u; float f; } bits;
  bits.f = value;
  if ((bits.u & 0x7f800000u) == 0x7f800000u && (bits.u & 0x007fffffu)) return 0x7fc0u;
  return (uint16_t)((bits.u + 0x7fffu + ((bits.u >> 16) & 1u)) >> 16);
}

int paged_prefill_truth(const uint16_t* q, const uint16_t* k_cache,
                        const uint16_t* v_cache, const int32_t* page_table,
                        uint16_t* output, uint32_t context_length) {
  if (!q || !k_cache || !v_cache || !page_table || !output) return 1;
  if (context_length < QLEN || context_length > 1024 || context_length % PAGE) return 2;
  const float scale = 0.08838834764831845f;
  for (int qi = 0; qi < QLEN; ++qi) {
    const int count = (int)context_length - QLEN + qi + 1;
    for (int hq = 0; hq < HQ; ++hq) {
      const int hkv = hq / (HQ / HKV);
      float weights[1024];
      float maximum = -INFINITY;
      for (int logical = 0; logical < count; ++logical) {
        const int physical = page_table[logical / PAGE];
        const int cache_token = physical * PAGE + logical % PAGE;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
          dot = fmaf(bf16_to_float(q[(qi * HQ + hq) * D + d]),
                     bf16_to_float(k_cache[(cache_token * HKV + hkv) * D + d]), dot);
        }
        weights[logical] = dot * scale;
        maximum = fmaxf(maximum, weights[logical]);
      }
      float denominator = 0.0f;
      for (int logical = 0; logical < count; ++logical) {
        weights[logical] = expf(weights[logical] - maximum);
        denominator += weights[logical];
      }
      for (int d = 0; d < D; ++d) {
        float accumulator = 0.0f;
        for (int logical = 0; logical < count; ++logical) {
          const int physical = page_table[logical / PAGE];
          const int cache_token = physical * PAGE + logical % PAGE;
          accumulator = fmaf(weights[logical] / denominator,
                             bf16_to_float(v_cache[(cache_token * HKV + hkv) * D + d]),
                             accumulator);
        }
        output[(qi * HQ + hq) * D + d] = float_to_bf16(accumulator);
      }
    }
  }
  return 0;
}
