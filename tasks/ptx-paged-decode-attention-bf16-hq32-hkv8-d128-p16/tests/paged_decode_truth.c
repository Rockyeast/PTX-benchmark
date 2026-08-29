#include <math.h>
#include <stdint.h>
#include <string.h>

static float bf16_to_float(uint16_t value) {
  uint32_t bits = ((uint32_t)value) << 16;
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static uint16_t float_to_bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu))
    return 0x7fc0u;
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

int paged_decode_truth(const uint16_t* q, const uint16_t* k_cache,
                       const uint16_t* v_cache, const uint32_t* page_table,
                       uint16_t* output, uint32_t kv_len) {
  if (!q || !k_cache || !v_cache || !page_table || !output || kv_len > 4096)
    return 1;
  if (kv_len == 0) {
    memset(output, 0, 32 * 128 * sizeof(uint16_t));
    return 0;
  }
  const double scale = 0.0883883476483184405501055452631;
  for (uint32_t hq = 0; hq < 32; ++hq) {
    const uint32_t hk = hq / 4;
    double scores[4096];
    double maximum = -INFINITY;
    for (uint32_t token = 0; token < kv_len; ++token) {
      const uint32_t physical_page = page_table[token / 16];
      if (physical_page >= 320) return 2;
      const uint32_t page_offset = token % 16;
      const uint64_t cache_base =
          (((uint64_t)physical_page * 16 + page_offset) * 8 + hk) * 128;
      double dot = 0.0;
      for (uint32_t d = 0; d < 128; ++d)
        dot += (double)bf16_to_float(q[hq * 128 + d]) *
               (double)bf16_to_float(k_cache[cache_base + d]);
      scores[token] = dot * scale;
      if (scores[token] > maximum) maximum = scores[token];
    }
    double denominator = 0.0;
    for (uint32_t token = 0; token < kv_len; ++token) {
      scores[token] = exp(scores[token] - maximum);
      denominator += scores[token];
    }
    for (uint32_t d = 0; d < 128; ++d) {
      double accumulator = 0.0;
      for (uint32_t token = 0; token < kv_len; ++token) {
        const uint32_t physical_page = page_table[token / 16];
        const uint32_t page_offset = token % 16;
        const uint64_t index =
            (((uint64_t)physical_page * 16 + page_offset) * 8 + hk) * 128 + d;
        accumulator += (scores[token] / denominator) *
                       (double)bf16_to_float(v_cache[index]);
      }
      output[hq * 128 + d] = float_to_bf16((float)accumulator);
    }
  }
  return 0;
}
