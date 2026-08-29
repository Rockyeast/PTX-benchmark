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
  const uint32_t exponent = bits & 0x7f800000u;
  const uint32_t mantissa = bits & 0x007fffffu;
  if (exponent == 0x7f800000u && mantissa != 0) return 0x7fc0u;
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

// Independent CPU truth. Returns nonzero only for an invalid host argument.
int attention_truth(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                    uint16_t* output, uint32_t num_heads) {
  if ((!q || !k || !v || !output) && num_heads != 0) return 1;
  const double scale = 0.0883883476483184405501055452631;
  for (uint32_t i = 0; i < 128; ++i) {
    for (uint32_t h = 0; h < num_heads; ++h) {
      double scores[128];
      double maximum = -INFINITY;
      for (uint32_t j = 0; j <= i; ++j) {
        double dot = 0.0;
        const uint64_t q_base = ((uint64_t)i * num_heads + h) * 128;
        const uint64_t k_base = ((uint64_t)j * num_heads + h) * 128;
        for (uint32_t d = 0; d < 128; ++d) {
          dot += (double)bf16_to_float(q[q_base + d]) *
                 (double)bf16_to_float(k[k_base + d]);
        }
        scores[j] = dot * scale;
        if (scores[j] > maximum) maximum = scores[j];
      }
      double denominator = 0.0;
      for (uint32_t j = 0; j <= i; ++j) {
        scores[j] = exp(scores[j] - maximum);
        denominator += scores[j];
      }
      const uint64_t out_base = ((uint64_t)i * num_heads + h) * 128;
      for (uint32_t d = 0; d < 128; ++d) {
        double accumulator = 0.0;
        for (uint32_t j = 0; j <= i; ++j) {
          const uint64_t v_index = ((uint64_t)j * num_heads + h) * 128 + d;
          accumulator += (scores[j] / denominator) *
                         (double)bf16_to_float(v[v_index]);
        }
        output[out_base + d] = float_to_bf16((float)accumulator);
      }
    }
  }
  return 0;
}
