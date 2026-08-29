#include <math.h>
#include <stdint.h>
#include <string.h>

#define S 128
#define D 128

static float bf16_to_float(uint16_t value) {
  union { uint32_t u; float f; } bits;
  bits.u = ((uint32_t)value) << 16;
  return bits.f;
}

int attention_bwd_truth(const uint16_t* q, const uint16_t* k,
                        const uint16_t* v, const uint16_t* grad_output,
                        float* grad_q, float* grad_k, float* grad_v) {
  if (!q || !k || !v || !grad_output || !grad_q || !grad_k || !grad_v) return 1;
  const float scale = 0.08838834764831845f;
  memset(grad_q, 0, S * D * sizeof(float));
  memset(grad_k, 0, S * D * sizeof(float));
  memset(grad_v, 0, S * D * sizeof(float));

  for (int i = 0; i < S; ++i) {
    float probability[S];
    float d_probability[S];
    float row_max = -INFINITY;
    for (int j = 0; j < S; ++j) {
      float qk = 0.0f;
      float dov = 0.0f;
      for (int d = 0; d < D; ++d) {
        qk = fmaf(bf16_to_float(q[i * D + d]),
                  bf16_to_float(k[j * D + d]), qk);
        dov = fmaf(bf16_to_float(grad_output[i * D + d]),
                   bf16_to_float(v[j * D + d]), dov);
      }
      probability[j] = qk * scale;
      d_probability[j] = dov;
      row_max = fmaxf(row_max, probability[j]);
    }
    float denominator = 0.0f;
    for (int j = 0; j < S; ++j) {
      probability[j] = expf(probability[j] - row_max);
      denominator += probability[j];
    }
    float delta = 0.0f;
    for (int j = 0; j < S; ++j) {
      probability[j] /= denominator;
      delta = fmaf(probability[j], d_probability[j], delta);
    }
    for (int j = 0; j < S; ++j) {
      const float ds = probability[j] * (d_probability[j] - delta);
      for (int d = 0; d < D; ++d) {
        grad_q[i * D + d] = fmaf(scale * ds,
                                 bf16_to_float(k[j * D + d]),
                                 grad_q[i * D + d]);
        grad_k[j * D + d] = fmaf(scale * ds,
                                 bf16_to_float(q[i * D + d]),
                                 grad_k[j * D + d]);
        grad_v[j * D + d] = fmaf(probability[j],
                                 bf16_to_float(grad_output[i * D + d]),
                                 grad_v[j * D + d]);
      }
    }
  }
  return 0;
}
