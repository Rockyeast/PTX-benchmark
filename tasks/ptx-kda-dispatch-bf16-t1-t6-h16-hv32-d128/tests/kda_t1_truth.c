#include <math.h>
#include <stdint.h>
#include <string.h>

enum {
  N = 32,
  H = 16,
  HV = 32,
  D = 128,
};

static float bf16_to_float(uint16_t value) {
  uint32_t bits = ((uint32_t)value) << 16;
  float result;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static uint16_t float_to_bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000u) == 0x7f800000u &&
      (bits & 0x007fffffu))
    return 0x7fc0u;
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}

static float signed_pattern(uint64_t index, uint32_t multiplier,
                            uint32_t modulus, float denominator) {
  const int32_t centered =
      (int32_t)((index * multiplier + multiplier * 7u) % modulus) -
      (int32_t)(modulus / 2u);
  return (float)centered / denominator;
}

int kda_make_case(uint16_t *q, uint16_t *k, uint16_t *v,
                  uint16_t *log_decay, uint16_t *beta, uint16_t *state,
                  uint32_t mode) {
  if (!q || !k || !v || !log_decay || !beta || !state || mode > 5)
    return 1;

  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t h = 0; h < H; ++h) {
      for (uint32_t d = 0; d < D; ++d) {
        const uint64_t index = ((uint64_t)n * H + h) * D + d;
        float q_value = signed_pattern(index, 17, 61, 24.0f);
        float k_value = signed_pattern(index, 29, 67, 26.0f);
        if (mode == 3) {
          q_value += (float)h * 0.0625f;
          k_value -= (float)h * 0.046875f;
          if (d == ((h * 7u + n * 3u) & 127u)) {
            q_value += 2.0f;
            k_value -= 1.5f;
          }
        }
        q[index] = float_to_bf16(q_value);
        k[index] = float_to_bf16(k_value);
      }
    }

    for (uint32_t hv = 0; hv < HV; ++hv) {
      const uint64_t beta_index = (uint64_t)n * HV + hv;
      float beta_value = 0.125f +
                         (float)((n * 11u + hv * 7u) % 25u) / 32.0f;
      if (mode == 2)
        beta_value = 0.0f;
      beta[beta_index] = float_to_bf16(beta_value);

      for (uint32_t d = 0; d < D; ++d) {
        const uint64_t index = ((uint64_t)n * HV + hv) * D + d;
        float v_value = signed_pattern(index, 19, 59, 20.0f) +
                        (float)hv * 0.0078125f;
        if (mode == 3 && d == ((hv * 13u + n) & 127u))
          v_value += (float)(hv + 1u) * 0.03125f;
        v[index] = float_to_bf16(v_value);

        float gate_value =
            -0.015625f - (float)((n * 5u + hv * 3u + d) % 17u) / 128.0f;
        if (mode == 4) {
          static const float gates[4] = {-0.0005f, -0.125f, -1.5f, -4.0f};
          gate_value = gates[(n + hv + d) & 3u];
        }
        log_decay[index] = float_to_bf16(gate_value);
      }

      for (uint32_t dv = 0; dv < D; ++dv) {
        for (uint32_t dk = 0; dk < D; ++dk) {
          const uint64_t index =
              (((uint64_t)n * HV + hv) * D + dv) * D + dk;
          float state_value = signed_pattern(index, 13, 71, 512.0f);
          if (mode == 1)
            state_value = 0.0f;
          else if (mode == 5) {
            const int32_t centered =
                (int32_t)((index * 37u + dv * 11u + dk * 5u) % 257u) -
                128;
            state_value = (float)centered / 8192.0f +
                          ((index & 1u) ? 0.000030517578125f
                                        : -0.000030517578125f);
          }
          state[index] = float_to_bf16(state_value);
        }
      }
    }
  }
  return 0;
}

int kda_truth(const uint16_t *q, const uint16_t *k, const uint16_t *v,
              const uint16_t *log_decay, const uint16_t *beta,
              uint16_t *state, uint16_t *output, float scale) {
  if (!q || !k || !v || !log_decay || !beta || !state || !output)
    return 1;

  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t hv = 0; hv < HV; ++hv) {
      const uint32_t qh = hv / 2u;
      float q_norm[D], k_norm[D], decay[D];
      float q_squares = 0.0f, k_squares = 0.0f;
      const uint64_t qk_base = ((uint64_t)n * H + qh) * D;
      const uint64_t vh_base = ((uint64_t)n * HV + hv) * D;

      for (uint32_t d = 0; d < D; ++d) {
        q_norm[d] = bf16_to_float(q[qk_base + d]);
        k_norm[d] = bf16_to_float(k[qk_base + d]);
        q_squares += q_norm[d] * q_norm[d];
        k_squares += k_norm[d] * k_norm[d];
      }
      const float q_inverse = scale / sqrtf(q_squares + 1.0e-6f);
      const float k_inverse = 1.0f / sqrtf(k_squares + 1.0e-6f);
      for (uint32_t d = 0; d < D; ++d) {
        q_norm[d] *= q_inverse;
        k_norm[d] *= k_inverse;
        decay[d] = expf(bf16_to_float(log_decay[vh_base + d]));
      }

      const float beta_value = bf16_to_float(beta[(uint64_t)n * HV + hv]);
      for (uint32_t dv = 0; dv < D; ++dv) {
        const uint64_t state_base =
            (((uint64_t)n * HV + hv) * D + dv) * D;
        float decayed[D];
        float prediction = 0.0f;
        for (uint32_t dk = 0; dk < D; ++dk) {
          decayed[dk] = bf16_to_float(state[state_base + dk]) * decay[dk];
          prediction += decayed[dk] * k_norm[dk];
        }
        const float residual = bf16_to_float(v[vh_base + dv]) - prediction;
        float projected = 0.0f;
        for (uint32_t dk = 0; dk < D; ++dk) {
          const float updated =
              decayed[dk] + beta_value * residual * k_norm[dk];
          const uint16_t rounded = float_to_bf16(updated);
          state[state_base + dk] = rounded;
          projected += bf16_to_float(rounded) * q_norm[dk];
        }
        output[vh_base + dv] = float_to_bf16(projected);
      }
    }
  }
  return 0;
}

int kda_compare(const uint16_t *got, const uint16_t *expected, uint64_t count,
                float atol, float rtol, uint64_t *bad_index, float *bad_got,
                float *bad_expected, float *max_abs, float *max_rel) {
  if (!got || !expected || !bad_index || !bad_got || !bad_expected ||
      !max_abs || !max_rel)
    return 2;
  *max_abs = 0.0f;
  *max_rel = 0.0f;
  for (uint64_t i = 0; i < count; ++i) {
    const float g = bf16_to_float(got[i]);
    const float e = bf16_to_float(expected[i]);
    const float absolute = fabsf(g - e);
    const float relative = absolute / fmaxf(fabsf(e), 1.0e-20f);
    if (absolute > *max_abs)
      *max_abs = absolute;
    if (relative > *max_rel)
      *max_rel = relative;
    if (!isfinite(g) || absolute > atol + rtol * fabsf(e)) {
      *bad_index = i;
      *bad_got = g;
      *bad_expected = e;
      return 1;
    }
  }
  return 0;
}
