// Independent CPU truth for fixed T=6 speculative KDA decode.
// This file intentionally does not include or call the INT21 implementation.

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { T = 6, N = 8, H = 16, HV = 32, D = 128, SLOTS = N * T + 6 };

static float bf16_to_float(uint16_t value) {
  union { uint32_t u; float f; } bits = {(uint32_t)value << 16};
  return bits.f;
}

static uint16_t float_to_bf16(float value) {
  union { float f; uint32_t u; } bits = {value};
  uint32_t lsb = (bits.u >> 16) & 1u;
  return (uint16_t)((bits.u + 0x7fffu + lsb) >> 16);
}

static float pattern(uint64_t index, uint32_t mul, uint32_t mod, float scale) {
  int32_t centered = (int32_t)((index * mul + 17u) % mod) - (int32_t)(mod / 2u);
  return (float)centered * scale;
}

uint64_t kda_t6_qk_elements(void) { return (uint64_t)N * T * H * D; }
uint64_t kda_t6_vg_elements(void) { return (uint64_t)N * T * HV * D; }
uint64_t kda_t6_beta_elements(void) { return (uint64_t)N * T * HV; }
uint64_t kda_t6_state_elements(void) { return (uint64_t)SLOTS * HV * D * D; }
uint64_t kda_t6_output_elements(void) { return (uint64_t)N * T * HV * D; }
uint64_t kda_t6_index_elements(void) { return (uint64_t)N * T; }
uint64_t kda_t6_accepted_elements(void) { return N; }

int kda_t6_make_case(uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *g,
                     uint16_t *beta, uint16_t *state, int32_t *state_indices,
                     int32_t *accepted, uint32_t mode) {
  if (!q || !k || !v || !g || !beta || !state || !state_indices || !accepted)
    return 1;

  const int32_t accepted_default[N] = {0, 1, 6, 13, 5, 1, 6, 0};
  const int32_t accepted_each[N] = {1, 2, 3, 4, 5, 6, 0, 99};
  for (int n = 0; n < N; ++n) {
    accepted[n] = (mode == 7) ? accepted_each[n] : accepted_default[n];
    for (int t = 0; t < T; ++t) {
      int logical = n * T + t;
      state_indices[logical] =
          (mode == 1) ? 1 + (logical * 17) % (N * T) : 1 + logical;
    }
  }

  for (uint64_t i = 0; i < kda_t6_qk_elements(); ++i) {
    float qv = pattern(i, 29u, 101u, 1.0f / 31.0f);
    float kv = pattern(i, 43u, 97u, 1.0f / 29.0f);
    if (mode == 4) {
      uint64_t head = (i / D) % H;
      uint64_t d = i % D;
      qv = ((float)head - 7.5f) * 0.11f + ((float)(d % 11) - 5.0f) * 0.017f;
      kv = ((float)(15 - head) - 7.5f) * 0.09f + ((float)(d % 13) - 6.0f) * 0.019f;
    }
    q[i] = float_to_bf16(qv);
    k[i] = float_to_bf16(kv);
  }

  for (uint64_t i = 0; i < kda_t6_vg_elements(); ++i) {
    float vv = pattern(i, 37u, 89u, 1.0f / 67.0f);
    float gv = -0.03f - (float)((i * 13u + 5u) % 23u) * 0.012f;
    if (mode == 5) {
      uint64_t d = i % D;
      gv = (d % 4 == 0) ? -0.0005f : (d % 4 == 1) ? -0.2f :
           (d % 4 == 2) ? -1.5f : -5.0f;
    }
    if (mode == 6)
      vv = pattern(i, 19u, 31u, 1.0f / 2048.0f);
    v[i] = float_to_bf16(vv);
    g[i] = float_to_bf16(gv);
  }

  for (uint64_t i = 0; i < kda_t6_beta_elements(); ++i) {
    float bv = 0.08f + (float)((i * 7u + 3u) % 17u) * 0.035f;
    if (mode == 3) bv = 0.0f;
    beta[i] = float_to_bf16(bv);
  }

  for (uint64_t i = 0; i < kda_t6_state_elements(); ++i) {
    float sv = pattern(i, 31u, 127u, 1.0f / 4096.0f);
    if (mode == 2) sv = 0.0f;
    if (mode == 6) sv = pattern(i, 23u, 47u, 1.0f / 16384.0f);
    state[i] = float_to_bf16(sv);
  }
  return 0;
}

int kda_t6_truth(const uint16_t *q, const uint16_t *k, const uint16_t *v,
                 const uint16_t *g, const uint16_t *beta, uint16_t *state,
                 uint16_t *output, const int32_t *state_indices,
                 const int32_t *accepted, float scale) {
  if (!q || !k || !v || !g || !beta || !state || !output ||
      !state_indices || !accepted)
    return 1;

  float *work = (float *)malloc((uint64_t)HV * D * D * sizeof(float));
  if (!work) return 2;

  for (int n = 0; n < N; ++n) {
    int accepted_token = accepted[n] - 1;
    if (accepted_token < 0) accepted_token = 0;
    if (accepted_token >= T) accepted_token = T - 1;
    int initial_slot = state_indices[n * T + accepted_token];
    if (initial_slot < 0) initial_slot = 0;
    if (initial_slot >= SLOTS) { free(work); return 3; }

    uint64_t initial_base = (uint64_t)initial_slot * HV * D * D;
    for (uint64_t i = 0; i < (uint64_t)HV * D * D; ++i)
      work[i] = bf16_to_float(state[initial_base + i]);

    for (int t = 0; t < T; ++t) {
      int token = n * T + t;
      int checkpoint_slot = state_indices[token];
      if (checkpoint_slot < 0 || checkpoint_slot >= SLOTS) {
        free(work);
        return 4;
      }
      for (int hv = 0; hv < HV; ++hv) {
        int qh = hv / 2;
        uint64_t qk_base = ((uint64_t)token * H + qh) * D;
        uint64_t vh_base = ((uint64_t)token * HV + hv) * D;
        float qnorm[D], knorm[D], decay[D];
        float qsum = 0.0f, ksum = 0.0f;
        for (int d = 0; d < D; ++d) {
          float qv = bf16_to_float(q[qk_base + d]);
          float kv = bf16_to_float(k[qk_base + d]);
          qnorm[d] = qv;
          knorm[d] = kv;
          qsum += qv * qv;
          ksum += kv * kv;
          decay[d] = expf(bf16_to_float(g[vh_base + d]));
        }
        float qscale = scale / sqrtf(qsum + 1.0e-6f);
        float kscale = 1.0f / sqrtf(ksum + 1.0e-6f);
        for (int d = 0; d < D; ++d) {
          qnorm[d] *= qscale;
          knorm[d] *= kscale;
        }
        float beta_value = bf16_to_float(beta[(uint64_t)token * HV + hv]);

        for (int dv = 0; dv < D; ++dv) {
          uint64_t work_base = ((uint64_t)hv * D + dv) * D;
          float prediction = 0.0f;
          for (int dk = 0; dk < D; ++dk)
            prediction += work[work_base + dk] * decay[dk] * knorm[dk];
          float residual = bf16_to_float(v[vh_base + dv]) - prediction;
          float update_scale = beta_value * residual;
          float projected = 0.0f;
          uint64_t checkpoint_base =
              (((uint64_t)checkpoint_slot * HV + hv) * D + dv) * D;
          for (int dk = 0; dk < D; ++dk) {
            float updated = work[work_base + dk] * decay[dk] +
                            update_scale * knorm[dk];
            uint16_t rounded = float_to_bf16(updated);
            state[checkpoint_base + dk] = rounded;
            work[work_base + dk] = bf16_to_float(rounded);
            projected += work[work_base + dk] * qnorm[dk];
          }
          output[vh_base + dv] = float_to_bf16(projected);
        }
      }
    }
  }
  free(work);
  return 0;
}

int kda_t6_compare(const uint16_t *got, const uint16_t *expected,
                   uint64_t count, float atol, float rtol,
                   uint64_t *bad_index, float *bad_got, float *bad_expected,
                   float *max_abs, float *max_rel) {
  if (!got || !expected || !bad_index || !bad_got || !bad_expected ||
      !max_abs || !max_rel)
    return 2;
  *max_abs = 0.0f;
  *max_rel = 0.0f;
  for (uint64_t i = 0; i < count; ++i) {
    float gv = bf16_to_float(got[i]);
    float ev = bf16_to_float(expected[i]);
    float abs_error = fabsf(gv - ev);
    float rel_error = abs_error / fmaxf(fabsf(ev), 1.0e-12f);
    if (abs_error > *max_abs) *max_abs = abs_error;
    if (rel_error > *max_rel) *max_rel = rel_error;
    if (!(abs_error <= atol + rtol * fabsf(ev))) {
      *bad_index = i;
      *bad_got = gv;
      *bad_expected = ev;
      return 1;
    }
  }
  return 0;
}
