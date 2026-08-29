// Independent CPU truth for the seven-route, D=128 KDA decode family.
// This file intentionally does not include or call the INT21 implementation.

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

enum { H = 16, D = 128 };

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

static int valid_shape(int tokens, int sequences, int value_heads, int slots) {
  if (tokens < 1 || tokens > 6 || sequences < 1 || slots < sequences) return 0;
  if (tokens == 3) return value_heads == 16 && slots >= sequences * tokens + 6;
  if (value_heads != 32) return 0;
  if (tokens == 1) return slots >= sequences;
  return slots >= sequences * tokens + 6;
}

int kda_full_make_case(
    uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *g, uint16_t *beta,
    float *a_log, float *dt_bias, uint16_t *state, int32_t *state_indices,
    int32_t *accepted, int tokens, int sequences, int value_heads, int slots,
    uint32_t mode) {
  if (!q || !k || !v || !g || !beta || !state || !state_indices || !accepted)
    return 1;
  if (!valid_shape(tokens, sequences, value_heads, slots)) return 2;
  if (tokens == 3 && (!a_log || !dt_bias)) return 3;

  const uint64_t total = (uint64_t)tokens * sequences;
  const uint64_t qk_elements = total * H * D;
  const uint64_t vg_elements = total * value_heads * D;
  const uint64_t beta_elements = total * value_heads;
  const uint64_t state_elements = (uint64_t)slots * value_heads * D * D;

  for (uint64_t i = 0; i < qk_elements; ++i) {
    float qv = pattern(i + (uint64_t)tokens * 11u, 29u, 101u, 1.0f / 31.0f);
    float kv = pattern(i + (uint64_t)sequences * 7u, 43u, 97u, 1.0f / 29.0f);
    if (mode == 1) {
      uint64_t head = (i / D) % H;
      uint64_t d = i % D;
      qv = ((float)head - 7.5f) * 0.07f + ((float)(d % 17) - 8.0f) * 0.013f;
      kv = ((float)(15u - head) - 7.5f) * 0.09f + ((float)(d % 13) - 6.0f) * 0.019f;
    }
    q[i] = float_to_bf16(qv);
    k[i] = float_to_bf16(kv);
  }

  for (uint64_t i = 0; i < vg_elements; ++i) {
    float vv = pattern(i + (uint64_t)tokens * 19u, 37u, 89u, 1.0f / 67.0f);
    float gv;
    if (tokens == 3) {
      gv = pattern(i, 23u, 53u, 1.0f / 19.0f);
      if (mode == 1 && (i % D) < 4u) gv = (float)(i % D) - 1.5f;
    } else {
      gv = -0.02f - (float)((i * 13u + 5u) % 29u) * 0.025f;
      if (mode == 1) {
        uint64_t d = i % D;
        gv = (d % 4u == 0u) ? -0.0005f :
             (d % 4u == 1u) ? -0.2f :
             (d % 4u == 2u) ? -1.5f : -5.0f;
      }
    }
    v[i] = float_to_bf16(vv);
    g[i] = float_to_bf16(gv);
  }

  for (uint64_t i = 0; i < beta_elements; ++i) {
    float bv = 0.06f + (float)((i * 7u + 3u) % 19u) * 0.035f;
    if (mode == 2 && (i % 5u) == 0u) bv = 0.0f;
    beta[i] = float_to_bf16(bv);
  }

  for (uint64_t i = 0; i < state_elements; ++i) {
    float sv = pattern(i + (uint64_t)tokens * 31u, 31u, 127u, 1.0f / 4096.0f);
    if (mode == 2) sv *= 0.125f;
    state[i] = float_to_bf16(sv);
  }

  if (tokens == 3) {
    for (int h = 0; h < H; ++h)
      a_log[h] = logf(1.0f + 0.025f * (float)(h + 1));
    for (int i = 0; i < H * D; ++i)
      dt_bias[i] = pattern((uint64_t)i, 17u, 61u, 1.0f / 23.0f);
  }

  for (int n = 0; n < sequences; ++n) {
    accepted[n] = (mode == 1) ? (n % (tokens + 3)) : ((n + tokens) % tokens) + 1;
    for (int t = 0; t < tokens; ++t) {
      int logical = n * tokens + t;
      if (tokens == 1) {
        state_indices[logical] = n;
      } else if (mode == 1) {
        state_indices[logical] = 1 + (logical * 17) % (sequences * tokens);
      } else {
        state_indices[logical] = 1 + logical;
      }
    }
  }
  return 0;
}

int kda_full_truth(
    const uint16_t *q, const uint16_t *k, const uint16_t *v,
    const uint16_t *g, const uint16_t *beta, const float *a_log,
    const float *dt_bias, uint16_t *state, uint16_t *output,
    const int32_t *state_indices, const int32_t *accepted, int tokens,
    int sequences, int value_heads, int slots, float scale,
    float lower_bound) {
  if (!q || !k || !v || !g || !beta || !state || !output ||
      !state_indices || !accepted)
    return 1;
  if (!valid_shape(tokens, sequences, value_heads, slots)) return 2;
  if (tokens == 3 && (!a_log || !dt_bias)) return 3;

  float *work = (float *)malloc((uint64_t)value_heads * D * D * sizeof(float));
  if (!work) return 4;
  const int head_ratio = value_heads / H;

  for (int n = 0; n < sequences; ++n) {
    int initial_slot;
    if (tokens == 1) {
      initial_slot = n;
    } else {
      int accepted_token = accepted[n] - 1;
      if (accepted_token < 0) accepted_token = 0;
      if (accepted_token >= tokens) accepted_token = tokens - 1;
      initial_slot = state_indices[n * tokens + accepted_token];
      if (initial_slot < 0) initial_slot = 0;
    }
    if (initial_slot >= slots) { free(work); return 5; }

    uint64_t initial_base = (uint64_t)initial_slot * value_heads * D * D;
    for (uint64_t i = 0; i < (uint64_t)value_heads * D * D; ++i)
      work[i] = bf16_to_float(state[initial_base + i]);

    for (int t = 0; t < tokens; ++t) {
      int token = n * tokens + t;
      int checkpoint_slot = (tokens == 1) ? n : state_indices[token];
      if (checkpoint_slot < 0 || checkpoint_slot >= slots) {
        free(work);
        return 6;
      }
      for (int hv = 0; hv < value_heads; ++hv) {
        int qh = hv / head_ratio;
        uint64_t qk_base = ((uint64_t)token * H + qh) * D;
        uint64_t vh_base = ((uint64_t)token * value_heads + hv) * D;
        float qnorm[D], knorm[D], decay[D];
        float qsum = 0.0f, ksum = 0.0f;
        for (int d = 0; d < D; ++d) {
          float qv = bf16_to_float(q[qk_base + d]);
          float kv = bf16_to_float(k[qk_base + d]);
          qnorm[d] = qv;
          knorm[d] = kv;
          qsum += qv * qv;
          ksum += kv * kv;
          float log_decay;
          if (tokens == 3) {
            float gate_a = expf(a_log[qh]);
            float x = gate_a * (bf16_to_float(g[vh_base + d]) + dt_bias[qh * D + d]);
            float sigmoid = 1.0f / (1.0f + expf(-x));
            log_decay = lower_bound * sigmoid;
          } else {
            log_decay = bf16_to_float(g[vh_base + d]);
          }
          decay[d] = expf(log_decay);
        }
        float qscale = scale / sqrtf(qsum + 1.0e-6f);
        float kscale = 1.0f / sqrtf(ksum + 1.0e-6f);
        for (int d = 0; d < D; ++d) {
          qnorm[d] *= qscale;
          knorm[d] *= kscale;
        }
        float beta_value = bf16_to_float(beta[(uint64_t)token * value_heads + hv]);

        for (int dv = 0; dv < D; ++dv) {
          uint64_t work_base = ((uint64_t)hv * D + dv) * D;
          float prediction = 0.0f;
          for (int dk = 0; dk < D; ++dk)
            prediction += work[work_base + dk] * decay[dk] * knorm[dk];
          float residual = bf16_to_float(v[vh_base + dv]) - prediction;
          float update_scale = beta_value * residual;
          float projected = 0.0f;
          uint64_t checkpoint_base =
              (((uint64_t)checkpoint_slot * value_heads + hv) * D + dv) * D;
          for (int dk = 0; dk < D; ++dk) {
            float updated = work[work_base + dk] * decay[dk] + update_scale * knorm[dk];
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

int kda_full_compare(
    const uint16_t *got, const uint16_t *expected, uint64_t count,
    float atol, float rtol, uint64_t *bad_index, float *bad_got,
    float *bad_expected, float *max_abs, float *max_rel) {
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
