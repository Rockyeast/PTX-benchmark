#include <math.h>
#include <stdint.h>
#include <string.h>

#define QLEN 128
#define HQ 8
#define HKV 2
#define D 128
#define PAGE 16
#define PAGES 32
#define QBLOCK 16

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

static int cache_index(const int32_t* page_table, int logical_token, int hkv, int d) {
  const int physical_page = page_table[logical_token / PAGE];
  return (((physical_page * PAGE + logical_token % PAGE) * HKV + hkv) * D + d);
}

static int validate_metadata(const int32_t* page_table, const int32_t* row_offsets,
                             const int32_t* page_indices) {
  uint32_t page_seen = 0;
  for (int p = 0; p < PAGES; ++p) {
    if (page_table[p] < 0 || page_table[p] >= PAGES) return 2;
    const uint32_t bit = 1u << page_table[p];
    if (page_seen & bit) return 2;
    page_seen |= bit;
  }
  if (row_offsets[0] != 0 || row_offsets[8] > 64) return 3;
  for (int r = 0; r < 8; ++r) {
    const int count = row_offsets[r + 1] - row_offsets[r];
    if (count < 2 || count > 8) return 3;
    uint32_t seen = 0;
    for (int s = row_offsets[r]; s < row_offsets[r + 1]; ++s) {
      const int p = page_indices[s];
      if (p < 0 || p >= PAGES || (seen & (1u << p))) return 3;
      seen |= 1u << p;
    }
  }
  return 0;
}

int paged_sparse_forward_truth(
    const uint16_t* q, const uint16_t* k_cache, const uint16_t* v_cache,
    const int32_t* page_table, const int32_t* row_offsets,
    const int32_t* page_indices, uint16_t* output, float* lse) {
  if (!q || !k_cache || !v_cache || !page_table || !row_offsets ||
      !page_indices || !output || !lse) return 1;
  const int metadata = validate_metadata(page_table, row_offsets, page_indices);
  if (metadata) return metadata;
  const float scale = 0.08838834764831845f;
  for (int qi = 0; qi < QLEN; ++qi) {
    const int row = qi / QBLOCK;
    const int global_query = 384 + qi;
    for (int hq = 0; hq < HQ; ++hq) {
      const int hkv = hq / 4;
      float scores[128];
      int tokens[128];
      int count = 0;
      float maximum = -INFINITY;
      for (int slot = row_offsets[row]; slot < row_offsets[row + 1]; ++slot) {
        const int logical_page = page_indices[slot];
        for (int offset = 0; offset < PAGE; ++offset) {
          const int token = logical_page * PAGE + offset;
          if (token > global_query) continue;
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) {
            dot = fmaf(bf16_to_float(q[(qi * HQ + hq) * D + d]),
                       bf16_to_float(k_cache[cache_index(page_table, token, hkv, d)]), dot);
          }
          tokens[count] = token;
          scores[count] = dot * scale;
          maximum = fmaxf(maximum, scores[count]);
          ++count;
        }
      }
      if (count == 0) return 4;
      float denominator = 0.0f;
      for (int s = 0; s < count; ++s) denominator += expf(scores[s] - maximum);
      const float logsumexp = logf(denominator) + maximum;
      lse[qi * HQ + hq] = logsumexp;
      for (int d = 0; d < D; ++d) {
        float accumulator = 0.0f;
        for (int s = 0; s < count; ++s) {
          const float probability = expf(scores[s] - logsumexp);
          accumulator = fmaf(probability,
                             bf16_to_float(v_cache[cache_index(page_table, tokens[s], hkv, d)]),
                             accumulator);
        }
        output[(qi * HQ + hq) * D + d] = float_to_bf16(accumulator);
      }
    }
  }
  return 0;
}

int paged_sparse_backward_truth(
    const uint16_t* q, const uint16_t* k_cache, const uint16_t* v_cache,
    const uint16_t* output, const uint16_t* grad_output, const float* lse,
    const int32_t* page_table, const int32_t* row_offsets,
    const int32_t* page_indices, float* grad_q, float* grad_k_cache,
    float* grad_v_cache) {
  if (!q || !k_cache || !v_cache || !output || !grad_output || !lse ||
      !page_table || !row_offsets || !page_indices || !grad_q ||
      !grad_k_cache || !grad_v_cache) return 1;
  const int metadata = validate_metadata(page_table, row_offsets, page_indices);
  if (metadata) return metadata;
  const float scale = 0.08838834764831845f;
  const int q_elements = QLEN * HQ * D;
  const int cache_elements = PAGES * PAGE * HKV * D;
  memset(grad_q, 0, q_elements * sizeof(float));
  memset(grad_k_cache, 0, cache_elements * sizeof(float));
  memset(grad_v_cache, 0, cache_elements * sizeof(float));

  for (int qi = 0; qi < QLEN; ++qi) {
    const int row = qi / QBLOCK;
    const int global_query = 384 + qi;
    for (int hq = 0; hq < HQ; ++hq) {
      const int hkv = hq / 4;
      float delta = 0.0f;
      for (int d = 0; d < D; ++d) {
        delta = fmaf(bf16_to_float(grad_output[(qi * HQ + hq) * D + d]),
                     bf16_to_float(output[(qi * HQ + hq) * D + d]), delta);
      }
      for (int slot = row_offsets[row]; slot < row_offsets[row + 1]; ++slot) {
        const int logical_page = page_indices[slot];
        for (int offset = 0; offset < PAGE; ++offset) {
          const int token = logical_page * PAGE + offset;
          if (token > global_query) continue;
          float score = 0.0f;
          float d_probability = 0.0f;
          for (int d = 0; d < D; ++d) {
            score = fmaf(bf16_to_float(q[(qi * HQ + hq) * D + d]),
                         bf16_to_float(k_cache[cache_index(page_table, token, hkv, d)]), score);
            d_probability = fmaf(
                bf16_to_float(grad_output[(qi * HQ + hq) * D + d]),
                bf16_to_float(v_cache[cache_index(page_table, token, hkv, d)]), d_probability);
          }
          const float probability = expf(score * scale - lse[qi * HQ + hq]);
          const float ds = probability * (d_probability - delta);
          for (int d = 0; d < D; ++d) {
            const int q_index = (qi * HQ + hq) * D + d;
            const int c_index = cache_index(page_table, token, hkv, d);
            grad_q[q_index] = fmaf(scale * ds, bf16_to_float(k_cache[c_index]), grad_q[q_index]);
            grad_k_cache[c_index] = fmaf(scale * ds, bf16_to_float(q[q_index]), grad_k_cache[c_index]);
            grad_v_cache[c_index] = fmaf(probability, bf16_to_float(grad_output[q_index]), grad_v_cache[c_index]);
          }
        }
      }
    }
  }
  return 0;
}
