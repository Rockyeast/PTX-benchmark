#include <math.h>
#include <stdint.h>

#define S 1024
#define D 128
#define BLOCK 16

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

int block_sparse_truth(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                       const int32_t* row_offsets, const int32_t* block_indices,
                       uint16_t* output) {
  if (!q || !k || !v || !row_offsets || !block_indices || !output) return 1;
  const float scale = 0.08838834764831845f;
  for (int query = 0; query < S; ++query) {
    const int row = query / BLOCK;
    const int begin = row_offsets[row];
    const int count = (row_offsets[row + 1] - begin) * BLOCK;
    if (count < 16 || count > 128) return 2;
    float weights[128];
    float maximum = -INFINITY;
    for (int slot = 0; slot < count; ++slot) {
      const int key = block_indices[begin + slot / BLOCK] * BLOCK + slot % BLOCK;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) {
        dot = fmaf(bf16_to_float(q[query * D + d]),
                   bf16_to_float(k[key * D + d]), dot);
      }
      weights[slot] = dot * scale;
      maximum = fmaxf(maximum, weights[slot]);
    }
    float denominator = 0.0f;
    for (int slot = 0; slot < count; ++slot) {
      weights[slot] = expf(weights[slot] - maximum);
      denominator += weights[slot];
    }
    for (int d = 0; d < D; ++d) {
      float accumulator = 0.0f;
      for (int slot = 0; slot < count; ++slot) {
        const int key = block_indices[begin + slot / BLOCK] * BLOCK + slot % BLOCK;
        accumulator = fmaf(weights[slot] / denominator,
                           bf16_to_float(v[key * D + d]), accumulator);
      }
      output[query * D + d] = float_to_bf16(accumulator);
    }
  }
  return 0;
}
