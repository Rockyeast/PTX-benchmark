#include <math.h>
#include <stdint.h>

#define N 128

static float bf16_to_float(uint16_t value) {
  union { uint32_t u; float f; } bits;
  bits.u = ((uint32_t)value) << 16;
  return bits.f;
}

int gemm_fp8_truth(const uint16_t* a, const uint16_t* b,
                   float* expected, float* row_scale) {
  if (!a || !b || !expected || !row_scale) return 1;
  for (int m = 0; m < N; ++m) {
    float maximum = 0.0f;
    for (int n = 0; n < N; ++n) {
      float accumulator = 0.0f;
      for (int k = 0; k < N; ++k) {
        accumulator = fmaf(bf16_to_float(a[m * N + k]),
                           bf16_to_float(b[n * N + k]), accumulator);
      }
      expected[m * N + n] = accumulator;
      maximum = fmaxf(maximum, fabsf(accumulator));
    }
    row_scale[m] = maximum > 0.0f ? maximum / 448.0f : 1.0f;
  }
  return 0;
}
