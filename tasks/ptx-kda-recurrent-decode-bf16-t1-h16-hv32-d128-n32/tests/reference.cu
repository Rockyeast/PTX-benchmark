// Trusted author-side source for the fixed same-ABI KDA performance Reference.
#include <cuda_bf16.h>
#include <stdint.h>

static __device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return __shfl_sync(0xffffffffu, value, 0);
}

extern "C" __global__ __launch_bounds__(32)
void kda_recurrent_decode_t1_h16_hv32_d128_n32(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *v, const __nv_bfloat16 *log_decay,
    const __nv_bfloat16 *beta, __nv_bfloat16 *state,
    __nv_bfloat16 *output, float scale) {
  const uint32_t lane = threadIdx.x;
  const uint32_t n = blockIdx.y;
  const uint32_t hv = blockIdx.x >> 3;
  const uint32_t row_group = blockIdx.x & 7u;
#ifdef KDA_WRONG_HEAD_MAP
  const uint32_t qh = hv & 15u;
#else
  const uint32_t qh = hv >> 1;
#endif
  const uint64_t qk_base = ((uint64_t)n * 16 + qh) * 128;
  const uint64_t vh_base = ((uint64_t)n * 32 + hv) * 128;

  float q_values[4], k_values[4], decay[4];
  float q_squares = 0.0f, k_squares = 0.0f;
#pragma unroll
  for (uint32_t group = 0; group < 4; ++group) {
    const uint32_t dk = lane + group * 32;
    q_values[group] = __bfloat162float(q[qk_base + dk]);
    k_values[group] = __bfloat162float(k[qk_base + dk]);
    q_squares = fmaf(q_values[group], q_values[group], q_squares);
    k_squares = fmaf(k_values[group], k_values[group], k_squares);
  }
  q_squares = warp_sum(q_squares);
  k_squares = warp_sum(k_squares);
  const float q_inverse = rsqrtf(q_squares + 1.0e-6f) * scale;
  const float k_inverse = rsqrtf(k_squares + 1.0e-6f);
#pragma unroll
  for (uint32_t group = 0; group < 4; ++group) {
    const uint32_t dk = lane + group * 32;
    q_values[group] *= q_inverse;
    k_values[group] *= k_inverse;
    decay[group] = expf(__bfloat162float(log_decay[vh_base + dk]));
  }

  const float beta_value = __bfloat162float(beta[(uint64_t)n * 32 + hv]);
#pragma unroll
  for (uint32_t row = 0; row < 16; ++row) {
    const uint32_t dv = row_group * 16 + row;
    const uint64_t state_base =
        (((uint64_t)n * 32 + hv) * 128 + dv) * 128;
    float decayed[4];
    float prediction = 0.0f;
#pragma unroll
    for (uint32_t group = 0; group < 4; ++group) {
      const uint32_t dk = lane + group * 32;
      decayed[group] = __bfloat162float(state[state_base + dk]) * decay[group];
      prediction = fmaf(decayed[group], k_values[group], prediction);
    }
    prediction = warp_sum(prediction);
    const float residual = __bfloat162float(v[vh_base + dv]) - prediction;
    float projected = 0.0f;
#pragma unroll
    for (uint32_t group = 0; group < 4; ++group) {
      const uint32_t dk = lane + group * 32;
      float updated = fmaf(beta_value * residual, k_values[group], decayed[group]);
#ifdef KDA_WRONG_IGNORE_BETA
      updated = fmaf(residual, k_values[group], decayed[group]);
#endif
      const __nv_bfloat16 rounded = __float2bfloat16_rn(updated);
      state[state_base + dk] = rounded;
#ifdef KDA_WRONG_OUTPUT_OLD_STATE
      projected = fmaf(decayed[group], q_values[group], projected);
#elif defined(KDA_WRONG_OUTPUT_PRE_ROUND)
      projected = fmaf(updated, q_values[group], projected);
#else
      projected = fmaf(__bfloat162float(rounded), q_values[group], projected);
#endif
    }
    projected = warp_sum(projected);
    if (lane == 0)
      output[vh_base + dv] = __float2bfloat16_rn(projected);
  }

#ifdef KDA_SLOW_CORRECT
  if (lane == 0) {
    const unsigned long long start = clock64();
    while (clock64() - start < 200000ull) {
      asm volatile("");
    }
  }
#endif
}
