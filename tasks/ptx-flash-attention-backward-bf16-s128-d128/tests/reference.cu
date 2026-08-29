// Trusted author-side source for the fixed same-ABI Reference PTX.
// This file is never exposed to the untrusted Agent.
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ __launch_bounds__(128)
void flash_attn_bwd_bf16_s128_d128(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    const __nv_bfloat16* grad_output,
    float* grad_q,
    float* grad_k,
    float* grad_v) {
  constexpr int S = 128;
  constexpr int D = 128;
  constexpr float scale = 0.08838834764831845f;
  const int i = static_cast<int>(threadIdx.x);
  if (i >= S) return;

  for (int d = 0; d < D; ++d) {
    grad_k[i * D + d] = 0.0f;
    grad_v[i * D + d] = 0.0f;
  }
  __syncthreads();

  float probability[S];
  float d_probability[S];
  float row_max = -INFINITY;
  for (int j = 0; j < S; ++j) {
    float qk = 0.0f;
    float dov = 0.0f;
    for (int d = 0; d < D; ++d) {
      qk = fmaf(__bfloat162float(q[i * D + d]),
                __bfloat162float(k[j * D + d]), qk);
      dov = fmaf(__bfloat162float(grad_output[i * D + d]),
                 __bfloat162float(v[j * D + d]), dov);
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

  for (int d = 0; d < D; ++d) {
    float dq = 0.0f;
    const float q_value = __bfloat162float(q[i * D + d]);
    const float do_value = __bfloat162float(grad_output[i * D + d]);
    for (int j = 0; j < S; ++j) {
      const float ds = probability[j] * (d_probability[j] - delta);
      dq = fmaf(ds, __bfloat162float(k[j * D + d]), dq);
      atomicAdd(&grad_k[j * D + d], scale * ds * q_value);
      atomicAdd(&grad_v[j * D + d], probability[j] * do_value);
    }
    grad_q[i * D + d] = scale * dq;
  }
}
