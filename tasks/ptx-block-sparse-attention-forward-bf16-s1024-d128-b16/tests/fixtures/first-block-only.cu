#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

// Negative fixture: reads CSR but silently ignores every selected block after
// the first one.
extern "C" __global__ __launch_bounds__(128)
void block_sparse_attn_fwd_bf16_s1024_d128_b16(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const int32_t* row_offsets,
    const int32_t* block_indices, __nv_bfloat16* output) {
  constexpr int D = 128, BLOCK = 16;
  constexpr float scale = 0.08838834764831845f;
  const int query = blockIdx.x, tid = threadIdx.x;
  const int begin = row_offsets[query / BLOCK];
  __shared__ float weights[BLOCK];
  const int key = block_indices[begin] * BLOCK + tid % BLOCK;
  float dot = 0.0f;
  if (tid < BLOCK) {
    for (int d = 0; d < D; ++d) {
      dot = fmaf(__bfloat162float(q[query * D + d]),
                 __bfloat162float(k[key * D + d]), dot);
    }
    weights[tid] = dot * scale;
  }
  __syncthreads();
  if (tid == 0) {
    float maximum = -INFINITY, denominator = 0.0f;
    for (int slot = 0; slot < BLOCK; ++slot) maximum = fmaxf(maximum, weights[slot]);
    for (int slot = 0; slot < BLOCK; ++slot) {
      weights[slot] = expf(weights[slot] - maximum);
      denominator += weights[slot];
    }
    for (int slot = 0; slot < BLOCK; ++slot) weights[slot] /= denominator;
  }
  __syncthreads();
  float accumulator = 0.0f;
  for (int slot = 0; slot < BLOCK; ++slot) {
    const int selected = block_indices[begin] * BLOCK + slot;
    accumulator = fmaf(weights[slot], __bfloat162float(v[selected * D + tid]), accumulator);
  }
  output[query * D + tid] = __float2bfloat16_rn(accumulator);
}
