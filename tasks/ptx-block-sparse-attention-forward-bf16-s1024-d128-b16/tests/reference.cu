#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

extern "C" __global__ __launch_bounds__(128)
void block_sparse_attn_fwd_bf16_s1024_d128_b16(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, const int32_t* row_offsets,
    const int32_t* block_indices, __nv_bfloat16* output) {
  constexpr int S = 1024;
  constexpr int D = 128;
  constexpr int BLOCK = 16;
  constexpr float scale = 0.08838834764831845f;
  const int query = static_cast<int>(blockIdx.x);
  const int tid = static_cast<int>(threadIdx.x);
  if (query >= S) return;
  const int row = query / BLOCK;
  const int begin = row_offsets[row];
  const int count = (row_offsets[row + 1] - begin) * BLOCK;
  __shared__ float weights[128];

  float score = -INFINITY;
  if (tid < count) {
    const int key_block = block_indices[begin + tid / BLOCK];
    const int key = key_block * BLOCK + tid % BLOCK;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
      dot = fmaf(__bfloat162float(q[query * D + d]),
                 __bfloat162float(k[key * D + d]), dot);
    }
    score = dot * scale;
  }
  weights[tid] = score;
  __syncthreads();

  if (tid == 0) {
    float maximum = -INFINITY;
    for (int slot = 0; slot < count; ++slot) maximum = fmaxf(maximum, weights[slot]);
    float denominator = 0.0f;
    for (int slot = 0; slot < count; ++slot) {
      weights[slot] = expf(weights[slot] - maximum);
      denominator += weights[slot];
    }
    for (int slot = 0; slot < count; ++slot) weights[slot] /= denominator;
    for (int slot = count; slot < 128; ++slot) weights[slot] = 0.0f;
  }
  __syncthreads();

  float accumulator = 0.0f;
  for (int slot = 0; slot < count; ++slot) {
    const int key_block = block_indices[begin + slot / BLOCK];
    const int key = key_block * BLOCK + slot % BLOCK;
    accumulator = fmaf(weights[slot], __bfloat162float(v[key * D + tid]), accumulator);
  }
  output[query * D + tid] = __float2bfloat16_rn(accumulator);
}
