/*
 * Copyright (c) 2024 by FlashInfer team.
 * Copyright (c) 2026 PTX-Bench contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Adapted from FlashInfer include/flashinfer/activation.cuh at commit
// 103fcf862b902e100775ac781a0de36d067de3a4. The benchmark specialization
// fixes BF16, output width 4096, a 512-thread launch, and the raw PTX ABI.
// Programmatic Dependent Launch instructions are deliberately omitted.

struct alignas(16) Bf16x8 {
  __nv_bfloat16 values[8];
};

extern "C" __global__ __launch_bounds__(512) void silu_gate_fwd_bf16_n4096(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int num_tokens) {
  const unsigned int token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }

  constexpr unsigned int kWidth = 4096;
  constexpr unsigned int kVectorElements = 8;
  constexpr unsigned int kVectorsPerRow = kWidth / kVectorElements;

  const unsigned int vector_index = threadIdx.x;
  const unsigned long long input_row =
      static_cast<unsigned long long>(token) * 2 * kVectorsPerRow;
  const unsigned long long output_row =
      static_cast<unsigned long long>(token) * kVectorsPerRow;

  const Bf16x8* input_vectors = reinterpret_cast<const Bf16x8*>(input);
  Bf16x8* output_vectors = reinterpret_cast<Bf16x8*>(output);
  const Bf16x8 gate = input_vectors[input_row + vector_index];
  const Bf16x8 up =
      input_vectors[input_row + kVectorsPerRow + vector_index];
  Bf16x8 result;

#pragma unroll
  for (unsigned int index = 0; index < kVectorElements; ++index) {
    const float gate_value = __bfloat162float(gate.values[index]);
    const float up_value = __bfloat162float(up.values[index]);
    const float silu = gate_value / (1.0f + __expf(-gate_value));
    result.values[index] = __float2bfloat16_rn(silu * up_value);
  }

  output_vectors[output_row + vector_index] = result;
}
