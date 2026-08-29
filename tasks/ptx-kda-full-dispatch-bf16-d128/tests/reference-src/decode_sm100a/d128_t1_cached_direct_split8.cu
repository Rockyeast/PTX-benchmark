// Copyright 2026 INT21 AI
// SPDX-License-Identifier: MIT
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) LoomTensorMapPack { LoomTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define LOOM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 32

#include <math_constants.h>

extern "C" {

// One-warp CTA is fixed by the launch shim; cap this ILP variant below 96 registers.
__global__ __maxnreg__(96) void
flash_kda_decode_sm100a_t1_split8(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ out, float scale)
{
    constexpr int H = 16;
    constexpr int HV = 32;
    constexpr int head_ratio = 2;
    constexpr int g_stride_token = 4096;
    constexpr int state_stride_slot = 524288;
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int work = blockIdx.x;
    const int value_splits = 8;
    int value_tile = work & value_splits - 1;
    int hv = work / value_splits;
    int n = blockIdx.y;
    int query_head = hv / head_ratio;
    int lane_0 = lane;
    int k_lane = lane_0 & 15;
    int v_lane = lane_0 / 16;
    int token_pos = n;
    int initial_slot = n;
    constexpr bool active = true;
    constexpr bool has_token = true;
    int tile_row_base = value_tile * 16;
    int elem_start = lane_0 * 4;
    float q_src[4];
    float k_src[4];
    float gate_src[4];
    float q_reg[8];
    float k_reg[8];
    float gate_reg[8];
    int q_base = (token_pos * H + query_head) * 128 + elem_start;
    int vg_base = (token_pos * HV + hv) * 128;
    int gate_base = token_pos * g_stride_token + hv * 128 + elem_start;
    {
        uint2 _vld_0 = *reinterpret_cast<const uint2*>(q + q_base);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&q_src[0 + _pair * 2])[0]), "=f"((&q_src[0 + _pair * 2])[1])
                : "r"(_vpairs_0[_pair]));
        }
    }
    {
        uint2 _vld_1 = *reinterpret_cast<const uint2*>(k + q_base);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&k_src[0 + _pair * 2])[0]), "=f"((&k_src[0 + _pair * 2])[1])
                : "r"(_vpairs_1[_pair]));
        }
    }
    {
        uint2 _vld_2 = *reinterpret_cast<const uint2*>(g + gate_base);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&gate_src[0 + _pair * 2])[0]), "=f"((&gate_src[0 + _pair * 2])[1])
                : "r"(_vpairs_2[_pair]));
        }
    }
    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    {
        q_sum_sq = q_src[0] * q_src[0] + q_src[1] * q_src[1] + (q_src[2] * q_src[2] + q_src[3] * q_src[3]);
        k_sum_sq = k_src[0] * k_src[0] + k_src[1] * k_src[1] + (k_src[2] * k_src[2] + k_src[3] * k_src[3]);
    }
    float _warp_reduce_0 = q_sum_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    q_sum_sq = _warp_reduce_0;
    float _warp_reduce_1 = k_sum_sq;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    k_sum_sq = _warp_reduce_1;
    float _rsqrt_0 = rsqrtf(q_sum_sq + 1e-06f);
    float q_scale = _rsqrt_0 * scale;
    float _rsqrt_1 = rsqrtf(k_sum_sq + 1e-06f);
    float k_scale = _rsqrt_1;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int source_lane = 2 * k_lane + i / 4;
        const int source_value = i & 3;
        float _shfl_0 = __shfl_sync(0xFFFFFFFF, q_src[source_value], source_lane);
        q_reg[i] = _shfl_0 * q_scale;
        float _shfl_1 = __shfl_sync(0xFFFFFFFF, k_src[source_value], source_lane);
        k_reg[i] = _shfl_1 * k_scale;
        float _shfl_2 = __shfl_sync(0xFFFFFFFF, gate_src[source_value], source_lane);
        float _expf_0 = __expf(_shfl_2);
        gate_reg[i] = _expf_0;
    }
    float k_dot_q = 0.0f;
    {
        k_dot_q = k_reg[0] * q_reg[0] + k_reg[1] * q_reg[1] + (k_reg[2] * q_reg[2] + k_reg[3] * q_reg[3]) + (k_reg[4] * q_reg[4] + k_reg[5] * q_reg[5] + (k_reg[6] * q_reg[6] + k_reg[7] * q_reg[7]));
    }
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, k_dot_q, 8);
    k_dot_q += _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_dot_q, 4);
    k_dot_q += _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, k_dot_q, 2);
    k_dot_q += _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_dot_q, 1);
    k_dot_q += _shfl_xor_3;
    float beta_value = (float)beta[token_pos * HV + hv];
    int initial_head_base = initial_slot * state_stride_slot + hv * 128 * 128;
    int output_head_base = (token_pos * HV + hv) * 128;
    int output_state_head_base = initial_slot * state_stride_slot + hv * 128 * 128;
    // INT21 Iteration-4 D: every lane still owns one 8-element K segment for
    // each of its value rows, but state loads are now issued as a four-row
    // prefetch window.  Rows (0,1) and (2,3) then advance as independent,
    // interleaved accumulator chains.  Per-row multiply/add and shuffle order
    // is unchanged; the second pair's LDG results can remain in flight while
    // arithmetic for the first pair issues.
    #pragma unroll 1
    for (int row_block = 0; row_block < 2; row_block++) {
        uint4 packed_rows[4];
        #pragma unroll
        for (int row_slot = 0; row_slot < 4; row_slot++) {
            const int row_in_tile = v_lane + 2 * (row_block * 4 + row_slot);
            const int value_row = tile_row_base + row_in_tile;
            const int state_base = initial_head_base + value_row * 128 + k_lane * 8;
            const uint4* state_ptr = reinterpret_cast<const uint4*>(state + state_base);
            asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                : "=r"(packed_rows[row_slot].x), "=r"(packed_rows[row_slot].y),
                  "=r"(packed_rows[row_slot].z), "=r"(packed_rows[row_slot].w)
                : "l"(state_ptr) : "memory");
        }

        #pragma unroll
        for (int row_pair = 0; row_pair < 2; row_pair++) {
            constexpr int kElems = 8;
            const int row_slot_a = 2 * row_pair;
            const int row_slot_b = row_slot_a + 1;
            const int row_in_tile_a = v_lane + 2 * (row_block * 4 + row_slot_a);
            const int row_in_tile_b = v_lane + 2 * (row_block * 4 + row_slot_b);
            const int value_row_a = tile_row_base + row_in_tile_a;
            const int value_row_b = tile_row_base + row_in_tile_b;
            float state_a[kElems];
            float state_b[kElems];
            uint32_t* packed_a = reinterpret_cast<uint32_t*>(&packed_rows[row_slot_a]);
            uint32_t* packed_b = reinterpret_cast<uint32_t*>(&packed_rows[row_slot_b]);
            #pragma unroll
            for (int pair = 0; pair < 4; pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"(state_a[pair * 2]), "=f"(state_a[pair * 2 + 1])
                    : "r"(packed_a[pair]));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"(state_b[pair * 2]), "=f"(state_b[pair * 2 + 1])
                    : "r"(packed_b[pair]));
            }

            float pred_a = 0.0f;
            float base_a = 0.0f;
            float pred_b = 0.0f;
            float base_b = 0.0f;
            #pragma unroll
            for (int i = 0; i < kElems; i++) {
                const float decay_a = state_a[i] * gate_reg[i];
                pred_a += decay_a * k_reg[i];
                base_a += decay_a * q_reg[i];
                const float decay_b = state_b[i] * gate_reg[i];
                pred_b += decay_b * k_reg[i];
                base_b += decay_b * q_reg[i];
            }
            #pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                pred_a += __shfl_xor_sync(0xFFFFFFFF, pred_a, offset);
                pred_b += __shfl_xor_sync(0xFFFFFFFF, pred_b, offset);
                base_a += __shfl_xor_sync(0xFFFFFFFF, base_a, offset);
                base_b += __shfl_xor_sync(0xFFFFFFFF, base_b, offset);
            }

            float value_a = 0.0f;
            float value_b = 0.0f;
            if (k_lane == 0) {
                value_a = (float)v[vg_base + value_row_a];
                value_b = (float)v[vg_base + value_row_b];
            }
            value_a = __shfl_sync(0xFFFFFFFF, value_a, v_lane * 16);
            value_b = __shfl_sync(0xFFFFFFFF, value_b, v_lane * 16);
            const float delta_a = (value_a - pred_a) * beta_value;
            const float delta_b = (value_b - pred_b) * beta_value;
            #pragma unroll
            for (int i = 0; i < kElems; i++) {
                state_a[i] = state_a[i] * gate_reg[i] + delta_a * k_reg[i];
                state_b[i] = state_b[i] * gate_reg[i] + delta_b * k_reg[i];
            }

            if (active) {
                const int state_base_a = output_state_head_base + value_row_a * 128 + k_lane * 8;
                const int state_base_b = output_state_head_base + value_row_b * 128 + k_lane * 8;
                __nv_bfloat162 packed_out_a[4];
                __nv_bfloat162 packed_out_b[4];
                #pragma unroll
                for (int pair = 0; pair < 4; pair++) {
                    packed_out_a[pair] = __floats2bfloat162_rn(state_a[pair * 2], state_a[pair * 2 + 1]);
                    packed_out_b[pair] = __floats2bfloat162_rn(state_b[pair * 2], state_b[pair * 2 + 1]);
                }
                *reinterpret_cast<uint4*>(state + state_base_a) = *reinterpret_cast<uint4*>(&packed_out_a[0]);
                *reinterpret_cast<uint4*>(state + state_base_b) = *reinterpret_cast<uint4*>(&packed_out_b[0]);
                if (k_lane == 0) {
                    out[output_head_base + value_row_a] = base_a + delta_a * k_dot_q;
                    out[output_head_base + value_row_b] = base_b + delta_b * k_dot_q;
                }
            } else if (has_token && k_lane == 0) {
                out[output_head_base + value_row_a] = 0.0f;
                out[output_head_base + value_row_b] = 0.0f;
            }
        }
    }
}

} // extern "C"
// END FROZEN GENERATED BODY
// clang-format on


// INT21 AOT launch shim;
extern "C" cudaError_t flash_kda_decode_sm100a_launch_t1_split8(void* q, void* k, void* value, void* gate, void* beta, void* state, void* out, int sequences, float scale, cudaStream_t stream) {
  auto bf = [](void* ptr) { return reinterpret_cast<__nv_bfloat16*>(ptr); };
  const dim3 grid(256, sequences, 1);
  flash_kda_decode_sm100a_t1_split8<<<grid, 32, 0, stream>>>(bf(q), bf(k), bf(value), bf(gate), bf(beta), bf(state), bf(out), scale);
  return cudaGetLastError();
}
