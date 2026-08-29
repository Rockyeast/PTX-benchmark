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

// INT21 Direction A: state is a one-pass 256-byte row stream per half warp.
// Bypass L1 allocation and request the aligned 256-byte L2 sector footprint;
// packed software lookahead keeps independent state segments outstanding.
__device__ __forceinline__ uint4 load_state_stream_l2_256(const __nv_bfloat16* ptr) {
    uint4 packed;
    asm volatile(
        "ld.global.L1::no_allocate.L2::256B.v4.b32 {%0, %1, %2, %3}, [%4];"
        : "=r"(packed.x), "=r"(packed.y), "=r"(packed.z), "=r"(packed.w)
        : "l"(ptr)
        : "memory");
    return packed;
}

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

__global__ __launch_bounds__(32) void
flash_kda_decode_sm100a_t1_split16(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ g, __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ state, __nv_bfloat16* __restrict__ out, float scale)
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
    const int value_splits = 16;
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
    int tile_row_base = value_tile * 8;
    int elem_start = lane_0 * 4;
    float q_src[4];
    float k_src[4];
    float gate_src[4];
    float q_reg[8];
    float k_reg[8];
    float gate_reg[8];
    float state_row[8];
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
    #pragma unroll 1
    for (int row_block = 0; row_block < 1; row_block++) {
        // Keep N=8 on the original split16 geometry and serial row schedule.
        // Four packed rows are issued before row 0 consumption, avoiding the
        // original 32-float staging footprint while preserving arithmetic.
        uint4 state_prefetch[4];
        #pragma unroll
        for (int row_prefetch = 0; row_prefetch < 4; row_prefetch++) {
            int prefetch_row_in_tile = v_lane + 2 * (row_block * 4 + row_prefetch);
            int prefetch_value_row = tile_row_base + prefetch_row_in_tile;
            int prefetch_state_base = initial_head_base + prefetch_value_row * 128 + k_lane * 8;
            state_prefetch[row_prefetch] = load_state_stream_l2_256(state + prefetch_state_base);
        }
        #pragma unroll
        for (int row_local_1 = 0; row_local_1 < 4; row_local_1++) {
            uint32_t* state_pairs = reinterpret_cast<uint32_t*>(&state_prefetch[row_local_1]);
            #pragma unroll
            for (int pair = 0; pair < 4; pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"(state_row[pair * 2]), "=f"(state_row[pair * 2 + 1])
                    : "r"(state_pairs[pair]));
            }
            int row_in_tile_1 = v_lane + 2 * (row_block * 4 + row_local_1);
            int value_row_1 = tile_row_base + row_in_tile_1;
            float pred = 0.0f;
            float base = 0.0f;
            {
                #pragma unroll
                for (int i_1 = 0; i_1 < 8; i_1++) {
                    float h_decay = state_row[i_1] * gate_reg[i_1];
                    pred += h_decay * k_reg[i_1];
                    base += h_decay * q_reg[i_1];
                }
            }
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, pred, 8);
            pred += _shfl_xor_4;
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, pred, 4);
            pred += _shfl_xor_5;
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, pred, 2);
            pred += _shfl_xor_6;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, pred, 1);
            pred += _shfl_xor_7;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, base, 8);
            base += _shfl_xor_8;
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, base, 4);
            base += _shfl_xor_9;
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, base, 2);
            base += _shfl_xor_10;
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, base, 1);
            base += _shfl_xor_11;
            float v_value = 0.0f;
            if (k_lane == 0) {
                v_value = (float)v[vg_base + value_row_1];
            }
            float _shfl_3 = __shfl_sync(0xFFFFFFFF, v_value, v_lane * 16);
            v_value = _shfl_3;
            float delta = (v_value - pred) * beta_value;
            #pragma unroll
            for (int i_2 = 0; i_2 < 8; i_2++) {
                state_row[i_2] = state_row[i_2] * gate_reg[i_2] + delta * k_reg[i_2];
            }
            if (active) {
                int state_base_1 = output_state_head_base + value_row_1 * 128 + k_lane * 8;
                {
                    __nv_bfloat162 _pk[4];
                    _pk[0] = __floats2bfloat162_rn(state_row[0], state_row[1]);
                    _pk[1] = __floats2bfloat162_rn(state_row[2], state_row[3]);
                    _pk[2] = __floats2bfloat162_rn(state_row[4], state_row[5]);
                    _pk[3] = __floats2bfloat162_rn(state_row[6], state_row[7]);
                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(state))[state_base_1 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
                if (k_lane == 0) {
                    out[output_head_base + value_row_1] = base + delta * k_dot_q;
                }
            } else if (has_token && k_lane == 0) {
                out[output_head_base + value_row_1] = 0.0f;
            }
        }
    }
}

} // extern "C"
// END FROZEN GENERATED BODY
// clang-format on


// INT21 AOT launch shim;
extern "C" cudaError_t flash_kda_decode_sm100a_launch_t1_split16(void* q, void* k, void* value, void* gate, void* beta, void* state, void* out, int sequences, float scale, cudaStream_t stream) {
  auto bf = [](void* ptr) { return reinterpret_cast<__nv_bfloat16*>(ptr); };
  const dim3 grid(512, sequences, 1);
  flash_kda_decode_sm100a_t1_split16<<<grid, 32, 0, stream>>>(bf(q), bf(k), bf(value), bf(gate), bf(beta), bf(state), bf(out), scale);
  return cudaGetLastError();
}
