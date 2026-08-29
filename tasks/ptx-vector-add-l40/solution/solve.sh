#!/bin/bash

set -euo pipefail

output_path="${OUTPUT_PATH:-/app/kernel.ptx}"

cat > "$output_path" <<'PTX'
.version 9.2
.target sm_89
.address_size 64

.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2,
    .param .u32 vector_add_param_3
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [vector_add_param_0];
    ld.param.u64 %rd2, [vector_add_param_1];
    ld.param.u64 %rd3, [vector_add_param_2];
    ld.param.u32 %r1, [vector_add_param_3];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;

    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.rn.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

DONE:
    ret;
}
PTX
