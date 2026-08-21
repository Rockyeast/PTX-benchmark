#!/bin/bash

set -euo pipefail

output_path="${OUTPUT_PATH:-/app/kernel.ptx}"

cat > "$output_path" <<'PTX'
.version 8.0
.target sm_89
.address_size 64

.visible .entry add_one(
    .param .u64 add_one_param_0,
    .param .u64 add_one_param_1,
    .param .u32 add_one_param_2
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;

    ld.param.u64 %rd1, [add_one_param_0];
    ld.param.u64 %rd2, [add_one_param_1];
    ld.param.u32 %r2, [add_one_param_2];

    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mov.u32 %r5, %tid.x;
    mad.lo.s32 %r1, %r3, %r4, %r5;

    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra DONE;

    mul.wide.u32 %rd3, %r1, 4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.s32 %r6, [%rd4];
    add.s32 %r7, %r6, 1;
    add.u64 %rd5, %rd2, %rd3;
    st.global.s32 [%rd5], %r7;

DONE:
    ret;
}
PTX
