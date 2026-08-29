# Oracle source provenance

The files in `decode_sm100a/` and `flash_kda_decode_sm100a.cpp` are copied from:

```text
repository: https://github.com/Int21-AI/KDA-B200
revision:   4896cd2da4ca421829907ab9fd3550e3f84a141b
license:    MIT; see INT21-MIT-LICENSE
```

The seven files in `generated/` were compiled separately with NVIDIA CUDA
13.2.1 `nvcc -ptx -O3 --use_fast_math -arch=sm_100a`, then combined by
`scripts/combine_ptx_modules.py` into `solution/oracle.ptx`.

These generated kernels are used only as the Oracle submission. They are not
used to calculate expected outputs: the hidden verifier instead compiles and
calls the independent generic C truth in `tests/kda_full_truth.c`.
