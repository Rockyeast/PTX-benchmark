# @@TITLE@@

Edit `/app/kernel.ptx` directly. Only that PTX file is graded.

The submission must define a visible entry named `@@ENTRY_NAME@@`. Do not use
CUDA C++, Triton, Python, NVRTC, NVVM, or another higher-level compiler to
generate the submission. `ptxas` and `cuobjdump` may be used to inspect PTX.

## Operation and tensors

TODO(task author): state the complete operation, input/output tensors, shapes,
dtypes, physical layouts, formulas, aliasing rules, and boundary behavior.

## Entry and launch contract

TODO(task author): state every parameter in exact order and type, the address
width, grid dimensions, block dimensions, dynamic shared memory, and who owns
each launch choice.

The target device is @@HARDWARE_TARGET@@ (`sm_@@SM@@`).

## Correctness and performance

TODO(task author): state observable tolerances, disclosed development cases,
and the performance objective. Correctness is a hard gate before performance.

A correct but slower mathematical implementation is available at
`/app/reference.py`. Use the public development harness with:

```bash
python3 /app/measure.py /app/kernel.ptx
```

The public harness does not reveal hidden cases, final reward, the performance
Baseline, or the performance Reference.
