# @@TITLE@@

Create `/app/kernel.ptx` containing NVIDIA PTX assembly. Only this file is
graded.

Write or edit the PTX directly. Do not generate the submission by compiling
CUDA C++, Triton, Python, or another higher-level language. `ptxas` and
`cuobjdump` may be used to validate and inspect PTX, but `nvcc`, NVRTC, and
NVVM are not available.

A task-pinned PTX ISA reference is available under `/opt/ptx-docs`. Use the
local snapshot rather than downloading replacement documentation or tools.

The submission must define a visible kernel entry named `@@ENTRY_NAME@@`.

## Logical contract

TODO(task author): specify the exact logical signature, tensor shapes, dtypes,
layouts, mathematical operation, aliasing rules, and edge cases.

## Fixed launch ABI

TODO(task author): specify parameter order and types, address width, grid and
block dimensions, dynamic shared memory, and any fixed specialization.

## Correctness and performance

TODO(task author): state observable correctness tolerances and whether/how
performance is measured. Do not reveal hidden cases or the Oracle
implementation.

The target device is @@HARDWARE_TARGET@@. Targeting `sm_@@SM@@` is recommended.
Correctness must pass before any performance score is awarded.
