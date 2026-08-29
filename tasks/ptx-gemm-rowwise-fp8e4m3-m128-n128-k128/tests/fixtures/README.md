# Negative fixtures

`wrong-scale.ptx` uses `amax/127` instead of the required E4M3 `amax/448`.
It must assemble/load and fail the row-scale correctness gate.
