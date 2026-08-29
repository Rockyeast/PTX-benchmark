# ptx-add-one validation

This is the first GPU infrastructure task for PTX-Bench. The agent produces
`/app/kernel.ptx`; the verifier loads it through the CUDA Driver API, launches
`add_one` on a B200, and writes a binary reward.

This task checks correctness only. It does not measure kernel performance and
is not yet a benchmark-quality task.
