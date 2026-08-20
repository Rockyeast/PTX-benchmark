# Feasibility reference provenance

`reference.cu` is a small task-owned CUDA implementation of the mathematical
RMSNorm contract. It is compiled to PTX with NVIDIA `nvcc` before timing and
uses exactly the Candidate ABI and launch configuration.

It was written for the initial interface smoke test. It is not copied from and
must not be reported as the performance of INT21's optimized implementation.

The operation, tested shape, public tolerance, and the choice of 128 threads
for the BF16 `N=4096` plain-forward specialization were checked against:

- <https://github.com/Int21-AI/RMSNorm-B200>
- revision `6b0a8beeaa306b1a55c0dc5c5e03dafeda914c5f`
- upstream license: MIT

Before this task becomes a benchmark release, the task-owned Reference must be
reviewed against the INT21 fast path and an independently measured INT21
operator result must be recorded. Correctness never comes from either
performance implementation; the verifier computes the formula independently.
