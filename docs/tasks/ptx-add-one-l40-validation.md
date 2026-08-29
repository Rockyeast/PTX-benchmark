# PTX add-one L40 calibration validation

This task is the L40 infrastructure-calibration counterpart of
`tasks/ptx-add-one/`. It verifies that a Harbor agent can submit pure PTX and
that the verifier can load and execute it on an NVIDIA L40 (`sm_89`).

It is deliberately excluded from `tasks/dataset.toml`; passing it is
infrastructure evidence, not a formal PTX-Bench score.

The task now follows the project metadata and environment contract for
`direct-ptx-v0`. Its CUDA 13.2.1 devel image keeps `ptxas` and `cuobjdump` but
removes `nvcc`, NVRTC, NVVM, and Triton. Agent execution remains restricted to
the Lovelace model API host, so package managers cannot fetch new tools while
the Agent is solving the task.

NVIDIA PTX ISA 9.2 is downloaded from the CUDA 13.2.1 official archive during
the local image build and checked against SHA-256
`82bebdef912a6c5622ea1622688fdfb40949c8e12071af184dd3349042a2756c`.
It is not stored in this repository because NVIDIA's notice requires advance
written approval for reproduction. The image-local copy is mode `0444` at
`/opt/ptx-docs/ptx-isa-9.2.pdf`; the built image must not be redistributed
without confirming permission. Provenance is recorded in
`tasks/ptx-add-one-l40/environment/PTX-ISA-SOURCE.md`.

The verifier runs `ptxas` only after Agent execution and stores its diagnostics
in `/logs/verifier/ptxas.txt`. The same Agent never receives those diagnostics.
The submitted `/app/kernel.ptx` is collected as a job artifact for failure
audit, but the Oracle remains outside the Agent environment.

The agent budget is 600 seconds. A first 120-second calibration run timed out
before Qwen Code completed its initial response on a single L40 model server;
that timeout measured serving latency rather than PTX correctness.

A later single-L40 run with Qwen thinking disabled created `/app/kernel.ptx`
after its first response (461 output tokens, zero thought tokens), but Qwen Code
then entered another long response and reached the 600-second agent timeout.
The retained candidate failed `cuModuleLoad` with CUDA error 218, so this run
proves prompt-to-tool progress but not task success. Its evidence is stored in
`jobs/ptx-add-one-l40-qwen38-no-thinking-20260821/`.

A follow-up run set `QWEN_CODE_MAX_OUTPUT_TOKENS=1024`. All nine model responses
were already shorter than that limit (461 tokens at most), so response length
was not the bottleneck. The agent instead installed `nvidia-cuda-toolkit` with
`apt-get`, spending about 197 seconds, and reached `ptxas` only near the end of
the 600-second budget. `ptxas` reported invalid register and predicate operands;
the agent timed out before repairing them, and the retained PTX again failed
`cuModuleLoad` with CUDA error 218. The evidence is stored in
`jobs/ptx-add-one-l40-qwen38-no-thinking-max1024-20260821/`.

The task now keeps the environment public during agent installation but changes
the agent execution phase to Harbor's network allowlist, permitting only the
Lovelace model API host. The instruction also explicitly forbids downloading or
installing toolchains. This is a Direct-PTX calibration constraint: an empty
initial toolchain alone is insufficient when the running agent has public
network access and root package-manager access.

The first hardened rerun used that historical hostname allowlist plus the
1024-token response cap. The current service no longer publishes its API on the
Lovelace physical interface; current L40 tasks allow only the Docker bridge
gateway `10.100.0.1` and require API authentication. This paragraph describes
the preserved 2026-08-21 job rather than the current deployment policy.
Harbor's egress-control sidecar reported controlled egress with only
`lovelace.ece.local.cmu.edu` allowed. Qwen Code made no install attempt and
terminated normally after about 355 seconds, so the infrastructure timeout was
resolved. It rewrote its first candidate after noticing pointer-register type
errors, but the final candidate still used a 32-bit source register in
`shl.b64`. An author-side `ptxas` diagnostic reported exactly one argument
mismatch at that instruction; the verifier consequently returned CUDA error
218 and reward 0. The evidence is stored in
`jobs/ptx-add-one-l40-qwen38-allowlist-max1024-20260821/`.

The first full `direct-ptx-v0` rerun kept the same model deployment, response
cap, and allowlist while adding the offline PTX ISA manual and Agent-visible
`ptxas`. The Agent used the manual, created PTX directly, created no
higher-level source, and made no install or network-download attempt. It spent
most of the 600-second budget searching the manual, wrote its first candidate
near the end, and invoked `ptxas` once. Harbor timed it out before a repair
iteration. The retained candidate failed assembly and then failed
`cuModuleLoad` with CUDA error 218; reward was 0. This is a model
time-management and first-candidate correctness failure, not a missing-document
or verifier failure. The Oracle passed the same rebuilt environment and hidden
verifier with reward 1. Evidence is stored in
`jobs/ptx-add-one-l40-qwen38-direct-v0-20260821/`.

The Docker backend in Harbor 0.21.0 does not allocate a specific local GPU.
`tasks/ptx-add-one-l40/environment/docker-compose.yaml` therefore reads the host GPU index from
`PTX_BENCH_GPU_DEVICE`. Set that variable to an idle L40 before starting the
job.
The verifier uploads its policy checker and pinned PTX-document manifest from
`tests/` only after the Agent phase. Grading therefore does not trust a checker
or manifest that the Agent could have modified inside the task environment.
