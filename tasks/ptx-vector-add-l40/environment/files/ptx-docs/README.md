# NVIDIA PTX ISA reference used by this environment

- Document: NVIDIA Parallel Thread Execution ISA 9.2
- CUDA Toolkit archive: 13.2.1
- Official source: https://docs.nvidia.com/cuda/archive/13.2.1/pdf/ptx_isa_9.2.pdf
- Official notices: https://docs.nvidia.com/cuda/archive/13.2.1/parallel-thread-execution/index.html#notices
- SHA-256: `82bebdef912a6c5622ea1622688fdfb40949c8e12071af184dd3349042a2756c`
- Runtime path: `/opt/ptx-docs/ptx-isa-9.2.pdf`

The NVIDIA notice does not grant a general copyright license and says that
reproduction requires NVIDIA's advance written approval. The PDF is therefore
not vendored in this repository. It is fetched from the official NVIDIA archive
while building the local evaluation image and accepted only when its SHA-256
matches the value above. Do not publish the resulting image or redistribute the
PDF without confirming that the required permission has been obtained.
