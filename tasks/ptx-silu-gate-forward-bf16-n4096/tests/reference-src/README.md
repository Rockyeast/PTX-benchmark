# Performance Reference provenance

The task-owned `../reference.ptx` is generated from `reference.cu` in a trusted
author environment. The source adapts the vectorized activation-and-multiply
structure from:

```text
repository: https://github.com/flashinfer-ai/flashinfer
commit:     103fcf862b902e100775ac781a0de36d067de3a4
file:       include/flashinfer/activation.cuh
license:    Apache-2.0
```

Benchmark-specific changes:

- fix input/output dtype to BF16;
- fix output width to 4096;
- expose the task ABI `(input, output, num_tokens)`;
- use exactly one 512-thread block per token;
- retain one 16-byte gate load, one 16-byte up load, and one 16-byte output
  store per thread;
- omit FlashInfer's Programmatic Dependent Launch instructions because the task
  launch ABI explicitly disables PDL;
- remove framework, template, Python/JIT, and host-dispatch dependencies.

Generation command in the CUDA 13.2.1 trusted author image:

```bash
nvcc \
  --ptx \
  --std=c++17 \
  -O3 \
  --generate-code=arch=compute_100,code=compute_100 \
  -o reference.ptx \
reference.cu
```

Compiler identity:

```text
CUDA compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
```

Pinned SHA-256:

```text
reference.cu:
c53f7f713687fddbfda5161eaf00b9712d0d47922fdb45a623a05a489bfbcfab

reference.ptx:
89889b1e098239bbbb0e47adc47152ce4af1de063220951c85e6b179a38dd39c
```

`nvcc` is not present in the untrusted Direct-PTX Agent environment; the
verifier only loads the pinned PTX artifact.
