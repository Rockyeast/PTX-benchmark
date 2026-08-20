# PTX-Bench Apptainer image

`ptx-bench-cuda13.2.def` builds the reusable GPU task environment used on
`catalyst-b200`.

It contains:

- Ubuntu 24.04;
- CUDA Toolkit 13.2.1, including `nvcc` and `ptxas`;
- Python 3 and a prebuilt `/opt/harbor-server` environment containing FastAPI
  and Uvicorn;
- `build-essential`, `curl`, `git`, `ripgrep`, `tmux`, and `asciinema`;
- writable task paths `/app`, `/tests`, and `/logs/verifier`.

It intentionally does not install Codex or another model-specific agent. Harbor
continues to install the selected agent during the agent setup phase.

## Build on Catalyst

The home filesystem has limited free space, so both the Apptainer cache and
expanded build tree must live under `/tmp`. The two-hour timeout remains below
the server's 2.5-hour per-process limit.

```bash
mkdir -p /tmp/zhuominc-ptx-apptainer-cache
mkdir -p /tmp/zhuominc-ptx-apptainer-tmp

APPTAINER_CACHEDIR=/tmp/zhuominc-ptx-apptainer-cache \
APPTAINER_TMPDIR=/tmp/zhuominc-ptx-apptainer-tmp \
timeout 7200 \
apptainer build --fakeroot \
  /home/zhuominc/rockyeast-work/containers/ptx-bench-cuda13.2.1-ubuntu24.04-v1.sif \
  /home/zhuominc/rockyeast-work/PTX-benchmark/images/ptx-bench-cuda13.2.def
```

## Static tool check

```bash
singularity exec --containall --no-mount home,tmp,bind-paths --pwd / \
  /home/zhuominc/rockyeast-work/containers/ptx-bench-cuda13.2.1-ubuntu24.04-v1.sif \
  bash -lc 'python3 --version; nvcc --version; ptxas --version'
```

GPU visibility still requires the patched Harbor Singularity backend to pass
`--nv`, or a direct `singularity exec --nv` invocation.

## Validated artifact

Built and tested on `catalyst-b200` on 2026-08-19:

```text
Path:   /home/zhuominc/rockyeast-work/containers/ptx-bench-cuda13.2.1-ubuntu24.04-v1.sif
Size:   3.5 GiB
SHA256: 841897a25dd7b05cf387c3b7a580c6bfcf20ae15b3a8858303d49139b15f9ebd
```

Validation completed:

- definition `%test` passed;
- `ptxas -arch=sm_100` passed;
- CUDA Driver API PTX execution passed on one B200;
- Harbor Oracle trial `ptx-add-one-oracle-fixed-sif` returned reward `1.0`;
- Harbor environment setup took about 8.1 seconds and performed no runtime
  Python/FastAPI installation.

Use the image without putting its server-specific absolute path in `task.toml`:

```bash
--environment-type singularity \
--environment-kwarg singularity_nv=true \
--environment-kwarg singularity_image_path=/home/zhuominc/rockyeast-work/containers/ptx-bench-cuda13.2.1-ubuntu24.04-v1.sif
```

The `singularity_image_path` runtime override is supplied by
`patches/harbor/0001-singularity-nvidia-workdir.patch`.

## Codex-specific image

`ptx-bench-codex0.148.def` derives from the generic CUDA image and adds:

- Node.js `22.23.2` from the official signed-release archive;
- Codex CLI `0.148.0` installed under `/opt/codex-cli`;
- stable `/usr/local/bin/node`, `npm`, `npx`, and `codex` links.

This image is intentionally versioned separately from the generic image. It
lets Harbor skip its per-trial NVM/npm Codex installation while preserving the
agent-neutral base image for other agents.

Built and tested on `catalyst-b200` on 2026-08-20:

```text
Path:   /home/zhuominc/rockyeast-work/containers/ptx-bench-cuda13.2.1-codex0.148-ubuntu24.04-v1.sif
Size:   3.6 GiB
SHA256: a5d3f9c53e9b8bd03f3631311e55790c851ce3c554137b7b3cbe804ab2634ed7
```

Validation completed:

- `node --version` returned `v22.23.2` and `codex --version` returned
  `codex-cli 0.148.0`;
- Harbor trial `ptx-add-one-codex-pinned-sif` returned reward `1.0`;
- Codex setup took about 1.0 second instead of the failed runtime-install path's
  35.3 seconds;
- Codex found and successfully ran `/usr/local/cuda/bin/ptxas`;
- the complete Codex trial took about 54.7 seconds.

After the matching Harbor bootstrap/shutdown patch, the Oracle fixed-overhead
comparison changed as follows:

| Trial | Reward | Environment setup | Shutdown tail | Total |
|---|---:|---:|---:|---:|
| `ptx-add-one-oracle-fixed-sif` | 1.0 | 8.069 s | 5.168 s | 16.548 s |
| `ptx-add-one-oracle-fast-generic-sif-gpu2` | 1.0 | 4.055 s | 0.531 s | 5.867 s |

The patch skips `apt-get update` only when all required tools are already
present. Missing-package paths still refresh apt metadata before installation.
Shutdown first calls the existing Harbor server `/shutdown` endpoint, then
falls back to SIGTERM and SIGKILL if needed.

Both rows in this table use the same generic SIF, so this is the primary
before/after comparison. A separate run with the Codex-specific SIF also
returned reward `1.0` in 5.141 seconds.

## Direct-PTX images

The original two images above are retained as historical `tool-assisted`
artifacts. They include `nvcc` and must not be used for the direct-PTX
leaderboard.

The restricted definitions are:

- `ptx-bench-direct-cuda13.2.def`: agent-neutral direct-PTX image;
- `ptx-bench-direct-codex0.148.def`: direct-PTX image with pinned Codex CLI.

They derive from the already validated SIFs, remove `nvcc`, NVRTC, `cudafe++`,
and NVVM, and retain `ptxas` for direct PTX syntax/resource checks. Task
References are compiled once during task authoring and committed as hidden,
digest-pinned `.ptx` files; the verifier no longer needs a CUDA C++ compiler.

Build the Codex image on Catalyst with:

```bash
APPTAINER_CACHEDIR=/tmp/zhuominc-ptx-apptainer-cache \
APPTAINER_TMPDIR=/tmp/zhuominc-ptx-apptainer-tmp \
timeout 7200 \
apptainer build --fakeroot \
  /home/zhuominc/rockyeast-work/containers/ptx-bench-direct-cuda13.2.1-codex0.148-ubuntu24.04-v1.sif \
  /home/zhuominc/rockyeast-work/PTX-benchmark/images/ptx-bench-direct-codex0.148.def
```

Because Codex still requires network access to call the model API, removing
local compiler binaries is not a hermetic proof that no compiler was
downloaded. Direct-PTX release runs must also retain and audit the Agent
trajectory, and should add egress allowlisting when the infrastructure supports
it.

The Codex direct-PTX image was built and tested on `catalyst-b200` on
2026-08-20:

```text
Path:   /home/zhuominc/rockyeast-work/containers/ptx-bench-direct-cuda13.2.1-codex0.148-ubuntu24.04-v1.sif
Size:   3,664,666,624 bytes
SHA256: 2b38a14d218aa78cd14eba8e5b66b991cf28296f6ae697b1473c64270c41ebe9
```

Validation completed:

- image `%test` confirmed `nvcc` and NVRTC are absent;
- `ptxas` 13.2.78, Codex CLI 0.148.0, and Python 3.12.3 remain available;
- vector-add Oracle passed with correctness `1.0` and reward `0.991443`;
- RMSNorm Oracle passed with correctness `1.0` and reward `1.0`;
- a real `gpt-5.6-sol` vector-add trial passed with correctness and reward
  `1.0` (`speedup_vs_reference=1.007243`); its trajectory directly created PTX
  and used only `ptxas`/`cuobjdump`
  for validation, with no high-level compiler or compiler download.
