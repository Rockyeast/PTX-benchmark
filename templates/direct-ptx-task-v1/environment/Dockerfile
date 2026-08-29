FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PTX_ISA_URL=https://docs.nvidia.com/cuda/archive/13.2.1/pdf/ptx_isa_9.2.pdf

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl poppler-utils python3 \
    && install -d -m 0555 /opt/ptx-docs \
    && curl -fsSL "$PTX_ISA_URL" -o /opt/ptx-docs/ptx-isa-9.2.pdf \
    && chmod 0444 /opt/ptx-docs/ptx-isa-9.2.pdf \
    && rm -f \
        /usr/local/cuda/bin/__nvcc_device_query \
        /usr/local/cuda/bin/cudafe++ \
        /usr/local/cuda/bin/nvcc \
        /usr/local/cuda/bin/nvcc.profile \
        /usr/local/cuda/targets/x86_64-linux/include/nvrtc*.h \
    && find -L /usr/local/cuda -type f -iname '*nvrtc*' -delete \
    && find -L /usr/local/cuda -type l -iname '*nvrtc*' -delete \
    && rm -rf /usr/local/cuda/nvvm \
    && find -L /usr/local/cuda -type f -iname '*nvvm*' -delete \
    && find -L /usr/local/cuda -type l -iname '*nvvm*' -delete \
    && command -v ptxas >/dev/null \
    && command -v cuobjdump >/dev/null \
    && ! command -v nvcc >/dev/null 2>&1 \
    && python3 -c 'import importlib.util,sys;sys.exit(importlib.util.find_spec("triton") is not None)' \
    && rm -rf /var/lib/apt/lists/*

COPY kernel.ptx reference.py measure.py /app/

WORKDIR /app
