FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PTX_ISA_URL=https://docs.nvidia.com/cuda/archive/13.2.1/pdf/ptx_isa_9.2.pdf
ARG PTX_ISA_SHA256=82bebdef912a6c5622ea1622688fdfb40949c8e12071af184dd3349042a2756c

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        poppler-utils \
        python3 \
    && install -d -m 0555 /opt/ptx-docs \
    && curl --fail --location --silent --show-error \
        "${PTX_ISA_URL}" \
        --output /opt/ptx-docs/ptx-isa-9.2.pdf \
    && echo "${PTX_ISA_SHA256}  /opt/ptx-docs/ptx-isa-9.2.pdf" \
        | sha256sum --check --strict \
    && chmod 0444 /opt/ptx-docs/ptx-isa-9.2.pdf \
    && rm -rf /var/lib/apt/lists/*

RUN rm -f \
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
    && apt-get clean

COPY PTX-ISA-SOURCE.md /opt/ptx-docs/README.md

RUN chmod 0444 /opt/ptx-docs/README.md \
    && set -eux; \
    ptxas --version; \
    cuobjdump --version; \
    ! command -v nvcc; \
    ! command -v triton; \
    test ! -e /usr/local/cuda/nvvm; \
    test -z "$(find -L /usr/local/cuda -iname '*nvrtc*' -print -quit)"; \
    test -z "$(find -L /usr/local/cuda -iname '*nvvm*' -print -quit)"; \
    python3 -c "import importlib.util; assert importlib.util.find_spec('triton') is None"; \
    test "$(stat -c '%a' /opt/ptx-docs/ptx-isa-9.2.pdf)" = 444; \
    echo "${PTX_ISA_SHA256}  /opt/ptx-docs/ptx-isa-9.2.pdf" \
        | sha256sum --check --strict

WORKDIR /app
