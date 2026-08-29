FROM nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:44a9504c6dfb50b1241464241b02a93871928f373de6f5a644cf5fe9f080aa63

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils python3 \
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
    && rm -rf /var/lib/apt/lists/*

COPY files/ptx-docs/ /opt/ptx-docs/
COPY files/direct-ptx-policy-check /usr/local/bin/direct-ptx-policy-check
COPY files/verify_hardware.py /usr/local/bin/direct-ptx-verify-hardware
COPY files/benchmark_candidate.py /usr/local/bin/direct-ptx-benchmark-candidate

RUN chmod 0555 /usr/local/bin/direct-ptx-policy-check \
    && chmod 0555 /usr/local/bin/direct-ptx-verify-hardware \
    && chmod 0555 /usr/local/bin/direct-ptx-benchmark-candidate \
    && chmod -R a-w /opt/ptx-docs \
    && /usr/local/bin/direct-ptx-policy-check

WORKDIR /app
