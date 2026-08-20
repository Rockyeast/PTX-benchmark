FROM nvidia/cuda:13.2.1-devel-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -f \
        /usr/local/cuda/bin/__nvcc_device_query \
        /usr/local/cuda/bin/cudafe++ \
        /usr/local/cuda/bin/nvcc \
        /usr/local/cuda/bin/nvcc.profile \
        /usr/local/cuda/targets/x86_64-linux/include/nvrtc.h \
    && rm -f /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc* \
    && rm -rf /usr/local/cuda/nvvm \
    && rm -rf /var/lib/apt/lists/*

RUN ! command -v nvcc && ptxas --version

WORKDIR /app
