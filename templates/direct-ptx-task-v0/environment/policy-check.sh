#!/bin/bash

set -euo pipefail

command -v ptxas >/dev/null
command -v cuobjdump >/dev/null

for prohibited_command in nvcc triton; do
    if command -v "$prohibited_command" >/dev/null 2>&1; then
        printf 'prohibited command is available: %s\n' "$prohibited_command" >&2
        exit 1
    fi
done

if [ -e /usr/local/cuda/nvvm ]; then
    printf 'prohibited NVVM directory is present\n' >&2
    exit 1
fi

if find /usr/local/cuda -iname '*nvrtc*' -print -quit | grep -q .; then
    printf 'prohibited NVRTC component is present\n' >&2
    exit 1
fi

if find /usr/local/cuda -iname '*nvvm*' -print -quit | grep -q .; then
    printf 'prohibited NVVM component is present\n' >&2
    exit 1
fi

python3 -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("triton") is not None)'
test -f /opt/ptx-docs/manifest.toml

printf 'Direct-PTX environment policy check passed\n'
