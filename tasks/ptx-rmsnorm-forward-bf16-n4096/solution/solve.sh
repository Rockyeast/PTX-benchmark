#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_path="${OUTPUT_PATH:-/app/kernel.ptx}"

nvcc \
  --ptx \
  --optimize=3 \
  --gpu-architecture=compute_100 \
  --output-file "$output_path" \
  "$script_dir/oracle.cu"
