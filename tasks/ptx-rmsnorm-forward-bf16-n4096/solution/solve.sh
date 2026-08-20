#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_path="${OUTPUT_PATH:-/app/kernel.ptx}"

cp "$script_dir/oracle.ptx" "$output_path"
