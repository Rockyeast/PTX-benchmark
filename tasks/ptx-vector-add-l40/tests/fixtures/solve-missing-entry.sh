#!/bin/bash

set -euo pipefail

cat > /app/kernel.ptx <<'PTX'
.version 9.2
.target sm_89
.address_size 64

.visible .entry wrong_name()
{
    ret;
}
PTX
