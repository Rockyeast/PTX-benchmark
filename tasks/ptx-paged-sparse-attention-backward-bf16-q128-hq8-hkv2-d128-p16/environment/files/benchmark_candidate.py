#!/usr/bin/env python3
"""Public Candidate-only assembly check; hidden correctness remains in /tests."""

import subprocess
from pathlib import Path

path = Path("/app/kernel.ptx")
if not path.is_file():
    raise SystemExit("missing /app/kernel.ptx")
source = path.read_text()
entries = (
    "paged_sparse_attn_bwd_zero_bf16_q128_hq8_hkv2_d128_p16",
    "paged_sparse_attn_bwd_main_bf16_q128_hq8_hkv2_d128_p16",
)
for entry in entries:
    if entry not in source:
        raise SystemExit(f"missing required entry: {entry}")
subprocess.run(
    ["ptxas", "-arch=sm_100a", "-v", "-o", "/tmp/public-candidate.cubin", str(path)],
    check=True,
)
print("Candidate PTX contains both entries and assembles for sm_100a")
