#!/bin/bash

set -uo pipefail

mkdir -p /logs/verifier
printf '{"reward":0,"correctness":0,"performance":0,"speedup_vs_reference":0}\n' \
    > /logs/verifier/reward.json

if ! /tests/direct-ptx-policy-check \
    --trusted-manifest /tests/ptx-docs-manifest.toml \
    > /logs/verifier/environment-policy.log 2>&1; then
    printf 'Direct-PTX environment policy check failed\n' >&2
    exit 1
fi

if ! python3 /tests/verify_hardware.py \
    --output /logs/verifier/hardware.json; then
    printf 'Direct-PTX hardware validation failed\n' >&2
    exit 1
fi

ptxas -arch=sm_100 -o /tmp/candidate.cubin /app/kernel.ptx \
    > /logs/verifier/ptxas.txt 2>&1 || true

python3 /tests/verify_and_benchmark.py \
    --candidate /app/kernel.ptx \
    --reference /tests/reference.ptx \
    --log /logs/verifier/metrics.json \
    --reward /logs/verifier/reward.json
status=$?

case "$status" in
    0)
        exit 0
        ;;
    1)
        printf 'RoPE Candidate rejected\n' >&2
        exit 0
        ;;
    *)
        printf 'RoPE verifier infrastructure failure (status %s)\n' "$status" >&2
        exit "$status"
        ;;
esac
