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

ptxas_log=/logs/verifier/ptxas.txt
candidate_cubin=/tmp/candidate.cubin
if test -f /app/kernel.ptx; then
    ptxas -arch=sm_89 -o "$candidate_cubin" /app/kernel.ptx \
        > "$ptxas_log" 2>&1
    ptxas_status=$?
    printf '\nPTXAS_EXIT_STATUS=%s\n' "$ptxas_status" >> "$ptxas_log"
    if test "$ptxas_status" -eq 0; then
        cuobjdump --dump-resource-usage "$candidate_cubin" \
            >> "$ptxas_log" 2>&1 || true
    fi
else
    printf 'Candidate /app/kernel.ptx is missing.\nPTXAS_EXIT_STATUS=missing\n' \
        > "$ptxas_log"
fi

if python3 /tests/verify_and_benchmark.py \
    --candidate /app/kernel.ptx \
    --reference /tests/reference.ptx \
    --log /logs/verifier/metrics.json \
    --reward /logs/verifier/reward.json; then
    exit 0
else
    status=$?
fi

printf 'PTX verifier failed with status %s\n' "$status" >&2
exit 0
