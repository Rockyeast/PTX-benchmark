#!/bin/bash

set -uo pipefail

mkdir -p /logs/verifier

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

if python3 /tests/verify_ptx.py /app/kernel.ptx; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
