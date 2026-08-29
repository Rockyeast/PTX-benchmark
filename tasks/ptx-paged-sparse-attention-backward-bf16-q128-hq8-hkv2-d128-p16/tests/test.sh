#!/bin/bash

set -uo pipefail

mkdir -p /logs/verifier
printf '{"reward":0,"correctness":0,"performance":0}\n' > /logs/verifier/reward.json

if ! /tests/direct-ptx-policy-check --trusted-manifest /tests/ptx-docs-manifest.toml \
    > /logs/verifier/environment-policy.log 2>&1; then
    printf 'Direct-PTX environment policy check failed\n' >&2
    exit 1
fi

if ! python3 /tests/verify_hardware.py --output /logs/verifier/hardware.json; then
    printf 'Direct-PTX hardware validation failed\n' >&2
    exit 1
fi

if ! gcc -O3 -Wall -Wextra -Werror -shared -fPIC \
    /tests/paged_sparse_bwd_truth.c -lm -o /tmp/libpaged_sparse_bwd_truth.so \
    > /logs/verifier/truth-build.txt 2>&1; then
    printf 'CPU truth helper build failed\n' >&2
    exit 2
fi

ptxas -arch=sm_100a -v -o /tmp/candidate.cubin /app/kernel.ptx \
    > /logs/verifier/ptxas.txt 2>&1 || true

python3 /tests/verify_and_benchmark.py \
    --candidate /app/kernel.ptx \
    --reference /tests/reference.ptx \
    --truth /tmp/libpaged_sparse_bwd_truth.so \
    --log /logs/verifier/metrics.json \
    --reward /logs/verifier/reward.json
status=$?

case "$status" in
    0) exit 0 ;;
    1)
        printf 'Direct-PTX Candidate rejected\n' >&2
        exit 0
        ;;
    *)
        printf 'Direct-PTX verifier infrastructure failure (status %s)\n' "$status" >&2
        exit "$status"
        ;;
esac
