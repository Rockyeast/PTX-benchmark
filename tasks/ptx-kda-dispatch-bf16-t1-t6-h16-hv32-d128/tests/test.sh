#!/bin/bash

set -uo pipefail

mkdir -p /logs/verifier
printf '{"reward":0,"correctness":0,"performance":0,"speedup_vs_reference":0}\n' \
    > /logs/verifier/reward.json

# A broken benchmark environment is an infrastructure failure, not a scored
# Candidate failure.
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

if ! gcc -O3 -Wall -Wextra -Werror -shared -fPIC \
    /tests/kda_t1_truth.c -lm -o /tmp/libkda_t1_truth.so \
    > /logs/verifier/t1-truth-build.txt 2>&1; then
    printf 'T1 CPU truth helper build failed\n' >&2
    exit 2
fi

if ! gcc -O3 -Wall -Wextra -Werror -shared -fPIC \
    /tests/kda_t6_truth.c -lm -o /tmp/libkda_t6_truth.so \
    > /logs/verifier/t6-truth-build.txt 2>&1; then
    printf 'T6 CPU truth helper build failed\n' >&2
    exit 2
fi

if ! ptxas -arch=sm_100a -v -o /tmp/reference_t6.cubin \
    /tests/reference_t6.ptx > /logs/verifier/reference-t6-ptxas.txt 2>&1; then
    printf 'Fixed T6 Reference PTX assembly failed\n' >&2
    exit 2
fi

rm -f /tmp/candidate.cubin
ptxas -arch=sm_100a -v -o /tmp/candidate.cubin /app/kernel.ptx \
    > /logs/verifier/ptxas.txt 2>&1 || true

python3 /tests/verify_t1.py \
    --candidate /app/kernel.ptx \
    --reference /tests/reference_t1.ptx \
    --truth /tmp/libkda_t1_truth.so \
    --log /logs/verifier/t1-metrics.json \
    --reward /logs/verifier/t1-reward.json
t1_status=$?

case "$t1_status" in
    0)
        ;;
    1)
        cp /logs/verifier/t1-metrics.json /logs/verifier/metrics.json
        cp /logs/verifier/t1-reward.json /logs/verifier/reward.json
        printf 'Direct-PTX Candidate rejected by the T1 verifier\n' >&2
        exit 0
        ;;
    *)
        printf 'T1 verifier infrastructure failure (status %s)\n' "$t1_status" >&2
        exit "$t1_status"
        ;;
esac

python3 /tests/verify_t6.py \
    --candidate /app/kernel.ptx \
    --candidate-image /tmp/candidate.cubin \
    --reference /tests/reference_t6.ptx \
    --reference-image /tmp/reference_t6.cubin \
    --truth /tmp/libkda_t6_truth.so \
    --log /logs/verifier/t6-metrics.json \
    --reward /logs/verifier/t6-reward.json
t6_status=$?

case "$t6_status" in
    0)
        ;;
    1)
        cp /logs/verifier/t6-metrics.json /logs/verifier/metrics.json
        cp /logs/verifier/t6-reward.json /logs/verifier/reward.json
        printf 'Direct-PTX Candidate rejected by the T6 verifier\n' >&2
        exit 0
        ;;
    *)
        printf 'T6 verifier infrastructure failure (status %s)\n' "$t6_status" >&2
        exit "$t6_status"
        ;;
esac

python3 /tests/verify_and_benchmark.py \
    --t1-metrics /logs/verifier/t1-metrics.json \
    --t1-reward /logs/verifier/t1-reward.json \
    --t6-metrics /logs/verifier/t6-metrics.json \
    --t6-reward /logs/verifier/t6-reward.json \
    --metrics /logs/verifier/metrics.json \
    --reward /logs/verifier/reward.json
aggregate_status=$?

case "$aggregate_status" in
    0)
        exit 0
        ;;
    1)
        printf 'Direct-PTX Candidate rejected by the joint correctness gate\n' >&2
        exit 0
        ;;
    *)
        printf 'Result aggregation infrastructure failure (status %s)\n' "$aggregate_status" >&2
        exit "$aggregate_status"
        ;;
esac
