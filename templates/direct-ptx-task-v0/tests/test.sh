#!/bin/bash

set -uo pipefail

mkdir -p /logs/verifier
printf '{"reward":0,"correctness":0,"performance":0,"speedup_vs_reference":0}\n' \
    > /logs/verifier/reward.json

# An invalid benchmark environment is infrastructure failure, not a Candidate
# failure. Do not silently turn a leaked compiler or missing PTX tool into a
# model score of zero.
if ! /usr/local/bin/direct-ptx-policy-check \
    > /logs/verifier/environment-policy.log 2>&1; then
    printf 'Direct-PTX environment policy check failed\n' >&2
    exit 1
fi

if python3 /tests/verify_and_benchmark.py \
    --candidate /app/kernel.ptx \
    --log /logs/verifier/metrics.json \
    --reward /logs/verifier/reward.json; then
    exit 0
else
    status=$?
fi

printf 'Direct-PTX verifier failed with status %s\n' "$status" >&2
# Harbor reads reward.json. A rejected candidate is a scored zero, not a
# verifier infrastructure exception.
exit 0
