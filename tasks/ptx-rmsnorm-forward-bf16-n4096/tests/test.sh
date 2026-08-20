#!/bin/bash

set -uo pipefail

mkdir -p /logs/verifier
printf '{"reward":0,"correctness":0,"performance":0,"speedup_vs_reference":0}\n' \
  > /logs/verifier/reward.json

if python3 /tests/verify_and_benchmark.py \
  --candidate /app/kernel.ptx \
  --reference-source /tests/reference.cu \
  --log /logs/verifier/metrics.json \
  --reward /logs/verifier/reward.json; then
  exit 0
else
  status=$?
fi

printf 'RMSNorm feasibility verifier failed with status %s\n' "$status" >&2
exit 0
