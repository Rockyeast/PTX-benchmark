#!/bin/bash

set -uo pipefail

if python3 /tests/verify_ptx.py /app/kernel.ptx; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
