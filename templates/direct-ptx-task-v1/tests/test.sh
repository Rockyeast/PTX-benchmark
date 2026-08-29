#!/bin/bash

set -uo pipefail

mkdir -p /logs/verifier
python3 /tests/verifier.py
status=$?

case "$status" in
    0)
        exit 0
        ;;
    1)
        printf 'Direct-PTX Candidate rejected\n' >&2
        exit 0
        ;;
    *)
        printf 'Direct-PTX verifier infrastructure failure (status %s)\n' "$status" >&2
        exit "$status"
        ;;
esac
