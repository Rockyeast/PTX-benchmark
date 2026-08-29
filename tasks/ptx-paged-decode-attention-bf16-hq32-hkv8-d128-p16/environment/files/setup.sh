#!/bin/bash

# Harbor's Singularity backend sources environment/files/setup.sh while its
# writable overlay is active. Docker receives the same files through COPY in
# environment/Dockerfile, so both backends expose one environment contract.
set -euo pipefail

if [ -z "${HARBOR_STAGING:-}" ]; then
    printf 'HARBOR_STAGING is required for Direct-PTX environment setup\n' >&2
    return 1
fi

install -m 0555 \
    "$HARBOR_STAGING/direct-ptx-policy-check" \
    /usr/local/bin/direct-ptx-policy-check
install -m 0555 \
    "$HARBOR_STAGING/verify_hardware.py" \
    /usr/local/bin/direct-ptx-verify-hardware

rm -rf /opt/ptx-docs
install -d -m 0555 /opt/ptx-docs
cp -a "$HARBOR_STAGING/ptx-docs/." /opt/ptx-docs/
chmod -R a-w /opt/ptx-docs

/usr/local/bin/direct-ptx-policy-check
