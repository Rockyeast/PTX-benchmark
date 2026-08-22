# Task-pinned PTX documentation

TODO(task author): replace this placeholder with the fixed PTX ISA reference
available to the Agent. Do not commit or redistribute NVIDIA documentation
until its license has been reviewed. A reproducible alternative is to fetch a
specific official artifact while building the image and verify its SHA-256.

Complete `manifest.toml` and keep the runtime copy under `/opt/ptx-docs`
read-only. The task instruction may point the Agent to that directory, but it
must not reveal task-specific solution code or hidden verifier cases.
