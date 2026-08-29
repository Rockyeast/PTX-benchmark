# Negative fixtures

- `ignore-page-table.ptx`: treats logical pages as physical pages.
- `only-dq.ptx`: omits both KV-cache gradients.
- `missing-main.ptx`: omits the required main entry.

The first two are author-built from `../reference.cu` with the corresponding
preprocessor definition. They must assemble successfully and receive reward 0.
