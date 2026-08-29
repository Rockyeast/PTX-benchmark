# Negative fixtures

`first-block-only.ptx` is a valid same-entry PTX module that uses only the first
selected block from each CSR row. It must pass assembly/module loading and fail
hidden rows containing more than one selected block.
