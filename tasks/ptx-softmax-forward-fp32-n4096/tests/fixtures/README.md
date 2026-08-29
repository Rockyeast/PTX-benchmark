# Author-side fixtures

These PTX files are hidden verifier-development fixtures, not Agent examples.

- `missing-entry.ptx`: must be rejected before launch;
- `uniform-output.ptx`: passes equal-row intuition but fails non-uniform rows;
- `partial-output.ptx`: writes only the first quarter of each row;
- `slow-correct.ptx`: serial thread-0 stable Softmax; correctness must pass and
  performance must score below the fixed Reference.
