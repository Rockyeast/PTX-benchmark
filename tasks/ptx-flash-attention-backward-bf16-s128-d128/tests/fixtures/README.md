# Negative fixtures

- `only-dq.ptx`: computes dQ but omits dK/dV;
- `depends-on-prezero.ptx`: accumulates dK/dV without initializing them.

Both must assemble and load, then receive correctness zero. They are compiled
from `reference_variants.cu` with `-DONLY_DQ` or `-DDEPEND_ON_PREZERO`.
