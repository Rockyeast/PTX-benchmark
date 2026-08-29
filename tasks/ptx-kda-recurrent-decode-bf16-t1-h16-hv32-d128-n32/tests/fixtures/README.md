# Author-side KDA fixtures

All fixtures retain the task entry and fixed launch ABI. They are generated
from `../reference.cu` with CUDA 13.2.1 and are hidden from the Agent.

- `wrong-head-map.ptx`: maps `hv` directly to a query head instead of `hv/2`;
- `ignore-beta.ptx`: behaves as if every beta value were one;
- `output-before-update.ptx`: projects the decayed old state instead of the
  updated state;
- `output-pre-round.ptx`: diagnostic only; projects the FP32 update instead of
  the BF16-rounded state. Its error remains inside the initial public tolerance,
  so it is not counted as a rejecting negative fixture;
- `slow-correct.ptx`: preserves correctness and adds a deterministic delay.

The validation record must retain exact commands and hashes after generation.
