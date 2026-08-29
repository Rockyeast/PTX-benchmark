# Author-side verifier fixtures

`reference_variants.cu` is compiled only in the trusted authoring environment:

- `FIXTURE_NON_CAUSAL` attends to future keys and must receive correctness 0;
- `FIXTURE_PARTIAL_OUTPUT` writes only features 0 through 63 and must receive 0;
- `FIXTURE_SLOW_CORRECT` uses only thread 0 and must pass correctness while
  receiving a performance reward below the fixed Reference.

Generated PTX files retain the same entry, parameter ABI, and fixed launch ABI
as the task. They are verifier calibration evidence, not Agent-visible hints.
