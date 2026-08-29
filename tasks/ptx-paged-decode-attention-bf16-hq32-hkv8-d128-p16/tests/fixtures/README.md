# Author-side paged decode fixtures

- `contiguous-pages.ptx` ignores the logical-to-physical page table;
- `wrong-gqa.ptx` maps query heads with `hq % 8` instead of `hq / 4`;
- `round-last-page.ptx` reads all 16 slots of the final partial page;
- `slow-correct.ptx` serializes all work onto thread 0 but preserves semantics.

All are compiled in the trusted author environment with the task entry and ABI.
