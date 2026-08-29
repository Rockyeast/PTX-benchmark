// Compile tests/reference.cu with one of these author-side definitions:
//   -DKDA_WRONG_HEAD_MAP
//   -DKDA_WRONG_IGNORE_BETA
//   -DKDA_WRONG_OUTPUT_OLD_STATE
//   -DKDA_WRONG_OUTPUT_PRE_ROUND
//   -DKDA_SLOW_CORRECT
//
// This provenance note is intentionally not a build-time dependency. The
// generated PTX fixtures are hidden from the Agent and retained beside it.
