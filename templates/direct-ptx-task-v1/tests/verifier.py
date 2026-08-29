#!/usr/bin/env python3

"""Fail-closed hidden verifier scaffold for one Direct-PTX task."""

import json
import sys
from pathlib import Path


CANDIDATE = Path("/app/kernel.ptx")
BASELINE = Path("/tests/performance_baseline.ptx")
REFERENCE = Path("/tests/performance_reference.ptx")
REWARD_DIR = Path("/logs/verifier")
ENTRY_NAME = "@@ENTRY_NAME@@"
MAX_PTX_BYTES = 1 << 20
# DIRECT_PTX_VERIFIER_UNIMPLEMENTED


class CandidateRejected(Exception):
    """The Candidate is missing, invalid, or computes a wrong result."""


def write_results(reward: float, details: dict[str, object]) -> None:
    REWARD_DIR.mkdir(parents=True, exist_ok=True)
    (REWARD_DIR / "reward.txt").write_text(f"{reward:.12g}\n")
    (REWARD_DIR / "reward.json").write_text(
        json.dumps({"reward": reward, **details}, indent=2, sort_keys=True) + "\n"
    )


def validate_ptx_file(path: Path, *, candidate: bool = False) -> str:
    if not path.is_file():
        error = f"missing PTX file: {path}"
        if candidate:
            raise CandidateRejected(error)
        raise RuntimeError(error)
    if path.stat().st_size == 0 or path.stat().st_size > MAX_PTX_BYTES:
        error = f"invalid PTX file size: {path}"
        if candidate:
            raise CandidateRejected(error)
        raise RuntimeError(error)
    source = path.read_text()
    if ".entry" not in source or ENTRY_NAME not in source:
        error = f"missing PTX entry {ENTRY_NAME!r}: {path}"
        if candidate:
            raise CandidateRejected(error)
        raise RuntimeError(error)
    return source


def verify_and_score() -> tuple[float, dict[str, object]]:
    validate_ptx_file(CANDIDATE, candidate=True)
    validate_ptx_file(BASELINE)
    validate_ptx_file(REFERENCE)
    # TODO(task author): verify the actual GPU/SM, load all three PTX files,
    # compute independent hidden truth, reject every wrong Candidate, then time
    # only correct implementations and normalize Baseline -> 0, Reference -> 1.
    raise NotImplementedError("task-specific hidden verifier is not implemented")


def main() -> int:
    try:
        reward, details = verify_and_score()
    except CandidateRejected as error:
        write_results(0.0, {"correctness": 0.0, "error": str(error)})
        return 1
    except Exception as error:
        write_results(
            0.0,
            {
                "correctness": 0.0,
                "infrastructure_error": type(error).__name__,
                "error": str(error),
            },
        )
        print(f"verifier infrastructure failure: {error}", file=sys.stderr)
        return 2
    write_results(reward, details)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
