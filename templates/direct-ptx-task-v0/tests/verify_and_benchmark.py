#!/usr/bin/env python3

"""Fail-closed verifier scaffold for a direct-PTX task.

Replace the task-specific TODO before running Oracle or a real agent. Keep the
generic safety checks and preserve correctness as a hard gate before timing.
"""

import argparse
import json
import sys
from pathlib import Path


ENTRY_NAME = "@@ENTRY_NAME@@"
MAX_PTX_BYTES = 1 << 20


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    return parser.parse_args()


def validate_submission_file(candidate: Path) -> str:
    if not candidate.is_file():
        raise FileNotFoundError(f"missing PTX submission: {candidate}")
    size = candidate.stat().st_size
    if size == 0:
        raise ValueError("PTX submission is empty")
    if size > MAX_PTX_BYTES:
        raise ValueError(
            f"PTX submission is {size} bytes; limit is {MAX_PTX_BYTES} bytes"
        )
    source = candidate.read_text()
    if ".entry" not in source or ENTRY_NAME not in source:
        raise ValueError(f"missing visible PTX entry {ENTRY_NAME!r}")
    return source


def verify_task_specific_correctness(_source: str) -> None:
    # TODO(task author): load PTX with the CUDA Driver API, execute independent
    # hidden cases, compare against a truth source that does not depend on the
    # Candidate or Oracle, and check memory canaries where applicable.
    raise NotImplementedError("task-specific correctness verifier is not implemented")


def benchmark_after_correctness() -> dict[str, float]:
    # TODO(task author): warm up Candidate and fixed same-ABI Reference, time
    # both on the same GPU, retain raw samples, and compute a documented score.
    raise NotImplementedError("task-specific performance protocol is not implemented")


def main() -> int:
    args = parse_args()
    reward: dict[str, float] = {
        "reward": 0.0,
        "correctness": 0.0,
        "performance": 0.0,
        "speedup_vs_reference": 0.0,
    }
    metrics: dict[str, object] = {
        "template_version": "direct-ptx-v0",
        "hardware_target": @@HARDWARE_TARGET_PY@@,
        "entry": ENTRY_NAME,
        "status": "failed",
    }

    try:
        source = validate_submission_file(args.candidate)
        verify_task_specific_correctness(source)
        reward["correctness"] = 1.0
        performance = benchmark_after_correctness()
        reward.update(performance)
        metrics["status"] = "passed"
        metrics["performance"] = performance
    except Exception as error:  # A bad submission receives zero with evidence.
        metrics["error_type"] = type(error).__name__
        metrics["error"] = str(error)
        print(f"PTX verification failed: {type(error).__name__}: {error}", file=sys.stderr)

    write_json(args.log, metrics)
    write_json(args.reward, reward)
    return 0 if reward["correctness"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
