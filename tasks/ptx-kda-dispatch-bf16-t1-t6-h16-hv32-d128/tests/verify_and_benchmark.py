#!/usr/bin/env python3
"""Aggregate T1 and T6 verifier outputs with a joint correctness gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1-metrics", required=True, type=Path)
    parser.add_argument("--t1-reward", required=True, type=Path)
    parser.add_argument("--t6-metrics", required=True, type=Path)
    parser.add_argument("--t6-reward", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--reward", required=True, type=Path)
    args = parser.parse_args()

    t1_metrics = json.loads(args.t1_metrics.read_text())
    t6_metrics = json.loads(args.t6_metrics.read_text())
    t1_reward = json.loads(args.t1_reward.read_text())
    t6_reward = json.loads(args.t6_reward.read_text())
    correctness = min(float(t1_reward["correctness"]), float(t6_reward["correctness"]))
    raw_t1 = float(t1_reward["speedup_vs_reference"])
    raw_t6 = float(t6_reward["speedup_vs_reference"])
    raw_speedup = math.sqrt(max(0.0, raw_t1) * max(0.0, raw_t6)) if correctness else 0.0
    performance = min(1.0, raw_speedup) if correctness else 0.0

    metrics = {
        "status": "passed" if correctness else "candidate_rejected",
        "format": "multi-entry-system",
        "entries": {"t1": t1_metrics, "t6": t6_metrics},
        "entry_speedups": {"t1": raw_t1, "t6": raw_t6},
        "raw_speedup_vs_reference": raw_speedup,
    }
    reward = {
        "correctness": correctness,
        "performance": performance,
        "speedup_vs_reference": raw_speedup,
        "reward": correctness * performance,
    }
    write_json(args.metrics, metrics)
    write_json(args.reward, reward)
    return 0 if correctness else 1


if __name__ == "__main__":
    raise SystemExit(main())
