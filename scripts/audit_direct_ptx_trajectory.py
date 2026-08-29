#!/usr/bin/env python3
"""Fail when a Harbor Agent trajectory contains a model-side web-search call."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


WEB_TOOL_NAMES = {"web_search", "web.run", "web__run", "web__run"}
WRAPPED_WEB_CALL = re.compile(
    r"(?:\btools\s*\.\s*web__run|\bweb\s*\.\s*run|\bweb_search)\s*\("
)


@dataclass(frozen=True)
class Violation:
    step_id: str
    function_name: str
    evidence: str


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)


def _shorten(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


def audit_trajectory(data: Any) -> list[Violation]:
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        raise ValueError("trajectory must be an object containing a steps list")

    violations: list[Violation] = []
    for index, step in enumerate(data["steps"]):
        if not isinstance(step, dict):
            continue
        step_id = str(step.get("step_id", index))
        tool_calls = step.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            continue

        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function_name = str(call.get("function_name", ""))
            normalized_name = function_name.lower().replace("-", "_")
            strings = list(_iter_strings(call.get("arguments", {})))
            extra = step.get("extra")
            if isinstance(extra, dict):
                strings.extend(_iter_strings(extra.get("raw_arguments")))

            if normalized_name in WEB_TOOL_NAMES:
                violations.append(
                    Violation(step_id, function_name, _shorten(" ".join(strings)))
                )
                continue

            wrapped_call = next(
                (text for text in strings if WRAPPED_WEB_CALL.search(text)), None
            )
            if wrapped_call is not None:
                violations.append(
                    Violation(step_id, function_name, _shorten(wrapped_call))
                )

    return violations


def _trajectory_paths(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for input_path in inputs:
        if input_path.is_file():
            paths.add(input_path)
        elif input_path.is_dir():
            paths.update(input_path.rglob("trajectory.json"))
        else:
            raise FileNotFoundError(input_path)
    return sorted(paths)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Harbor trajectory.json files for Codex/model-side web search. "
            "This is separate from container egress auditing."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path, help="trajectory file or Job directory")
    parser.add_argument(
        "--policy",
        choices=["strict-offline", "online-docs"],
        default="strict-offline",
        help=(
            "strict-offline fails on web search; online-docs records web search "
            "without treating it as a policy violation"
        ),
    )
    args = parser.parse_args()

    try:
        paths = _trajectory_paths(args.paths)
    except FileNotFoundError as exc:
        print(f"ERROR missing path: {exc}", file=sys.stderr)
        return 2
    if not paths:
        print("ERROR no trajectory.json files found", file=sys.stderr)
        return 2

    failed_paths = 0
    error_paths = 0
    web_calls = 0
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            violations = audit_trajectory(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"ERROR {path}: {exc}", file=sys.stderr)
            error_paths += 1
            continue

        if not violations:
            print(f"PASS {path}: no model-side web-search calls")
            continue

        web_calls += len(violations)
        if args.policy == "strict-offline":
            failed_paths += 1
            label = "FAIL"
        else:
            label = "INFO"
        print(f"{label} {path}: {len(violations)} model-side web-search call(s)")
        for violation in violations:
            print(
                f"  step={violation.step_id} tool={violation.function_name or '<unknown>'} "
                f"evidence={violation.evidence}"
            )

    print(
        f"SUMMARY trajectories={len(paths)} violations={failed_paths} "
        f"web_calls={web_calls} parse_errors={error_paths}"
    )
    if error_paths:
        return 2
    return 1 if failed_paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
