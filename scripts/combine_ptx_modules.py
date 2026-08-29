#!/usr/bin/env python3
"""Combine self-contained PTX files into one module with multiple entries."""

from __future__ import annotations

import argparse
from pathlib import Path


def module_body(source: str, path: Path) -> str:
    lines = source.splitlines()
    address_line = next(
        (index for index, line in enumerate(lines) if line.strip().startswith(".address_size")),
        None,
    )
    if address_line is None:
        raise ValueError(f"{path}: missing .address_size directive")
    return "\n".join(lines[address_line + 1 :]).lstrip("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("at least two PTX inputs are required")

    first = args.inputs[0].read_text(encoding="utf-8").rstrip()
    bodies = [module_body(path.read_text(encoding="utf-8"), path) for path in args.inputs[1:]]
    combined = first + "\n\n// ---- combined PTX entry ----\n\n" + "\n\n".join(bodies) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(combined, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
