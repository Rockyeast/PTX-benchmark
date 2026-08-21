#!/usr/bin/env python3

"""Check structural and release invariants for a direct-PTX Harbor task."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tomllib
from pathlib import Path


REQUIRED_FILES = (
    "README.md",
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "environment/policy-check.sh",
    "environment/ptx-docs/manifest.toml",
    "solution/solve.sh",
    "tests/test.sh",
    "tests/verify_and_benchmark.py",
)
SHA256 = re.compile(r"[0-9a-f]{64}$")
NETWORK_MODES = {"public", "allowlist", "no-network"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a direct-PTX task scaffold or release candidate."
    )
    parser.add_argument("task", type=Path)
    parser.add_argument(
        "--release",
        action="store_true",
        help=(
            "Require completed TODOs, restricted effective Agent networking, "
            "pinned docs, and release_ready=true"
        ),
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        try:
            path.read_text()
        except UnicodeDecodeError:
            continue
        files.append(path)
    return files


def validate_network_policy(
    table: object,
    scope: str,
    errors: list[str],
    *,
    require_mode: bool,
) -> tuple[str | None, list[str] | None]:
    if not isinstance(table, dict):
        errors.append(f"missing [{scope}] table")
        return None, None

    mode = table.get("network_mode")
    hosts = table.get("allowed_hosts")
    if mode is None and not require_mode:
        if hosts is not None:
            errors.append(
                f"{scope}.allowed_hosts requires an explicit allowlist network mode"
            )
        return None, None
    if mode not in NETWORK_MODES:
        errors.append(f"{scope}.network_mode must be explicitly set")
        return None, None
    if hosts is not None and not isinstance(hosts, list):
        errors.append(f"{scope}.allowed_hosts must be a list")
        return mode, None
    if mode == "allowlist" and not hosts:
        errors.append(f"{scope} allowlist network mode requires allowed_hosts")
    if mode != "allowlist" and hosts:
        errors.append(
            f"{scope}.allowed_hosts is only valid for allowlist network mode"
        )
    return mode, hosts


def validate_document_snapshot(root: Path, errors: list[str]) -> None:
    docs_root = root / "environment" / "ptx-docs"
    manifest_path = docs_root / "manifest.toml"
    try:
        manifest = load_toml(manifest_path)
    except (OSError, tomllib.TOMLDecodeError) as error:
        errors.append(f"invalid PTX documentation manifest: {error}")
        return

    required = ("document_path", "document_version", "source_url", "sha256", "license")
    for field in required:
        value = manifest.get(field)
        if not isinstance(value, str) or not value.strip() or "TODO" in value:
            errors.append(f"PTX documentation manifest field is incomplete: {field}")

    relative = manifest.get("document_path")
    expected_hash = manifest.get("sha256")
    if not isinstance(relative, str) or "TODO" in relative:
        return
    document = (docs_root / relative).resolve()
    try:
        document.relative_to(docs_root.resolve())
    except ValueError:
        errors.append("PTX document_path escapes environment/ptx-docs")
        return
    if not document.is_file():
        errors.append(f"missing pinned PTX documentation file: {relative}")
        return
    if not isinstance(expected_hash, str) or not SHA256.fullmatch(expected_hash):
        errors.append("PTX documentation sha256 must be 64 lowercase hexadecimal characters")
        return
    actual_hash = hashlib.sha256(document.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        errors.append(
            f"PTX documentation SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )


def main() -> int:
    args = parse_args()
    root = args.task.resolve()
    errors: list[str] = []

    if not root.is_dir():
        print(f"error: task directory does not exist: {root}", file=sys.stderr)
        return 2

    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    try:
        task = load_toml(root / "task.toml")
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"error: invalid task.toml: {error}", file=sys.stderr)
        return 1

    metadata = task.get("metadata", {})
    environment = task.get("environment", {})
    agent = task.get("agent", {})
    if not isinstance(metadata, dict) or metadata.get("benchmark_setting") != "direct-ptx":
        errors.append("metadata.benchmark_setting must be direct-ptx")
    environment_mode, _ = validate_network_policy(
        environment,
        "environment",
        errors,
        require_mode=True,
    )
    agent_mode, _ = validate_network_policy(
        agent,
        "agent",
        errors,
        require_mode=False,
    )
    effective_agent_mode = agent_mode or environment_mode

    for path in text_files(root):
        content = path.read_text()
        if "@@" in content:
            errors.append(f"unresolved template marker in {path.relative_to(root)}")
        if args.release and "TODO" in content:
            errors.append(f"unresolved TODO in {path.relative_to(root)}")

    if args.release:
        if effective_agent_mode == "public":
            errors.append(
                "release tasks may not use an effective public Agent network policy"
            )
        if not isinstance(metadata, dict) or metadata.get("release_ready") is not True:
            errors.append("metadata.release_ready must be true for --release")
        validate_document_snapshot(root, errors)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    mode = "release" if args.release else "scaffold"
    print(f"Direct-PTX {mode} check passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
