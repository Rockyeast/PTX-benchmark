#!/usr/bin/env python3

"""Check structural and release invariants for a direct-PTX Harbor task."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path


V1_REQUIRED_FILES = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "environment/docker-compose.yaml",
    "environment/kernel.ptx",
    "environment/reference.py",
    "environment/measure.py",
    "solution/solve.sh",
    "solution/performance_reference.ptx",
    "tests/test.sh",
    "tests/verifier.py",
    "tests/performance_baseline.ptx",
    "tests/performance_reference.ptx",
)
FROM_INSTRUCTION = re.compile(r"^\s*FROM\s+.+$", re.IGNORECASE | re.MULTILINE)
NETWORK_MODES = {"public", "allowlist", "no-network"}
V1_UNIMPLEMENTED_MARKERS = (
    "DIRECT_PTX_REFERENCE_UNIMPLEMENTED",
    "DIRECT_PTX_MEASURE_UNIMPLEMENTED",
    "DIRECT_PTX_VERIFIER_UNIMPLEMENTED",
    "DIRECT_PTX_PERFORMANCE_BASELINE_UNIMPLEMENTED",
    "DIRECT_PTX_PERFORMANCE_REFERENCE_UNIMPLEMENTED",
)


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


def validate_kernel_artifact(task: dict[str, object], errors: list[str]) -> None:
    artifacts = task.get("artifacts")
    if not isinstance(artifacts, list):
        errors.append("task artifacts must be a list")
        return
    for artifact in artifacts:
        if isinstance(artifact, str) and artifact == "/app/kernel.ptx":
            return
        if (
            isinstance(artifact, dict)
            and artifact.get("source") == "/app/kernel.ptx"
            and artifact.get("destination") == "kernel.ptx"
        ):
            return
    errors.append(
        "task artifacts must collect /app/kernel.ptx as destination kernel.ptx"
    )


def validate_v1_structure(root: Path, errors: list[str]) -> None:
    dockerfile = (root / "environment" / "Dockerfile").read_text()
    for public_file in ("kernel.ptx", "reference.py", "measure.py"):
        if public_file not in dockerfile:
            errors.append(
                f"environment/Dockerfile must copy {public_file} into /app"
            )
    for required_text in ("ptxas", "cuobjdump", "nvcc", "nvrtc", "nvvm"):
        if required_text.casefold() not in dockerfile.casefold():
            errors.append(
                f"environment/Dockerfile must enforce the Direct-PTX tool policy for {required_text}"
            )

    instruction = (root / "instruction.md").read_text()
    for required_text in ("/app/kernel.ptx", "/app/reference.py", "/app/measure.py"):
        if required_text not in instruction:
            errors.append(f"instruction.md must mention {required_text}")

    solve = (root / "solution" / "solve.sh").read_text()
    if "/solution/performance_reference.ptx" not in solve or "/app/kernel.ptx" not in solve:
        errors.append(
            "solution/solve.sh must copy performance_reference.ptx to /app/kernel.ptx"
        )

    test_entry = (root / "tests" / "test.sh").read_text()
    if "/tests/verifier.py" not in test_entry:
        errors.append("tests/test.sh must run /tests/verifier.py")

    solution_reference = root / "solution" / "performance_reference.ptx"
    hidden_reference = root / "tests" / "performance_reference.ptx"
    if solution_reference.read_bytes() != hidden_reference.read_bytes():
        errors.append(
            "solution and hidden performance_reference.ptx files must match"
        )


def validate_v1_release_implementation(root: Path, errors: list[str]) -> None:
    for path in (
        root / "environment" / "reference.py",
        root / "environment" / "measure.py",
        root / "tests" / "verifier.py",
        root / "tests" / "performance_baseline.ptx",
        root / "tests" / "performance_reference.ptx",
        root / "solution" / "performance_reference.ptx",
    ):
        content = path.read_text()
        for marker in V1_UNIMPLEMENTED_MARKERS:
            if marker in content:
                errors.append(
                    f"unimplemented v1 marker remains in {path.relative_to(root)}"
                )
        if "NotImplementedError" in content:
            errors.append(f"unimplemented code remains in {path.relative_to(root)}")

    baseline = (root / "tests" / "performance_baseline.ptx").read_bytes()
    reference = (root / "tests" / "performance_reference.ptx").read_bytes()
    starter = (root / "environment" / "kernel.ptx").read_bytes()
    if baseline == reference:
        errors.append("performance Baseline and Reference must not be identical")
    if starter == reference:
        errors.append("Candidate starter and performance Reference must not be identical")

    dockerfile = (root / "environment" / "Dockerfile").read_text()
    if not FROM_INSTRUCTION.search(dockerfile):
        errors.append("environment/Dockerfile must contain a FROM instruction")


def main() -> int:
    args = parse_args()
    root = args.task.resolve()
    errors: list[str] = []

    if not root.is_dir():
        print(f"error: task directory does not exist: {root}", file=sys.stderr)
        return 2

    try:
        task = load_toml(root / "task.toml")
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"error: invalid task.toml: {error}", file=sys.stderr)
        return 1

    metadata = task.get("metadata", {})
    environment = task.get("environment", {})
    agent = task.get("agent", {})
    template_version = (
        metadata.get("template_version") if isinstance(metadata, dict) else None
    )
    if template_version != "direct-ptx-v1":
        print(
            "error: unsupported template_version; migrate the task to direct-ptx-v1",
            file=sys.stderr,
        )
        return 1
    for relative in V1_REQUIRED_FILES:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    validate_kernel_artifact(task, errors)
    validate_v1_structure(root, errors)
    if not isinstance(metadata, dict) or metadata.get("benchmark_setting") != "direct-ptx":
        errors.append("metadata.benchmark_setting must be direct-ptx")
    if not isinstance(metadata, dict) or metadata.get("required_gpus") != 1:
        errors.append("metadata.required_gpus must be 1")
    if not isinstance(metadata, dict) or not metadata.get("required_gpu_type"):
        errors.append("metadata.required_gpu_type must be set")
    if isinstance(environment, dict) and (
        "gpus" in environment or "gpu_types" in environment
    ):
        errors.append(
            "GPU scheduling is runtime-backend-specific; keep gpus/gpu_types out "
            "of [environment] and record the requirement in [metadata]"
        )
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
        if (
            not isinstance(metadata, dict)
            or metadata.get("implementation_status") != "implemented"
        ):
            errors.append(
                "metadata.implementation_status must be implemented for --release"
            )
        validate_v1_release_implementation(root, errors)

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    mode = "release" if args.release else "scaffold"
    print(f"Direct-PTX {mode} check passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
