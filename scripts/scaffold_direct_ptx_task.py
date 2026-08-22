#!/usr/bin/env python3

"""Generate a fail-closed Harbor direct-PTX task from the v0 template."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPOSITORY_ROOT / "templates" / "direct-ptx-task-v0"
PLACEHOLDER = re.compile(r"@@[A-Z0-9_]+@@")
SLUG = re.compile(r"ptx-[a-z0-9]+(?:-[a-z0-9]+)*$")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def toml_string(value: str) -> str:
    """JSON strings are valid TOML basic strings for the values used here."""
    return json.dumps(value, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a provisional, fail-closed Harbor direct-PTX task."
    )
    parser.add_argument("slug", help="Task slug, e.g. ptx-layernorm-forward-bf16-n4096")
    parser.add_argument("--title", help="Human-readable instruction title")
    parser.add_argument("--description", required=True, help="One-line task description")
    parser.add_argument("--entry", required=True, help="Required PTX entry name")
    parser.add_argument("--hardware-target", default="NVIDIA B200")
    parser.add_argument("--gpu-type", default="B200")
    parser.add_argument("--sm", default="100", help="SM number without the sm_ prefix")
    parser.add_argument("--agent-timeout", type=float, default=600.0)
    parser.add_argument("--verifier-timeout", type=float, default=180.0)
    parser.add_argument(
        "--network-mode",
        choices=("public", "allowlist", "no-network"),
        required=True,
        help=(
            "Explicit Harbor Agent-phase network policy; public is "
            "development-only and cannot pass the release check"
        ),
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help=(
            "Agent-accessible hostname/IP for network-mode=allowlist; "
            "repeat as needed"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "tasks",
        help="Parent directory for the generated task",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if not SLUG.fullmatch(args.slug):
        raise ValueError("slug must start with 'ptx-' and contain lowercase words/digits")
    if not IDENTIFIER.fullmatch(args.entry):
        raise ValueError("entry must be a C/PTX identifier")
    if not args.sm.isdigit() or not (10 <= int(args.sm) <= 999):
        raise ValueError("sm must be a numeric architecture code such as 89, 90, or 100")
    if args.agent_timeout <= 0 or args.verifier_timeout <= 0:
        raise ValueError("timeouts must be positive")
    if args.network_mode == "allowlist" and not args.allowed_host:
        raise ValueError("--network-mode allowlist requires at least one --allowed-host")
    if args.network_mode != "allowlist" and args.allowed_host:
        raise ValueError("--allowed-host is valid only with --network-mode allowlist")


def replacements(args: argparse.Namespace) -> dict[str, str]:
    title = args.title or f"Implement {args.slug} in direct PTX"
    allowed_hosts = ""
    if args.network_mode == "allowlist":
        allowed_hosts = "allowed_hosts = " + json.dumps(args.allowed_host) + "\n"
    return {
        "@@TASK_SLUG@@": args.slug,
        "@@TITLE@@": title,
        "@@DESCRIPTION_TOML@@": toml_string(args.description),
        "@@ENTRY_NAME@@": args.entry,
        "@@HARDWARE_TARGET@@": args.hardware_target,
        "@@HARDWARE_TARGET_PY@@": repr(args.hardware_target),
        "@@HARDWARE_TARGET_TOML@@": toml_string(args.hardware_target),
        "@@GPU_TYPE@@": args.gpu_type,
        "@@GPU_TYPE_TOML@@": toml_string(args.gpu_type),
        "@@GPU_KEYWORD_TOML@@": toml_string(args.gpu_type.lower()),
        "@@SM@@": args.sm,
        "@@AGENT_TIMEOUT@@": str(float(args.agent_timeout)),
        "@@VERIFIER_TIMEOUT@@": str(float(args.verifier_timeout)),
        "@@NETWORK_MODE@@": args.network_mode,
        "@@ALLOWED_HOSTS_TOML@@": allowed_hosts.rstrip(),
    }


def render_tree(destination: Path, values: dict[str, str]) -> None:
    shutil.copytree(
        TEMPLATE_ROOT,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    for path in sorted(destination.rglob("*")):
        if not path.is_file():
            continue
        content = path.read_text()
        for marker, value in values.items():
            content = content.replace(marker, value)
        leftovers = sorted(set(PLACEHOLDER.findall(content)))
        if leftovers:
            raise RuntimeError(f"unresolved placeholders in {path}: {leftovers}")
        path.write_text(content)
    for relative in (
        "environment/files/direct-ptx-policy-check",
        "environment/files/setup.sh",
        "solution/solve.sh",
        "tests/direct-ptx-policy-check",
        "tests/test.sh",
        "tests/verify_hardware.py",
        "tests/verify_and_benchmark.py",
    ):
        path = destination / relative
        path.chmod(path.stat().st_mode | 0o111)


def main() -> int:
    args = parse_args()
    try:
        validate(args)
        destination = args.output_root.resolve() / args.slug
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite existing path: {destination}")
        if args.dry_run:
            print(f"Would generate {destination} from {TEMPLATE_ROOT}")
            return 0
        render_tree(destination, replacements(args))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(f"Generated fail-closed task: {destination}")
    print("Next: replace every TODO, align ABI across instruction/Oracle/verifier,")
    print("pin the local PTX documentation snapshot, then run the task checker,")
    print("Oracle, negative fixtures, and a real Agent in that order.")
    print("Docker: export PTX_BENCH_GPU_DEVICE=<idle GPU index>")
    print("Singularity: pass --override-gpus 1 and the validated SIF kwargs")
    print(f"Check: python3 scripts/check_direct_ptx_task.py {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
