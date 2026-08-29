#!/usr/bin/env python3

import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD = REPOSITORY_ROOT / "scripts" / "scaffold_direct_ptx_task.py"
CHECKER = REPOSITORY_ROOT / "scripts" / "check_direct_ptx_task.py"


EXPECTED_FILES = {
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
}


def generate(root: Path, *, network_mode: str = "allowlist") -> Path:
    command = [
        sys.executable,
        str(SCAFFOLD),
        "ptx-test-kernel",
        "--description",
        "Generated test task.",
        "--entry",
        "test_kernel",
        "--agent-timeout",
        "900",
        "--output-root",
        str(root),
        "--network-mode",
        network_mode,
    ]
    if network_mode == "allowlist":
        command.extend(["--allowed-host", "models.internal.example"])
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    if "Generated fail-closed task" not in result.stdout:
        raise AssertionError(result.stdout)
    return root / "ptx-test-kernel"


def run_checker(task: Path, *, release: bool = False) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(CHECKER), str(task)]
    if release:
        command.append("--release")
    return subprocess.run(command, capture_output=True, text=True)


def complete_release_fixture(task: Path) -> None:
    for path in task.rglob("*"):
        if not path.is_file():
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            continue
        path.write_text(content.replace("TODO", "COMPLETED"))

    config_path = task / "task.toml"
    config_path.write_text(
        config_path.read_text()
        .replace('implementation_status = "scaffold"', 'implementation_status = "implemented"')
        .replace("release_ready = false", "release_ready = true")
    )
    (task / "environment" / "reference.py").write_text(
        "#!/usr/bin/env python3\ndef reference(x):\n    return x\n"
    )
    (task / "environment" / "measure.py").write_text(
        "#!/usr/bin/env python3\nprint('correctness=1 performance_ms=1.0')\n"
    )
    verifier = task / "tests" / "verifier.py"
    verifier.write_text(
        verifier.read_text()
        .replace("# DIRECT_PTX_VERIFIER_UNIMPLEMENTED\n", "")
        .replace(
            'raise NotImplementedError("task-specific hidden verifier is not implemented")',
            'return 1.0, {"correctness": 1.0, "performance": 1.0}',
        )
    )
    baseline = (
        ".version 9.2\n.target sm_100\n.address_size 64\n"
        "// completed baseline\n.visible .entry test_kernel() { ret; }\n"
    )
    reference = (
        ".version 9.2\n.target sm_100\n.address_size 64\n"
        "// completed reference\n.visible .entry test_kernel() { ret; }\n"
    )
    (task / "tests" / "performance_baseline.ptx").write_text(baseline)
    (task / "tests" / "performance_reference.ptx").write_text(reference)
    (task / "solution" / "performance_reference.ptx").write_text(reference)


class ScaffoldDirectPtxTaskTest(unittest.TestCase):
    def test_generates_minimal_v1_scaffold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = generate(Path(temporary))
            actual = {
                str(path.relative_to(task))
                for path in task.rglob("*")
                if path.is_file()
            }
            self.assertEqual(actual, EXPECTED_FILES)

            config = tomllib.loads((task / "task.toml").read_text())
            self.assertEqual(config["metadata"]["template_version"], "direct-ptx-v1")
            self.assertEqual(config["agent"]["network_mode"], "allowlist")
            self.assertEqual(
                config["agent"]["allowed_hosts"], ["models.internal.example"]
            )
            self.assertEqual(
                config["artifacts"],
                [{"source": "/app/kernel.ptx", "destination": "kernel.ptx"}],
            )
            self.assertNotIn("gpus", config["environment"])
            self.assertNotIn("gpu_types", config["environment"])

            for path in task.rglob("*"):
                if path.is_file():
                    self.assertNotIn("@@", path.read_text())
            for relative in (
                "environment/reference.py",
                "environment/measure.py",
                "tests/verifier.py",
            ):
                source = task / relative
                compile(source.read_text(), str(source), "exec")
            for relative in (
                "environment/measure.py",
                "solution/solve.sh",
                "tests/test.sh",
                "tests/verifier.py",
            ):
                self.assertTrue((task / relative).stat().st_mode & 0o111)

            scaffold_check = run_checker(task)
            self.assertEqual(scaffold_check.returncode, 0, scaffold_check.stderr)
            release_check = run_checker(task, release=True)
            self.assertEqual(release_check.returncode, 1)
            self.assertIn("unresolved TODO", release_check.stderr)
            self.assertIn("unimplemented v1 marker", release_check.stderr)

    def test_completed_v1_release_and_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = generate(Path(temporary))
            complete_release_fixture(task)

            release_check = run_checker(task, release=True)
            self.assertEqual(release_check.returncode, 0, release_check.stderr)

            hidden_reference = task / "tests" / "performance_reference.ptx"
            original_reference = hidden_reference.read_text()
            hidden_reference.write_text(original_reference + "// mismatch\n")
            mismatch = run_checker(task, release=True)
            self.assertEqual(mismatch.returncode, 1)
            self.assertIn("files must match", mismatch.stderr)
            hidden_reference.write_text(original_reference)

            baseline = task / "tests" / "performance_baseline.ptx"
            original_baseline = baseline.read_text()
            baseline.write_text(original_reference)
            identical = run_checker(task, release=True)
            self.assertEqual(identical.returncode, 1)
            self.assertIn("Baseline and Reference must not be identical", identical.stderr)
            baseline.write_text(original_baseline)

            task_toml = task / "task.toml"
            original_task_toml = task_toml.read_text()
            task_toml.write_text(
                original_task_toml.replace(
                    'network_mode = "allowlist"\nallowed_hosts = ["models.internal.example"]',
                    'network_mode = "public"',
                    1,
                )
            )
            public_network = run_checker(task, release=True)
            self.assertEqual(public_network.returncode, 1)
            self.assertIn("effective public Agent network policy", public_network.stderr)
            task_toml.write_text(original_task_toml)

    def test_scaffold_argument_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_host = subprocess.run(
                [
                    sys.executable,
                    str(SCAFFOLD),
                    "ptx-test-kernel",
                    "--description",
                    "Generated test task.",
                    "--entry",
                    "test_kernel",
                    "--agent-timeout",
                    "900",
                    "--output-root",
                    str(root),
                    "--network-mode",
                    "allowlist",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(missing_host.returncode, 2)
            self.assertIn("requires at least one --allowed-host", missing_host.stderr)

    def test_checker_rejects_legacy_template(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = generate(Path(temporary))
            task_toml = task / "task.toml"
            task_toml.write_text(
                task_toml.read_text().replace(
                    'template_version = "direct-ptx-v1"',
                    'template_version = "direct-ptx-v0.3"',
                )
            )
            result = run_checker(task)
            self.assertEqual(result.returncode, 1)
            self.assertIn("migrate the task to direct-ptx-v1", result.stderr)


if __name__ == "__main__":
    unittest.main()
