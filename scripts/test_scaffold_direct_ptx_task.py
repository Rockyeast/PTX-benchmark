#!/usr/bin/env python3

import json
import hashlib
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "scaffold_direct_ptx_task.py"
CHECKER = REPOSITORY_ROOT / "scripts" / "check_direct_ptx_task.py"


class ScaffoldDirectPtxTaskTest(unittest.TestCase):
    def test_generates_parseable_fail_closed_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "ptx-test-kernel",
                    "--description",
                    "Generated test task.",
                    "--entry",
                    "test_kernel",
                    "--output-root",
                    str(root),
                    "--network-mode",
                    "allowlist",
                    "--allowed-host",
                    "models.internal.example",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            task = root / "ptx-test-kernel"
            self.assertIn("Generated fail-closed task", result.stdout)
            expected = {
                "README.md",
                "task.toml",
                "instruction.md",
                "environment/Dockerfile",
                "environment/policy-check.sh",
                "environment/ptx-docs/README.md",
                "environment/ptx-docs/manifest.toml",
                "solution/solve.sh",
                "tests/test.sh",
                "tests/verify_and_benchmark.py",
            }
            self.assertTrue(expected.issubset({str(p.relative_to(task)) for p in task.rglob("*") if p.is_file()}))
            config = tomllib.loads((task / "task.toml").read_text())
            self.assertEqual(config["task"]["name"], "ptx-bench/ptx-test-kernel")
            self.assertEqual(config["environment"]["network_mode"], "public")
            self.assertEqual(config["agent"]["network_mode"], "allowlist")
            self.assertEqual(config["environment"]["gpu_types"], ["B200"])
            self.assertEqual(
                config["agent"]["allowed_hosts"],
                ["models.internal.example"],
            )
            for path in task.rglob("*"):
                if path.is_file():
                    self.assertNotIn("@@", path.read_text())
            self.assertTrue((task / "solution/solve.sh").stat().st_mode & 0o111)
            self.assertTrue((task / "tests/test.sh").stat().st_mode & 0o111)
            self.assertTrue((task / "environment/policy-check.sh").stat().st_mode & 0o111)

            check = subprocess.run(
                [sys.executable, str(CHECKER), str(task)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(check.returncode, 0, check.stderr)
            release_check = subprocess.run(
                [sys.executable, str(CHECKER), str(task), "--release"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(release_check.returncode, 1)
            self.assertIn("unresolved TODO", release_check.stderr)

            document = task / "environment" / "ptx-docs" / "ptx-isa.txt"
            document.write_text("Pinned PTX ISA test fixture.\n")
            digest = hashlib.sha256(document.read_bytes()).hexdigest()
            manifest = task / "environment" / "ptx-docs" / "manifest.toml"
            manifest.write_text(
                'schema_version = "1.0"\n'
                'document_path = "ptx-isa.txt"\n'
                'document_version = "test-version"\n'
                'source_url = "https://docs.nvidia.com/test"\n'
                f'sha256 = "{digest}"\n'
                'license = "test-only"\n'
            )
            for path in task.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    content = path.read_text()
                except UnicodeDecodeError:
                    continue
                path.write_text(content.replace("TODO", "COMPLETED"))
            task_toml = task / "task.toml"
            task_toml.write_text(
                task_toml.read_text().replace("release_ready = false", "release_ready = true")
            )
            completed_release_check = subprocess.run(
                [sys.executable, str(CHECKER), str(task), "--release"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                completed_release_check.returncode,
                0,
                completed_release_check.stderr,
            )

            split_phase_release_check = subprocess.run(
                [sys.executable, str(CHECKER), str(task), "--release"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                split_phase_release_check.returncode,
                0,
                split_phase_release_check.stderr,
            )

            task_toml.write_text(
                task_toml.read_text().replace(
                    'network_mode = "allowlist"\n'
                    'allowed_hosts = ["models.internal.example"]\n',
                    'network_mode = "public"\n',
                    1,
                )
            )
            public_agent_release_check = subprocess.run(
                [sys.executable, str(CHECKER), str(task), "--release"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(public_agent_release_check.returncode, 1)
            self.assertIn(
                "effective public Agent network policy",
                public_agent_release_check.stderr,
            )

            candidate = task / "candidate.ptx"
            candidate.write_text(
                ".version 8.7\n.target sm_100\n.address_size 64\n"
                ".visible .entry test_kernel() { ret; }\n"
            )
            metrics = task / "metrics.json"
            reward = task / "reward.json"
            verifier = subprocess.run(
                [
                    sys.executable,
                    str(task / "tests/verify_and_benchmark.py"),
                    "--candidate",
                    str(candidate),
                    "--log",
                    str(metrics),
                    "--reward",
                    str(reward),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(verifier.returncode, 1)
            self.assertEqual(json.loads(reward.read_text())["reward"], 0.0)
            self.assertEqual(
                json.loads(metrics.read_text())["error_type"], "NotImplementedError"
            )

    def test_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = [
                sys.executable,
                str(SCRIPT),
                "ptx-test-kernel",
                "--description",
                "Generated test task.",
                "--entry",
                "test_kernel",
                "--network-mode",
                "no-network",
                "--output-root",
                str(root),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 2)
            self.assertIn("refusing to overwrite", second.stderr)

    def test_requires_explicit_network_mode(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "ptx-test-kernel",
                "--description",
                "Generated test task.",
                "--entry",
                "test_kernel",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--network-mode", result.stderr)


if __name__ == "__main__":
    unittest.main()
