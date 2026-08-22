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
                "environment/docker-compose.yaml",
                "environment/files/direct-ptx-policy-check",
                "environment/files/setup.sh",
                "environment/files/ptx-docs/README.md",
                "environment/files/ptx-docs/manifest.toml",
                "solution/solve.sh",
                "tests/direct-ptx-policy-check",
                "tests/ptx-docs-manifest.toml",
                "tests/test.sh",
                "tests/verify_hardware.py",
                "tests/verify_and_benchmark.py",
            }
            self.assertTrue(expected.issubset({str(p.relative_to(task)) for p in task.rglob("*") if p.is_file()}))
            config = tomllib.loads((task / "task.toml").read_text())
            self.assertEqual(config["task"]["name"], "ptx-bench/ptx-test-kernel")
            self.assertEqual(config["environment"]["network_mode"], "public")
            self.assertEqual(config["agent"]["network_mode"], "allowlist")
            self.assertNotIn("gpus", config["environment"])
            self.assertNotIn("gpu_types", config["environment"])
            self.assertEqual(config["metadata"]["required_gpus"], 1)
            self.assertEqual(config["metadata"]["required_gpu_type"], "B200")
            self.assertEqual(config["metadata"]["template_version"], "direct-ptx-v0.1")
            self.assertEqual(config["metadata"]["implementation_status"], "scaffold")
            self.assertEqual(
                config["artifacts"],
                [{"source": "/app/kernel.ptx", "destination": "kernel.ptx"}],
            )
            self.assertEqual(
                config["agent"]["allowed_hosts"],
                ["models.internal.example"],
            )
            for path in task.rglob("*"):
                if path.is_file():
                    self.assertNotIn("@@", path.read_text())
            for relative in (
                "tests/verify_hardware.py",
                "tests/verify_and_benchmark.py",
            ):
                source_path = task / relative
                compile(source_path.read_text(), str(source_path), "exec")
            self.assertTrue((task / "solution/solve.sh").stat().st_mode & 0o111)
            self.assertTrue((task / "tests/test.sh").stat().st_mode & 0o111)
            self.assertTrue(
                (task / "tests/direct-ptx-policy-check").stat().st_mode & 0o111
            )
            self.assertTrue(
                (task / "environment/files/direct-ptx-policy-check").stat().st_mode
                & 0o111
            )
            self.assertTrue(
                (task / "environment/files/setup.sh").stat().st_mode & 0o111
            )
            self.assertTrue((task / "tests/verify_hardware.py").stat().st_mode & 0o111)

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

            candidate = task / "candidate.ptx"
            candidate.write_text(
                ".version 8.7\n.target sm_100\n.address_size 64\n"
                ".visible .entry test_kernel() { ret; }\n"
            )
            metrics = task / "metrics.json"
            reward = task / "reward.json"
            fail_closed_verifier = subprocess.run(
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
            self.assertEqual(fail_closed_verifier.returncode, 2)
            self.assertEqual(json.loads(reward.read_text())["reward"], 0.0)
            self.assertEqual(
                json.loads(metrics.read_text())["status"], "infrastructure_error"
            )
            self.assertEqual(
                json.loads(metrics.read_text())["error_type"], "NotImplementedError"
            )

            document = (
                task / "environment" / "files" / "ptx-docs" / "ptx-isa.txt"
            )
            document.write_text("Pinned PTX ISA test fixture.\n")
            digest = hashlib.sha256(document.read_bytes()).hexdigest()
            runtime_manifest = (
                task / "environment" / "files" / "ptx-docs" / "manifest.toml"
            )
            manifest_source = (
                'schema_version = "1.0"\n'
                'document_path = "ptx-isa.txt"\n'
                'document_version = "test-version"\n'
                'source_url = "https://docs.nvidia.com/test"\n'
                f'sha256 = "{digest}"\n'
                'license = "test-only"\n'
            )
            runtime_manifest.write_text(manifest_source)
            hidden_manifest = task / "tests" / "ptx-docs-manifest.toml"
            hidden_manifest.write_text(manifest_source)
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
                task_toml.read_text()
                .replace('implementation_status = "scaffold"', 'implementation_status = "implemented"')
                .replace("release_ready = false", "release_ready = true")
            )
            incomplete_release_check = subprocess.run(
                [sys.executable, str(CHECKER), str(task), "--release"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(incomplete_release_check.returncode, 1)
            self.assertIn("scaffold implementation marker", incomplete_release_check.stderr)
            self.assertIn("pinned by SHA-256 digest", incomplete_release_check.stderr)

            dockerfile = task / "environment" / "Dockerfile"
            dockerfile.write_text(
                dockerfile.read_text().replace(
                    "FROM nvidia/cuda:13.2.1-devel-ubuntu24.04",
                    "FROM nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:" + "0" * 64,
                )
            )
            oracle = task / "solution" / "solve.sh"
            oracle.write_text(
                "#!/bin/bash\nset -euo pipefail\n"
                "printf '.visible .entry test_kernel() { ret; }\\n' > /app/kernel.ptx\n"
            )
            verifier_path = task / "tests" / "verify_and_benchmark.py"
            verifier_source = verifier_path.read_text()
            verifier_source = verifier_source.replace(
                "# DIRECT_PTX_SCAFFOLD_UNIMPLEMENTED\n", ""
            )
            verifier_source = verifier_source.replace(
                'raise NotImplementedError("task-specific correctness verifier is not implemented")',
                "return None",
            )
            verifier_source = verifier_source.replace(
                'raise NotImplementedError("task-specific performance protocol is not implemented")',
                'return {"reward": 1.0, "performance": 1.0, "speedup_vs_reference": 1.0}',
            )
            verifier_path.write_text(verifier_source)

            # This is deliberately only a structural completion fixture. A
            # static checker cannot prove that a task-specific verifier is
            # semantically correct; Oracle and negative Jobs provide that evidence.
            structurally_completed_release_check = subprocess.run(
                [sys.executable, str(CHECKER), str(task), "--release"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                structurally_completed_release_check.returncode,
                0,
                structurally_completed_release_check.stderr,
            )

            public_policy = (
                task / "environment" / "files" / "direct-ptx-policy-check"
            )
            original_public_policy = public_policy.read_text()
            public_policy.write_text(original_public_policy + "\n# tampered\n")
            mismatched_policy_check = subprocess.run(
                [sys.executable, str(CHECKER), str(task), "--release"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(mismatched_policy_check.returncode, 1)
            self.assertIn(
                "Agent-visible and hidden Direct-PTX policy checkers must match",
                mismatched_policy_check.stderr,
            )
            public_policy.write_text(original_public_policy)

            original_hidden_manifest = hidden_manifest.read_text()
            hidden_manifest.write_text(
                original_hidden_manifest.replace("test-version", "tampered-version")
            )
            mismatched_manifest_check = subprocess.run(
                [sys.executable, str(CHECKER), str(task), "--release"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(mismatched_manifest_check.returncode, 1)
            self.assertIn(
                "runtime and hidden PTX documentation manifests must match",
                mismatched_manifest_check.stderr,
            )
            hidden_manifest.write_text(original_hidden_manifest)

            original_dockerfile = dockerfile.read_text()
            dockerfile.write_text(
                original_dockerfile
                + "\nFROM ubuntu:24.04 AS unpinned-final-stage\n"
            )
            multistage_release_check = subprocess.run(
                [sys.executable, str(CHECKER), str(task), "--release"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(multistage_release_check.returncode, 1)
            self.assertIn(
                "every environment/Dockerfile FROM image must be pinned",
                multistage_release_check.stderr,
            )
            self.assertIn("FROM ubuntu:24.04", multistage_release_check.stderr)
            dockerfile.write_text(
                original_dockerfile
                + "\nFROM ubuntu@sha256:"
                + "1" * 64
                + " AS pinned-final-stage\n"
            )
            pinned_multistage_release_check = subprocess.run(
                [sys.executable, str(CHECKER), str(task), "--release"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                pinned_multistage_release_check.returncode,
                0,
                pinned_multistage_release_check.stderr,
            )
            dockerfile.write_text(original_dockerfile)

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
            self.assertEqual(verifier.returncode, 0)
            self.assertEqual(json.loads(reward.read_text())["reward"], 1.0)

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
