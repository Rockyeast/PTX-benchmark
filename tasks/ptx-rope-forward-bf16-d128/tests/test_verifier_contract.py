#!/usr/bin/env python3

"""Regression tests for Candidate-vs-infrastructure exit status classification."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = TASK_ROOT / "tests" / "verify_and_benchmark.py"
ORACLE = TASK_ROOT / "solution" / "oracle.ptx"
REFERENCE = TASK_ROOT / "tests" / "reference.ptx"


def run_verifier(candidate: Path, reference: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--candidate",
            str(candidate),
            "--reference",
            str(reference),
            "--log",
            str(output / "metrics.json"),
            "--reward",
            str(output / "reward.json"),
        ],
        capture_output=True,
        text=True,
    )


class VerifierContractTest(unittest.TestCase):
    def test_missing_candidate_is_scored_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = run_verifier(output / "missing.ptx", REFERENCE, output)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(
                json.loads((output / "metrics.json").read_text())["status"],
                "candidate_rejected",
            )
            self.assertEqual(
                json.loads((output / "reward.json").read_text())["reward"], 0.0
            )

    def test_missing_reference_is_infrastructure_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = run_verifier(ORACLE, output / "missing.ptx", output)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(
                json.loads((output / "metrics.json").read_text())["status"],
                "infrastructure_error",
            )
            self.assertEqual(
                json.loads((output / "reward.json").read_text())["reward"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
