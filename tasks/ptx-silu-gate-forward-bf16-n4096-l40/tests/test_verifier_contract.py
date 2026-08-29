#!/usr/bin/env python3
"""CPU-only regression tests for the SiLU-gate verifier contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


TASK_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = TASK_ROOT / "tests" / "verify_and_benchmark.py"
ORACLE_PATH = TASK_ROOT / "solution" / "oracle.ptx"
REFERENCE_PATH = TASK_ROOT / "tests" / "reference.ptx"


def load_verifier():
    spec = importlib.util.spec_from_file_location("silu_gate_verifier", VERIFIER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import verifier at {VERIFIER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load_verifier()


class CpuTruthTest(unittest.TestCase):
    def test_case_shape_and_generation_are_deterministic(self) -> None:
        first_input, first_expected = verifier.make_case(2, verifier.RANDOM_SEED)
        second_input, second_expected = verifier.make_case(2, verifier.RANDOM_SEED)

        self.assertEqual(len(first_input), 2 * verifier.INPUT_WIDTH)
        self.assertEqual(len(first_expected), 2 * verifier.OUTPUT_WIDTH)
        self.assertEqual(first_input, second_input)
        self.assertEqual(first_expected, second_expected)

    def test_anchor_values_exercise_silu_and_bf16_output(self) -> None:
        input_bits, expected = verifier.make_case(1, verifier.RANDOM_SEED)
        gates = [verifier.bf16_to_float(bits) for bits in input_bits[:10]]

        self.assertEqual(gates, [-20.0, -8.0, -4.0, -1.0, -0.0, 0.0, 1.0, 4.0, 8.0, 20.0])
        self.assertEqual(expected[0], 0.0)  # up=0 for the first anchor group.
        self.assertTrue(all(math.isfinite(value) for value in expected))

    def test_unwritten_nan_sentinel_is_rejected(self) -> None:
        payload = verifier.array("H", [verifier.OUTPUT_SENTINEL]).tobytes()
        with self.assertRaises(verifier.CandidateRejected):
            verifier.assert_bf16_output(payload, [0.0], "sentinel")

    def test_modified_guard_is_rejected(self) -> None:
        payload = verifier.guarded(b"\x00\x00", 0x5A)
        payload[0] ^= 1
        with self.assertRaises(verifier.CandidateRejected):
            verifier.assert_guards(bytes(payload), 0x5A, "output")

    def test_diagnostic_shape_is_excluded_from_geometric_mean(self) -> None:
        ratios = {1: 100.0, 32: 0.5, 256: 1.0, 4096: 2.0}

        def fake_shape(_cuda, _candidate, _reference, num_tokens, scored):
            return {
                "num_tokens": num_tokens,
                "scored": scored,
                "speedup_vs_reference": ratios[num_tokens],
            }

        with mock.patch.object(verifier, "benchmark_shape", side_effect=fake_shape):
            rows, speedup = verifier.benchmark(None, None, None)

        self.assertEqual([row["num_tokens"] for row in rows], [1, 32, 256, 4096])
        self.assertTrue(math.isclose(speedup, 1.0, rel_tol=1e-12))


class ExitStatusContractTest(unittest.TestCase):
    def test_missing_candidate_is_scored_rejection_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = subprocess.run(
                [
                    sys.executable,
                    str(VERIFIER_PATH),
                    "--candidate",
                    str(output / "missing.ptx"),
                    "--reference",
                    str(REFERENCE_PATH),
                    "--log",
                    str(output / "metrics.json"),
                    "--reward",
                    str(output / "reward.json"),
                    "--correctness-only",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(
                json.loads((output / "metrics.json").read_text())["status"],
                "candidate_rejected",
            )
            self.assertEqual(
                json.loads((output / "reward.json").read_text())["reward"], 0.0
            )

    def test_verifier_failure_is_infrastructure_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = argparse.Namespace(
                candidate=output / "unused.ptx",
                reference=REFERENCE_PATH,
                log=output / "metrics.json",
                reward=output / "reward.json",
                correctness_only=True,
            )
            with (
                mock.patch.object(verifier, "parse_args", return_value=args),
                mock.patch.object(
                    verifier,
                    "run",
                    side_effect=RuntimeError("synthetic verifier failure"),
                ),
            ):
                status = verifier.main()

            self.assertEqual(status, 2)
            self.assertEqual(
                json.loads(args.log.read_text())["status"],
                "infrastructure_error",
            )
            self.assertEqual(json.loads(args.reward.read_text())["reward"], 0.0)

    def test_missing_reference_is_infrastructure_error_without_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = subprocess.run(
                [
                    sys.executable,
                    str(VERIFIER_PATH),
                    "--candidate",
                    str(ORACLE_PATH),
                    "--reference",
                    str(output / "missing-reference.ptx"),
                    "--log",
                    str(output / "metrics.json"),
                    "--reward",
                    str(output / "reward.json"),
                    "--correctness-only",
                ],
                capture_output=True,
                text=True,
                check=False,
            )

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
