#!/usr/bin/env python3
"""Feasibility verifier for BF16 RMSNorm forward specialized to N=4096."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import random
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path


N = 4096
BLOCK_SIZE = 128
DYNAMIC_SHARED_BYTES = 1024
MAX_PTX_BYTES = 4 * 1024 * 1024
GUARD_BYTES = 256
ABS_TOL = 1e-1
REL_TOL = 1e-3
RANDOM_SEED = 0x524D534E
CORRECTNESS_CASES = [
    (0, 1e-6),
    (1, 1e-6),
    (2, 1e-5),
    (3, 1e-6),
    (17, 1e-6),
    (199, 1e-6),
]
PERFORMANCE_ROWS = [1024, 8192, 32768]


class CudaError(RuntimeError):
    pass


class CudaDriver:
    """Small CUDA Driver API wrapper with no third-party Python dependency."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._configure_signatures()
        self.device = ctypes.c_int()
        self.context = ctypes.c_void_p()
        self.allocations: set[int] = set()
        self.check(self.lib.cuInit(0), "cuInit")
        self.check(self.lib.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")
        self.check(
            self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device),
            "cuCtxCreate",
        )

    def _configure_signatures(self) -> None:
        lib = self.lib
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        lib.cuCtxSynchronize.argtypes = []
        lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemcpyHtoD_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_size_t,
        ]
        lib.cuMemsetD16_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_ushort,
            ctypes.c_size_t,
        ]
        lib.cuMemsetD32_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint,
            ctypes.c_size_t,
        ]
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        lib.cuEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def error_name(self, result: int) -> str:
        name = ctypes.c_char_p()
        if self.lib.cuGetErrorName(result, ctypes.byref(name)) == 0 and name.value:
            return name.value.decode(errors="replace")
        return "UNKNOWN"

    def check(self, result: int, operation: str) -> None:
        if result != 0:
            raise CudaError(f"{operation} failed: {self.error_name(result)} ({result})")

    def device_name(self) -> str:
        buffer = ctypes.create_string_buffer(256)
        self.check(
            self.lib.cuDeviceGetName(buffer, len(buffer), self.device),
            "cuDeviceGetName",
        )
        return buffer.value.decode(errors="replace")

    def load_function(
        self, ptx_path: Path, entry: bytes
    ) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
        module = ctypes.c_void_p()
        function = ctypes.c_void_p()
        self.check(
            self.lib.cuModuleLoad(ctypes.byref(module), str(ptx_path).encode()),
            f"cuModuleLoad({ptx_path})",
        )
        try:
            self.check(
                self.lib.cuModuleGetFunction(ctypes.byref(function), module, entry),
                f"cuModuleGetFunction({entry.decode()})",
            )
        except Exception:
            self.lib.cuModuleUnload(module)
            raise
        return module, function

    def alloc(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        self.check(self.lib.cuMemAlloc_v2(ctypes.byref(pointer), size), "cuMemAlloc")
        self.allocations.add(pointer.value)
        return pointer.value

    def free(self, pointer: int) -> None:
        if pointer in self.allocations:
            self.check(self.lib.cuMemFree_v2(pointer), "cuMemFree")
            self.allocations.remove(pointer)

    def copy_to_device(self, destination: int, data: bytes | bytearray) -> None:
        host = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        self.check(
            self.lib.cuMemcpyHtoD_v2(destination, host, len(data)),
            "cuMemcpyHtoD",
        )

    def copy_from_device(self, source: int, size: int) -> bytes:
        host = (ctypes.c_ubyte * size)()
        self.check(self.lib.cuMemcpyDtoH_v2(host, source, size), "cuMemcpyDtoH")
        return bytes(host)

    def memset_d16(self, destination: int, value: int, count: int) -> None:
        self.check(
            self.lib.cuMemsetD16_v2(destination, value, count),
            "cuMemsetD16",
        )

    def memset_d32(self, destination: int, value: int, count: int) -> None:
        self.check(
            self.lib.cuMemsetD32_v2(destination, value, count),
            "cuMemsetD32",
        )

    def launch(
        self,
        function: ctypes.c_void_p,
        x: int,
        weight: int,
        output: int,
        rows: int,
        eps: float,
    ) -> None:
        x_arg = ctypes.c_uint64(x)
        weight_arg = ctypes.c_uint64(weight)
        output_arg = ctypes.c_uint64(output)
        rows_arg = ctypes.c_uint32(rows)
        eps_arg = ctypes.c_float(eps)
        arguments = (ctypes.c_void_p * 5)(
            ctypes.cast(ctypes.byref(x_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(weight_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(output_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(rows_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(eps_arg), ctypes.c_void_p),
        )
        self.check(
            self.lib.cuLaunchKernel(
                function,
                max(1, rows),
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                DYNAMIC_SHARED_BYTES,
                None,
                arguments,
                None,
            ),
            "cuLaunchKernel",
        )

    def synchronize(self) -> None:
        self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def measure_ms(
        self,
        function: ctypes.c_void_p,
        x: int,
        weight: int,
        output: int,
        rows: int,
        eps: float,
        iterations: int,
    ) -> float:
        start = ctypes.c_void_p()
        end = ctypes.c_void_p()
        self.check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
        self.check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate")
        try:
            self.check(self.lib.cuEventRecord(start, None), "cuEventRecord(start)")
            for _ in range(iterations):
                self.launch(function, x, weight, output, rows, eps)
            self.check(self.lib.cuEventRecord(end, None), "cuEventRecord(end)")
            self.check(self.lib.cuEventSynchronize(end), "cuEventSynchronize")
            elapsed = ctypes.c_float()
            self.check(
                self.lib.cuEventElapsedTime(ctypes.byref(elapsed), start, end),
                "cuEventElapsedTime",
            )
            return float(elapsed.value) / iterations
        finally:
            if end.value:
                self.lib.cuEventDestroy_v2(end)
            if start.value:
                self.lib.cuEventDestroy_v2(start)

    def close(self) -> None:
        for pointer in tuple(self.allocations):
            self.lib.cuMemFree_v2(pointer)
        self.allocations.clear()
        if self.context.value:
            self.lib.cuCtxDestroy_v2(self.context)
            self.context = ctypes.c_void_p()


def f32(value: float) -> float:
    return ctypes.c_float(value).value


def float_to_bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def bf16_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def guarded(payload: bytes, guard_byte: int) -> bytearray:
    return (
        bytearray([guard_byte]) * GUARD_BYTES
        + payload
        + bytearray([guard_byte]) * GUARD_BYTES
    )


def make_case(rows: int, seed: int) -> tuple[array, list[float], array]:
    rng = random.Random(seed)
    x_bits = array("H")
    x_values: list[float] = []
    for row in range(rows):
        for _ in range(N):
            if rows >= 2 and row == 0:
                source = 0.0
            elif rows >= 2 and row == 1:
                source = 1.25
            else:
                source = rng.uniform(-2.0, 2.0)
            bits = float_to_bf16(source)
            x_bits.append(bits)
            x_values.append(bf16_to_float(bits))
    weights = array("f", (f32(rng.uniform(0.5, 1.5)) for _ in range(N)))
    return x_bits, x_values, weights


def expected_output(
    x_values: list[float], weights: array, rows: int, eps: float
) -> list[float]:
    expected: list[float] = []
    for row in range(rows):
        start = row * N
        values = x_values[start : start + N]
        mean_sq = sum(value * value for value in values) / N
        rstd = 1.0 / math.sqrt(mean_sq + eps)
        for value, weight in zip(values, weights):
            expected.append(bf16_to_float(float_to_bf16(value * rstd * weight)))
    return expected


def validate_function(
    cuda: CudaDriver,
    function: ctypes.c_void_p,
    label: str,
) -> None:
    for case_index, (rows, eps) in enumerate(CORRECTNESS_CASES):
        x_bits, x_values, weights = make_case(rows, RANDOM_SEED + case_index)
        expected = expected_output(x_values, weights, rows, eps)
        x_host = guarded(x_bits.tobytes(), 0xA1)
        weight_host = guarded(weights.tobytes(), 0xB2)
        output_host = guarded(bytes([0xCD]) * (rows * N * 2), 0xC3)
        d_x = cuda.alloc(len(x_host))
        d_weight = cuda.alloc(len(weight_host))
        d_output = cuda.alloc(len(output_host))
        try:
            cuda.copy_to_device(d_x, x_host)
            cuda.copy_to_device(d_weight, weight_host)
            cuda.copy_to_device(d_output, output_host)
            cuda.launch(
                function,
                d_x + GUARD_BYTES,
                d_weight + GUARD_BYTES,
                d_output + GUARD_BYTES,
                rows,
                eps,
            )
            cuda.synchronize()
            x_after = cuda.copy_from_device(d_x, len(x_host))
            weight_after = cuda.copy_from_device(d_weight, len(weight_host))
            output_after = cuda.copy_from_device(d_output, len(output_host))
        finally:
            cuda.free(d_output)
            cuda.free(d_weight)
            cuda.free(d_x)

        if x_after != bytes(x_host):
            raise AssertionError(f"{label}: x was modified for M={rows}")
        if weight_after != bytes(weight_host):
            raise AssertionError(f"{label}: weight was modified for M={rows}")
        if output_after[:GUARD_BYTES] != bytes([0xC3]) * GUARD_BYTES:
            raise AssertionError(f"{label}: output prefix guard changed for M={rows}")
        if output_after[-GUARD_BYTES:] != bytes([0xC3]) * GUARD_BYTES:
            raise AssertionError(f"{label}: output suffix guard changed for M={rows}")

        actual_bits = array("H")
        actual_bits.frombytes(output_after[GUARD_BYTES : GUARD_BYTES + rows * N * 2])
        for index, (got_bits, want) in enumerate(zip(actual_bits, expected)):
            got = bf16_to_float(got_bits)
            if not math.isfinite(got) or not math.isclose(
                got, want, rel_tol=REL_TOL, abs_tol=ABS_TOL
            ):
                raise AssertionError(
                    f"{label}: wrong output at M={rows}, index={index}: "
                    f"expected {want}, got {got}"
                )


def compile_reference(source: Path, destination: Path) -> dict[str, object]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise FileNotFoundError("nvcc is required to compile the hidden reference")
    command = [
        nvcc,
        "--ptx",
        "--optimize=3",
        "--gpu-architecture=compute_100",
        "--output-file",
        str(destination),
        str(source),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "reference compilation failed:\n" + completed.stdout + completed.stderr
        )
    version = subprocess.run(
        [nvcc, "--version"], text=True, capture_output=True, check=False
    )
    return {
        "command": command,
        "nvcc_version": (version.stdout + version.stderr).strip(),
    }


def validate_constant_rows(
    cuda: CudaDriver,
    function: ctypes.c_void_p,
    x: int,
    weight: int,
    output: int,
    rows: int,
    eps: float,
    label: str,
) -> None:
    cuda.launch(function, x, weight, output, rows, eps)
    cuda.synchronize()
    expected = bf16_to_float(float_to_bf16(1.0 / math.sqrt(1.0 + eps)))
    for row in sorted({0, rows // 2, rows - 1}):
        payload = cuda.copy_from_device(output + row * N * 2, N * 2)
        values = array("H")
        values.frombytes(payload)
        for col, bits in enumerate(values):
            got = bf16_to_float(bits)
            if not math.isclose(got, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                raise AssertionError(
                    f"{label}: performance-shape check failed at "
                    f"M={rows}, row={row}, col={col}: expected {expected}, got {got}"
                )


def dispersion(values: list[float]) -> float:
    median = statistics.median(values)
    return math.inf if median == 0 else (max(values) - min(values)) / median


def benchmark_batches(
    cuda: CudaDriver,
    candidate: ctypes.c_void_p,
    reference: ctypes.c_void_p,
    x: int,
    weight: int,
    output: int,
    rows: int,
    eps: float,
    iterations: int,
) -> tuple[list[float], list[float]]:
    timings = {"candidate": [], "reference": []}
    for batch_index in range(3):
        order = (
            (("candidate", candidate), ("reference", reference))
            if batch_index % 2 == 0
            else (("reference", reference), ("candidate", candidate))
        )
        for label, function in order:
            timings[label].append(
                cuda.measure_ms(function, x, weight, output, rows, eps, iterations)
            )
    return timings["candidate"], timings["reference"]


def benchmark(
    cuda: CudaDriver,
    candidate: ctypes.c_void_p,
    reference: ctypes.c_void_p,
) -> tuple[list[dict[str, object]], float]:
    results: list[dict[str, object]] = []
    ratios: list[float] = []
    eps = 1e-6
    for rows in PERFORMANCE_ROWS:
        elements = rows * N
        d_x = cuda.alloc(elements * 2)
        d_weight = cuda.alloc(N * 4)
        d_output = cuda.alloc(elements * 2)
        try:
            cuda.memset_d16(d_x, 0x3F80, elements)
            cuda.memset_d32(d_weight, 0x3F800000, N)
            cuda.memset_d16(d_output, 0, elements)
            validate_constant_rows(
                cuda, candidate, d_x, d_weight, d_output, rows, eps, "candidate"
            )
            validate_constant_rows(
                cuda, reference, d_x, d_weight, d_output, rows, eps, "reference"
            )
            for _ in range(5):
                cuda.launch(candidate, d_x, d_weight, d_output, rows, eps)
                cuda.launch(reference, d_x, d_weight, d_output, rows, eps)
            cuda.synchronize()

            candidate_probe = cuda.measure_ms(
                candidate, d_x, d_weight, d_output, rows, eps, 5
            )
            reference_probe = cuda.measure_ms(
                reference, d_x, d_weight, d_output, rows, eps, 5
            )
            slower_probe = max(candidate_probe, reference_probe, 1e-6)
            iterations = max(5, min(2000, math.ceil(20.0 / slower_probe)))
            candidate_times, reference_times = benchmark_batches(
                cuda,
                candidate,
                reference,
                d_x,
                d_weight,
                d_output,
                rows,
                eps,
                iterations,
            )
            candidate_median = statistics.median(candidate_times)
            reference_median = statistics.median(reference_times)
            ratio = reference_median / candidate_median
            ratios.append(ratio)
            results.append(
                {
                    "M": rows,
                    "N": N,
                    "iterations_per_batch": iterations,
                    "candidate_ms_per_launch": candidate_times,
                    "reference_ms_per_launch": reference_times,
                    "candidate_median_ms": candidate_median,
                    "reference_median_ms": reference_median,
                    "speedup_vs_reference": ratio,
                    "candidate_dispersion": dispersion(candidate_times),
                    "reference_dispersion": dispersion(reference_times),
                }
            )
        finally:
            cuda.free(d_output)
            cuda.free(d_weight)
            cuda.free(d_x)
    geometric_mean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    return results, geometric_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--correctness-only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    if not args.candidate.is_file():
        raise FileNotFoundError(f"missing PTX output: {args.candidate}")
    if args.candidate.stat().st_size > MAX_PTX_BYTES:
        raise ValueError(f"candidate PTX exceeds {MAX_PTX_BYTES} bytes")
    if not args.reference_source.is_file():
        raise FileNotFoundError(f"missing reference source: {args.reference_source}")

    with tempfile.TemporaryDirectory(prefix="ptx-rmsnorm-") as temp_dir:
        reference_ptx = Path(temp_dir) / "reference.ptx"
        compiler = compile_reference(args.reference_source, reference_ptx)
        cuda = CudaDriver()
        candidate_module = ctypes.c_void_p()
        reference_module = ctypes.c_void_p()
        try:
            candidate_module, candidate = cuda.load_function(
                args.candidate, b"rmsnorm_fwd_bf16_n4096"
            )
            reference_module, reference = cuda.load_function(
                reference_ptx, b"rmsnorm_fwd_bf16_n4096"
            )
            validate_function(cuda, candidate, "candidate")
            validate_function(cuda, reference, "reference")
            if args.correctness_only:
                performance_rows: list[dict[str, object]] = []
                speedup = 1.0
            else:
                performance_rows, speedup = benchmark(cuda, candidate, reference)
            performance = max(0.0, min(1.0, speedup))
            metrics = {
                "status": "passed",
                "validation_level": "feasibility",
                "gpu": cuda.device_name(),
                "candidate_path": str(args.candidate),
                "candidate_bytes": args.candidate.stat().st_size,
                "correctness_cases": [
                    {"M": rows, "N": N, "eps": eps} for rows, eps in CORRECTNESS_CASES
                ],
                "compiler": compiler,
                "performance_rows": performance_rows,
                "speedup_vs_reference": speedup,
            }
            rewards = {
                "reward": performance,
                "correctness": 1.0,
                "performance": performance,
                "speedup_vs_reference": speedup,
            }
            write_json(args.log, metrics)
            write_json(args.reward, rewards)
            print(
                "PTX_RMSNORM_FEASIBILITY_OK: "
                f"{len(CORRECTNESS_CASES)} correctness cases; "
                f"speedup_vs_reference={speedup:.6f}; reward={performance:.6f}"
            )
            return 0
        finally:
            if reference_module.value:
                cuda.lib.cuModuleUnload(reference_module)
            if candidate_module.value:
                cuda.lib.cuModuleUnload(candidate_module)
            cuda.close()


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        try:
            write_json(args.log, failure)
        except Exception:
            pass
        print(
            f"PTX_RMSNORM_FEASIBILITY_FAIL: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
