#!/usr/bin/env python3
"""Correctness-gate and benchmark a submitted PTX vector-add kernel."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path


BLOCK_SIZE = 256
MAX_PTX_BYTES = 1024 * 1024
GUARD_BYTES = 256
ABS_TOL = 1e-6
REL_TOL = 1e-6
RANDOM_SEED = 0x505458
CORRECTNESS_SIZES = [
    0,
    1,
    2,
    31,
    32,
    33,
    127,
    255,
    256,
    257,
    1023,
    1024,
    1025,
    65537,
]
PERFORMANCE_SIZES = [1 << 20, 1 << 24, 1 << 26]


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

    def memset_d32(self, destination: int, value: int, count: int) -> None:
        self.check(
            self.lib.cuMemsetD32_v2(destination, value, count),
            "cuMemsetD32",
        )

    def launch(
        self,
        function: ctypes.c_void_p,
        a: int,
        b: int,
        output: int,
        n_value: int,
    ) -> None:
        a_arg = ctypes.c_uint64(a)
        b_arg = ctypes.c_uint64(b)
        output_arg = ctypes.c_uint64(output)
        n_arg = ctypes.c_uint32(n_value)
        arguments = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.byref(a_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(b_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(output_arg), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n_arg), ctypes.c_void_p),
        )
        grid_x = max(1, (n_value + BLOCK_SIZE - 1) // BLOCK_SIZE)
        self.check(
            self.lib.cuLaunchKernel(
                function,
                grid_x,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
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
        a: int,
        b: int,
        output: int,
        n_value: int,
        iterations: int,
    ) -> float:
        start = ctypes.c_void_p()
        end = ctypes.c_void_p()
        self.check(
            self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate(start)"
        )
        self.check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate(end)")
        try:
            self.check(self.lib.cuEventRecord(start, None), "cuEventRecord(start)")
            for _ in range(iterations):
                self.launch(function, a, b, output, n_value)
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


def make_inputs(n_value: int, seed: int) -> tuple[array, array]:
    rng = random.Random(seed)
    a = array("f", (f32(rng.uniform(-1000.0, 1000.0)) for _ in range(n_value)))
    b = array("f", (f32(rng.uniform(-1000.0, 1000.0)) for _ in range(n_value)))
    return a, b


def guarded(payload: bytes, guard_byte: int) -> bytearray:
    return (
        bytearray([guard_byte]) * GUARD_BYTES
        + payload
        + bytearray([guard_byte]) * GUARD_BYTES
    )


def correctness_sizes() -> list[int]:
    rng = random.Random(RANDOM_SEED)
    sizes = list(CORRECTNESS_SIZES)
    sizes.extend(rng.randrange(3, 1 << 18) for _ in range(5))
    sizes.append((1 << 20) + rng.randrange(1, 4096))
    return sizes


def validate_function(
    cuda: CudaDriver,
    function: ctypes.c_void_p,
    label: str,
    sizes: list[int],
) -> None:
    for case_index, n_value in enumerate(sizes):
        a_values, b_values = make_inputs(n_value, RANDOM_SEED + case_index)
        a_host = guarded(a_values.tobytes(), 0xA1)
        b_host = guarded(b_values.tobytes(), 0xB2)
        output_host = guarded(bytes([0xCD]) * (n_value * 4), 0xC3)
        d_a = cuda.alloc(len(a_host))
        d_b = cuda.alloc(len(b_host))
        d_output = cuda.alloc(len(output_host))
        try:
            cuda.copy_to_device(d_a, a_host)
            cuda.copy_to_device(d_b, b_host)
            cuda.copy_to_device(d_output, output_host)
            cuda.launch(
                function,
                d_a + GUARD_BYTES,
                d_b + GUARD_BYTES,
                d_output + GUARD_BYTES,
                n_value,
            )
            cuda.synchronize()

            a_after = cuda.copy_from_device(d_a, len(a_host))
            b_after = cuda.copy_from_device(d_b, len(b_host))
            output_after = cuda.copy_from_device(d_output, len(output_host))
        finally:
            cuda.free(d_output)
            cuda.free(d_b)
            cuda.free(d_a)

        if a_after != bytes(a_host):
            raise AssertionError(f"{label}: input a was modified for n={n_value}")
        if b_after != bytes(b_host):
            raise AssertionError(f"{label}: input b was modified for n={n_value}")
        if output_after[:GUARD_BYTES] != bytes([0xC3]) * GUARD_BYTES:
            raise AssertionError(
                f"{label}: output prefix guard changed for n={n_value}"
            )
        if output_after[-GUARD_BYTES:] != bytes([0xC3]) * GUARD_BYTES:
            raise AssertionError(
                f"{label}: output suffix guard changed for n={n_value}"
            )

        actual = array("f")
        actual.frombytes(output_after[GUARD_BYTES : GUARD_BYTES + n_value * 4])
        for index, (got, left, right) in enumerate(zip(actual, a_values, b_values)):
            expected = f32(left + right)
            if not math.isclose(got, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                raise AssertionError(
                    f"{label}: wrong output at n={n_value}, i={index}: "
                    f"expected {expected}, got {got}"
                )


def compile_reference(source: Path, destination: Path) -> dict[str, object]:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise FileNotFoundError("nvcc is required to build the hidden CUDA reference")
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


def dispersion(values: list[float]) -> float:
    median = statistics.median(values)
    if median == 0:
        return math.inf
    return (max(values) - min(values)) / median


def benchmark_batches(
    cuda: CudaDriver,
    candidate: ctypes.c_void_p,
    reference: ctypes.c_void_p,
    d_a: int,
    d_b: int,
    d_output: int,
    n_value: int,
    iterations: int,
) -> tuple[list[float], list[float]]:
    candidate_batches: list[float] = []
    reference_batches: list[float] = []
    for batch_index in range(5):
        ordered = (
            (("candidate", candidate), ("reference", reference))
            if batch_index % 2 == 0
            else (("reference", reference), ("candidate", candidate))
        )
        for label, function in ordered:
            elapsed = cuda.measure_ms(function, d_a, d_b, d_output, n_value, iterations)
            if label == "candidate":
                candidate_batches.append(elapsed)
            else:
                reference_batches.append(elapsed)
    return candidate_batches, reference_batches


def benchmark(
    cuda: CudaDriver,
    candidate: ctypes.c_void_p,
    reference: ctypes.c_void_p,
) -> tuple[list[dict[str, object]], float]:
    rows: list[dict[str, object]] = []
    ratios: list[float] = []
    for n_value in PERFORMANCE_SIZES:
        byte_count = n_value * 4
        d_a = cuda.alloc(byte_count)
        d_b = cuda.alloc(byte_count)
        d_output = cuda.alloc(byte_count)
        try:
            cuda.memset_d32(d_a, 0x3F800000, n_value)
            cuda.memset_d32(d_b, 0x40000000, n_value)
            cuda.memset_d32(d_output, 0, n_value)
            for _ in range(10):
                cuda.launch(candidate, d_a, d_b, d_output, n_value)
                cuda.launch(reference, d_a, d_b, d_output, n_value)
            cuda.synchronize()

            candidate_probe = cuda.measure_ms(
                candidate, d_a, d_b, d_output, n_value, 10
            )
            reference_probe = cuda.measure_ms(
                reference, d_a, d_b, d_output, n_value, 10
            )
            slower_probe = max(candidate_probe, reference_probe, 1e-6)
            iterations = max(10, min(10000, math.ceil(50.0 / slower_probe)))
            candidate_batches, reference_batches = benchmark_batches(
                cuda,
                candidate,
                reference,
                d_a,
                d_b,
                d_output,
                n_value,
                iterations,
            )
            repeated = False
            if max(dispersion(candidate_batches), dispersion(reference_batches)) > 0.05:
                candidate_batches, reference_batches = benchmark_batches(
                    cuda,
                    candidate,
                    reference,
                    d_a,
                    d_b,
                    d_output,
                    n_value,
                    iterations,
                )
                repeated = True

            candidate_median = statistics.median(candidate_batches)
            reference_median = statistics.median(reference_batches)
            ratio = reference_median / candidate_median
            ratios.append(ratio)
            rows.append(
                {
                    "n": n_value,
                    "iterations_per_batch": iterations,
                    "candidate_ms_per_launch": candidate_batches,
                    "reference_ms_per_launch": reference_batches,
                    "candidate_median_ms": candidate_median,
                    "reference_median_ms": reference_median,
                    "speedup_vs_reference": ratio,
                    "candidate_dispersion": dispersion(candidate_batches),
                    "reference_dispersion": dispersion(reference_batches),
                    "repeated_for_instability": repeated,
                }
            )
        finally:
            cuda.free(d_output)
            cuda.free(d_b)
            cuda.free(d_a)

    geometric_mean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    return rows, geometric_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Run both implementations through correctness but skip performance timing.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    if not args.candidate.is_file():
        raise FileNotFoundError(f"missing PTX output: {args.candidate}")
    if args.candidate.stat().st_size > MAX_PTX_BYTES:
        raise ValueError(
            f"candidate PTX exceeds {MAX_PTX_BYTES} bytes: "
            f"{args.candidate.stat().st_size}"
        )
    if not args.reference_source.is_file():
        raise FileNotFoundError(f"missing reference source: {args.reference_source}")

    with tempfile.TemporaryDirectory(prefix="ptx-vector-add-") as temp_dir:
        reference_ptx = Path(temp_dir) / "reference.ptx"
        compiler = compile_reference(args.reference_source, reference_ptx)
        cuda = CudaDriver()
        candidate_module = ctypes.c_void_p()
        reference_module = ctypes.c_void_p()
        try:
            candidate_module, candidate = cuda.load_function(
                args.candidate, b"vector_add"
            )
            reference_module, reference = cuda.load_function(
                reference_ptx, b"vector_add"
            )
            sizes = correctness_sizes()
            validate_function(cuda, candidate, "candidate", sizes)
            validate_function(cuda, reference, "reference", sizes)

            if (
                args.correctness_only
                or os.environ.get("PTX_BENCH_CORRECTNESS_ONLY") == "1"
            ):
                rows: list[dict[str, object]] = []
                speedup = 1.0
                performance = 1.0
            else:
                rows, speedup = benchmark(cuda, candidate, reference)
                performance = max(0.0, min(1.0, speedup))

            metrics = {
                "status": "passed",
                "gpu": cuda.device_name(),
                "candidate_path": str(args.candidate),
                "candidate_bytes": args.candidate.stat().st_size,
                "correctness_sizes": sizes,
                "compiler": compiler,
                "performance_rows": rows,
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
                "PTX_VECTOR_ADD_OK: "
                f"{len(sizes)} correctness cases; "
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
        print(f"PTX_VECTOR_ADD_FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
