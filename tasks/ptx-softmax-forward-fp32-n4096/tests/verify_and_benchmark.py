#!/usr/bin/env python3
"""Correctness-gated B200 verifier for row-wise FP32 Softmax N=4096."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import random
import statistics
from array import array
from pathlib import Path


ENTRY = b"softmax_fwd_fp32_n4096"
WIDTH = 4096
BLOCK_SIZE = 1024
DYNAMIC_SHARED_BYTES = 0
MAX_PTX_BYTES = 1 << 20
GUARD_BYTES = 256
ABS_TOL = 2e-6
REL_TOL = 2e-3
ROW_SUM_TOL = 2e-4
RANDOM_SEED = 0x534F4654
CORRECTNESS_ROWS = [0, 1, 3, 17, 128]
PERFORMANCE_ROWS = [1, 32, 256]
OUTPUT_SENTINEL = 0x7FC00000


class CudaError(RuntimeError):
    """CUDA Driver API failure outside the Candidate-owned boundary."""


class CandidateRejected(Exception):
    """The submitted PTX is invalid, unsafe, or computes a wrong result."""


def candidate_call(operation: str, function, *args, **kwargs):
    """Convert a Candidate-owned failure into a scored rejection."""
    try:
        return function(*args, **kwargs)
    except CandidateRejected:
        raise
    except Exception as error:
        raise CandidateRejected(f"{operation}: {error}") from error


class CudaDriver:
    """Minimal CUDA Driver API wrapper with no third-party dependency."""

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

    def load_function(self, path: Path) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
        module = ctypes.c_void_p()
        function = ctypes.c_void_p()
        self.check(
            self.lib.cuModuleLoad(ctypes.byref(module), str(path).encode()),
            f"cuModuleLoad({path})",
        )
        try:
            self.check(
                self.lib.cuModuleGetFunction(ctypes.byref(function), module, ENTRY),
                f"cuModuleGetFunction({ENTRY.decode()})",
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
        self.check(self.lib.cuMemsetD32_v2(destination, value, count), "cuMemsetD32")

    def launch(
        self,
        function: ctypes.c_void_p,
        input_pointer: int,
        output_pointer: int,
        num_rows: int,
    ) -> None:
        raw = [
            ctypes.c_uint64(input_pointer),
            ctypes.c_uint64(output_pointer),
            ctypes.c_uint32(num_rows),
        ]
        arguments = (ctypes.c_void_p * len(raw))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in raw)
        )
        self.check(
            self.lib.cuLaunchKernel(
                function,
                max(1, num_rows),
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
        input_pointer: int,
        output_pointer: int,
        num_rows: int,
        iterations: int,
    ) -> float:
        start = ctypes.c_void_p()
        end = ctypes.c_void_p()
        self.check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate(start)")
        self.check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate(end)")
        try:
            self.check(self.lib.cuEventRecord(start, None), "cuEventRecord(start)")
            for _ in range(iterations):
                self.launch(function, input_pointer, output_pointer, num_rows)
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


def stable_softmax(row: list[float]) -> list[float]:
    maximum = max(row)
    exponentials = [math.exp(float(value) - maximum) for value in row]
    denominator = math.fsum(exponentials)
    return [value / denominator for value in exponentials]


def make_case(num_rows: int, seed: int) -> tuple[array, list[float]]:
    rng = random.Random(seed)
    input_values = array("f")
    expected: list[float] = []
    anchors = (-100.0, -20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0, 100.0)
    for row_index in range(num_rows):
        if row_index == 0:
            row = [0.0] * WIDTH
        elif row_index == 1:
            row = [3.25] * WIDTH
        elif row_index == 2:
            row = [
                f32(anchors[column % len(anchors)] + (column % 7) * 0.03125)
                for column in range(WIDTH)
            ]
        else:
            offset = (-100.0, 0.0, 100.0)[row_index % 3]
            row = [f32(offset + rng.gauss(0.0, 4.0)) for _ in range(WIDTH)]
            row[(row_index * 977 + 13) % WIDTH] = f32(offset + 24.0)
        input_values.extend(row)
        expected.extend(stable_softmax(row))
    return input_values, expected


def make_performance_row(seed: int) -> tuple[array, list[float]]:
    rng = random.Random(seed)
    row = [f32(rng.uniform(-8.0, 8.0)) for _ in range(WIDTH)]
    row[3079] = 16.0
    return array("f", row), stable_softmax(row)


def guarded(payload: bytes, byte: int) -> bytearray:
    guard = bytearray([byte]) * GUARD_BYTES
    return guard + payload + guard


def assert_guards(payload: bytes, byte: int, label: str) -> None:
    guard = bytes([byte]) * GUARD_BYTES
    if payload[:GUARD_BYTES] != guard or payload[-GUARD_BYTES:] != guard:
        raise CandidateRejected(f"{label}: memory guard changed")


def assert_float_output(payload: bytes, expected: list[float], label: str) -> None:
    actual = array("f")
    actual.frombytes(payload)
    if len(actual) != len(expected):
        raise RuntimeError(f"{label}: verifier output length mismatch")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if (
            not math.isfinite(got)
            or got < 0.0
            or not math.isclose(got, want, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        ):
            raise CandidateRejected(
                f"{label}: wrong output at index={index}: expected {want}, got {got}"
            )
    for row_index in range(len(actual) // WIDTH):
        row_sum = math.fsum(actual[row_index * WIDTH : (row_index + 1) * WIDTH])
        if not math.isclose(row_sum, 1.0, rel_tol=0.0, abs_tol=ROW_SUM_TOL):
            raise CandidateRejected(
                f"{label}: row {row_index} sums to {row_sum}, expected 1"
            )


def validate_function(cuda: CudaDriver, function: ctypes.c_void_p) -> None:
    for case_index, num_rows in enumerate(CORRECTNESS_ROWS):
        input_values, expected = make_case(num_rows, RANDOM_SEED + case_index * 0x101)
        input_bytes = input_values.tobytes()
        output_elements = num_rows * WIDTH
        output_bits = array("I", [OUTPUT_SENTINEL]) * output_elements
        hosts = {
            "input": guarded(input_bytes, 0xA5),
            "output": guarded(output_bits.tobytes(), 0x5A),
        }
        devices = {name: cuda.alloc(len(value)) for name, value in hosts.items()}
        after: dict[str, bytes] = {}
        try:
            for name, value in hosts.items():
                cuda.copy_to_device(devices[name], value)
            candidate_call(
                f"Candidate launch failed for num_rows={num_rows}",
                cuda.launch,
                function,
                devices["input"] + GUARD_BYTES,
                devices["output"] + GUARD_BYTES,
                num_rows,
            )
            candidate_call(
                f"Candidate synchronization failed for num_rows={num_rows}",
                cuda.synchronize,
            )
            for name, value in hosts.items():
                after[name] = candidate_call(
                    f"Candidate result copy failed for {name}, num_rows={num_rows}",
                    cuda.copy_from_device,
                    devices[name],
                    len(value),
                )
        except BaseException:
            for pointer in reversed(tuple(devices.values())):
                try:
                    cuda.free(pointer)
                except Exception:
                    pass
            raise
        else:
            for pointer in reversed(tuple(devices.values())):
                candidate_call(
                    f"Candidate cleanup failed for num_rows={num_rows}", cuda.free, pointer
                )
        if after["input"] != bytes(hosts["input"]):
            raise CandidateRejected(f"input was modified for num_rows={num_rows}")
        assert_guards(after["output"], 0x5A, f"output rows={num_rows}")
        assert_float_output(
            after["output"][GUARD_BYTES : GUARD_BYTES + output_elements * 4],
            expected,
            f"output rows={num_rows}",
        )


def validate_performance_samples(
    cuda: CudaDriver,
    function: ctypes.c_void_p,
    input_pointer: int,
    output_pointer: int,
    num_rows: int,
    expected_input: bytes,
    expected_output_row: list[float],
    label: str,
) -> None:
    cuda.memset_d32(output_pointer, OUTPUT_SENTINEL, num_rows * WIDTH)
    cuda.launch(function, input_pointer, output_pointer, num_rows)
    cuda.synchronize()
    if cuda.copy_from_device(input_pointer, len(expected_input)) != expected_input:
        raise CandidateRejected(
            f"{label}: performance input was modified for num_rows={num_rows}"
        )
    for row_index in sorted({0, num_rows // 2, num_rows - 1}):
        payload = cuda.copy_from_device(
            output_pointer + row_index * WIDTH * 4, WIDTH * 4
        )
        assert_float_output(
            payload,
            expected_output_row,
            f"{label} performance row={row_index}, num_rows={num_rows}",
        )


def dispersion(values: list[float]) -> float:
    median = statistics.median(values)
    return math.inf if median == 0 else (max(values) - min(values)) / median


def benchmark_batches(
    cuda: CudaDriver,
    candidate: ctypes.c_void_p,
    reference: ctypes.c_void_p,
    input_pointer: int,
    output_pointer: int,
    num_rows: int,
    iterations: int,
) -> tuple[list[float], list[float]]:
    timings: dict[str, list[float]] = {"candidate": [], "reference": []}
    for batch_index in range(5):
        order = (
            (("candidate", candidate), ("reference", reference))
            if batch_index % 2 == 0
            else (("reference", reference), ("candidate", candidate))
        )
        for name, function in order:
            if name == "candidate":
                elapsed = candidate_call(
                    f"Candidate timing failed for num_rows={num_rows}",
                    cuda.measure_ms,
                    function,
                    input_pointer,
                    output_pointer,
                    num_rows,
                    iterations,
                )
            else:
                elapsed = cuda.measure_ms(
                    function,
                    input_pointer,
                    output_pointer,
                    num_rows,
                    iterations,
                )
            timings[name].append(elapsed)
    return timings["candidate"], timings["reference"]


def benchmark_shape(
    cuda: CudaDriver,
    candidate: ctypes.c_void_p,
    reference: ctypes.c_void_p,
    num_rows: int,
) -> dict[str, object]:
    input_pointer = cuda.alloc(num_rows * WIDTH * 4)
    output_pointer = cuda.alloc(num_rows * WIDTH * 4)
    try:
        row_values, expected_output_row = make_performance_row(RANDOM_SEED ^ num_rows)
        expected_input = row_values.tobytes() * num_rows
        cuda.copy_to_device(input_pointer, expected_input)
        candidate_call(
            f"Candidate performance-shape validation failed for num_rows={num_rows}",
            validate_performance_samples,
            cuda,
            candidate,
            input_pointer,
            output_pointer,
            num_rows,
            expected_input,
            expected_output_row,
            "candidate",
        )
        try:
            validate_performance_samples(
                cuda,
                reference,
                input_pointer,
                output_pointer,
                num_rows,
                expected_input,
                expected_output_row,
                "reference",
            )
        except CandidateRejected as error:
            raise RuntimeError(
                f"Reference performance-shape validation failed: {error}"
            ) from error
        for _ in range(10):
            candidate_call(
                f"Candidate warmup failed for num_rows={num_rows}",
                cuda.launch,
                candidate,
                input_pointer,
                output_pointer,
                num_rows,
            )
            cuda.launch(reference, input_pointer, output_pointer, num_rows)
        cuda.synchronize()
        candidate_probe = candidate_call(
            f"Candidate timing probe failed for num_rows={num_rows}",
            cuda.measure_ms,
            candidate,
            input_pointer,
            output_pointer,
            num_rows,
            10,
        )
        reference_probe = cuda.measure_ms(
            reference, input_pointer, output_pointer, num_rows, 10
        )
        iterations = max(
            10,
            min(20000, math.ceil(30.0 / max(candidate_probe, reference_probe, 1e-6))),
        )
        candidate_times, reference_times = benchmark_batches(
            cuda,
            candidate,
            reference,
            input_pointer,
            output_pointer,
            num_rows,
            iterations,
        )
        repeated = False
        if max(dispersion(candidate_times), dispersion(reference_times)) > 0.05:
            candidate_times, reference_times = benchmark_batches(
                cuda,
                candidate,
                reference,
                input_pointer,
                output_pointer,
                num_rows,
                iterations,
            )
            repeated = True
        candidate_median = statistics.median(candidate_times)
        reference_median = statistics.median(reference_times)
        ratio = reference_median / candidate_median
        minimum_bytes = num_rows * WIDTH * 8
        result: dict[str, object] = {
            "num_rows": num_rows,
            "scored": True,
            "iterations_per_batch": iterations,
            "candidate_ms_per_launch": candidate_times,
            "reference_ms_per_launch": reference_times,
            "candidate_median_ms": candidate_median,
            "reference_median_ms": reference_median,
            "speedup_vs_reference": ratio,
            "candidate_logical_bandwidth_gbps": minimum_bytes
            / (candidate_median * 1e6),
            "reference_logical_bandwidth_gbps": minimum_bytes
            / (reference_median * 1e6),
            "candidate_dispersion": dispersion(candidate_times),
            "reference_dispersion": dispersion(reference_times),
            "repeated_for_instability": repeated,
        }
    except BaseException:
        for pointer in (output_pointer, input_pointer):
            try:
                cuda.free(pointer)
            except Exception:
                pass
        raise
    else:
        for pointer in (output_pointer, input_pointer):
            candidate_call(
                f"Candidate benchmark cleanup failed for num_rows={num_rows}",
                cuda.free,
                pointer,
            )
    return result


def benchmark(
    cuda: CudaDriver,
    candidate: ctypes.c_void_p,
    reference: ctypes.c_void_p,
) -> tuple[list[dict[str, object]], float]:
    rows = [
        benchmark_shape(cuda, candidate, reference, num_rows)
        for num_rows in PERFORMANCE_ROWS
    ]
    ratios = [float(row["speedup_vs_reference"]) for row in rows]
    geometric_mean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    return rows, geometric_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--correctness-only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_submission_file(path: Path) -> None:
    if not path.is_file():
        raise CandidateRejected(f"missing Candidate PTX: {path}")
    size = path.stat().st_size
    if size == 0:
        raise CandidateRejected("Candidate PTX is empty")
    if size > MAX_PTX_BYTES:
        raise CandidateRejected(f"Candidate PTX exceeds {MAX_PTX_BYTES} bytes")
    try:
        source = path.read_text()
    except UnicodeDecodeError as error:
        raise CandidateRejected("Candidate PTX is not UTF-8 text") from error
    if ".entry" not in source or ENTRY.decode() not in source:
        raise CandidateRejected(f"Candidate PTX is missing entry {ENTRY.decode()}")


def validate_reference_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing fixed Reference PTX: {path}")
    size = path.stat().st_size
    if size == 0 or size > MAX_PTX_BYTES:
        raise RuntimeError("fixed Reference PTX has invalid size")
    try:
        source = path.read_text()
    except UnicodeDecodeError as error:
        raise RuntimeError("fixed Reference PTX is not UTF-8 text") from error
    if ".entry" not in source or ENTRY.decode() not in source:
        raise RuntimeError(f"fixed Reference PTX is missing entry {ENTRY.decode()}")


def run(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, float]]:
    validate_submission_file(args.candidate)
    validate_reference_file(args.reference)
    cuda = CudaDriver()
    candidate_module = ctypes.c_void_p()
    reference_module = ctypes.c_void_p()
    try:
        reference_module, reference = cuda.load_function(args.reference)
        try:
            validate_function(cuda, reference)
        except CandidateRejected as error:
            raise RuntimeError(f"fixed Reference correctness failed: {error}") from error
        candidate_module, candidate = candidate_call(
            "Candidate module load failed", cuda.load_function, args.candidate
        )
        candidate_call(
            "Candidate correctness validation failed", validate_function, cuda, candidate
        )
        metrics: dict[str, object] = {
            "status": "correctness_passed",
            "validation_level": "correctness-only",
            "gpu": cuda.device_name(),
            "entry": ENTRY.decode(),
            "candidate_bytes": args.candidate.stat().st_size,
            "reference_bytes": args.reference.stat().st_size,
            "correctness_rows": CORRECTNESS_ROWS,
            "atol": ABS_TOL,
            "rtol": REL_TOL,
            "row_sum_atol": ROW_SUM_TOL,
        }
        if args.correctness_only:
            reward: dict[str, float] = {
                "reward": 0.0,
                "correctness": 1.0,
                "performance": 0.0,
                "speedup_vs_reference": 0.0,
            }
        else:
            timing, raw_speedup = benchmark(cuda, candidate, reference)
            performance = min(1.0, max(0.0, raw_speedup))
            metrics.update(
                status="passed",
                validation_level="correctness-and-performance",
                timing=timing,
                raw_speedup_vs_reference=raw_speedup,
                performance_capped_at_one=performance,
            )
            reward = {
                "reward": performance,
                "correctness": 1.0,
                "performance": performance,
                "speedup_vs_reference": raw_speedup,
            }
        return metrics, reward
    finally:
        if candidate_module.value:
            cuda.lib.cuModuleUnload(candidate_module)
        if reference_module.value:
            cuda.lib.cuModuleUnload(reference_module)
        cuda.close()


def main() -> int:
    args = parse_args()
    reward: dict[str, float] = {
        "reward": 0.0,
        "correctness": 0.0,
        "performance": 0.0,
        "speedup_vs_reference": 0.0,
    }
    metrics: dict[str, object] = {
        "status": "failed",
        "validation_level": "none",
        "entry": ENTRY.decode(),
    }
    try:
        metrics, reward = run(args)
    except CandidateRejected as error:
        metrics.update(
            status="candidate_rejected",
            error_type=type(error).__name__,
            error=str(error),
        )
        print(f"PTX verification failed: {type(error).__name__}: {error}")
        write_json(args.log, metrics)
        write_json(args.reward, reward)
        return 1
    except Exception as error:
        metrics.update(
            status="infrastructure_error",
            error_type=type(error).__name__,
            error=str(error),
        )
        print(f"PTX verifier infrastructure failure: {type(error).__name__}: {error}")
        write_json(args.log, metrics)
        write_json(args.reward, reward)
        return 2
    write_json(args.log, metrics)
    write_json(args.reward, reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
