#!/usr/bin/env python3
"""Correctness-gated B200 verifier for BF16 SiLU-gate with N=4096."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import random
import statistics
import struct
import sys
from array import array
from pathlib import Path


ENTRY = b"silu_gate_fwd_bf16_n4096"
OUTPUT_WIDTH = 4096
INPUT_WIDTH = 2 * OUTPUT_WIDTH
BLOCK_SIZE = 512
DYNAMIC_SHARED_BYTES = 0
MAX_PTX_BYTES = 1 << 20
GUARD_BYTES = 256
ABS_TOL = 3e-2
REL_TOL = 1e-2
RANDOM_SEED = 0x53494C55
CORRECTNESS_TOKENS = [0, 1, 2, 3, 7, 31, 257]
DIAGNOSTIC_TOKENS = [1]
PERFORMANCE_TOKENS = [32, 256, 4096]
OUTPUT_SENTINEL = 0xFFFF  # BF16 NaN: an unwritten output is always rejected.


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
        lib.cuMemsetD16_v2.argtypes = [
            ctypes.c_uint64,
            ctypes.c_ushort,
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
        lib.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]

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

    def memset_d16(self, destination: int, value: int, count: int) -> None:
        self.check(
            self.lib.cuMemsetD16_v2(destination, value, count),
            "cuMemsetD16",
        )

    def launch(
        self,
        function: ctypes.c_void_p,
        input_pointer: int,
        output_pointer: int,
        num_tokens: int,
    ) -> None:
        raw = [
            ctypes.c_uint64(input_pointer),
            ctypes.c_uint64(output_pointer),
            ctypes.c_uint32(num_tokens),
        ]
        arguments = (ctypes.c_void_p * len(raw))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in raw)
        )
        self.check(
            self.lib.cuLaunchKernel(
                function,
                max(1, num_tokens),
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
        num_tokens: int,
        iterations: int,
    ) -> float:
        start = ctypes.c_void_p()
        end = ctypes.c_void_p()
        self.check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate(start)")
        self.check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate(end)")
        try:
            self.check(self.lib.cuEventRecord(start, None), "cuEventRecord(start)")
            for _ in range(iterations):
                self.launch(function, input_pointer, output_pointer, num_tokens)
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
    """Round a Python float to IEEE float32."""
    return ctypes.c_float(value).value


def float_to_bf16(value: float) -> int:
    """Round finite float32 to BF16 with round-to-nearest-even."""
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def bf16_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def cpu_silu_gate(gate: float, up: float) -> float:
    """Independent high-precision formula rounded to the task output dtype."""
    activated = gate / (1.0 + math.exp(-gate))
    return bf16_to_float(float_to_bf16(activated * up))


def make_case(num_tokens: int, seed: int) -> tuple[array, list[float]]:
    """Create deterministic finite BF16 inputs and independent expected values."""
    rng = random.Random(seed)
    gate_anchors = (-20.0, -8.0, -4.0, -1.0, -0.0, 0.0, 1.0, 4.0, 8.0, 20.0)
    up_anchors = (0.0, -3.0, 0.5, 1.0, -1.25, 4.0)
    input_bits = array("H")
    expected: list[float] = []

    for token in range(num_tokens):
        gates: list[float] = []
        ups: list[float] = []
        for index in range(OUTPUT_WIDTH):
            if token == 0:
                gate_source = gate_anchors[index % len(gate_anchors)]
                up_source = up_anchors[(index // len(gate_anchors)) % len(up_anchors)]
            elif token == 1:
                gate_source = 1.25
                up_source = -0.75
            else:
                gate_source = max(-10.0, min(10.0, rng.gauss(0.0, 3.0)))
                up_source = rng.uniform(-4.0, 4.0)

            gate = bf16_to_float(float_to_bf16(gate_source))
            up = bf16_to_float(float_to_bf16(up_source))
            gates.append(gate)
            ups.append(up)
            expected.append(cpu_silu_gate(gate, up))

        input_bits.extend(float_to_bf16(value) for value in gates)
        input_bits.extend(float_to_bf16(value) for value in ups)

    return input_bits, expected


def guarded(payload: bytes, byte: int) -> bytearray:
    guard = bytearray([byte]) * GUARD_BYTES
    return guard + payload + guard


def assert_guards(payload: bytes, byte: int, label: str) -> None:
    guard = bytes([byte]) * GUARD_BYTES
    if payload[:GUARD_BYTES] != guard or payload[-GUARD_BYTES:] != guard:
        raise CandidateRejected(f"{label}: memory guard changed")


def assert_bf16_output(payload: bytes, expected: list[float], label: str) -> None:
    actual = array("H")
    actual.frombytes(payload)
    if len(actual) != len(expected):
        raise RuntimeError(f"{label}: verifier output length mismatch")
    for index, (bits, want) in enumerate(zip(actual, expected)):
        got = bf16_to_float(bits)
        if not math.isfinite(got) or not math.isclose(
            got,
            want,
            rel_tol=REL_TOL,
            abs_tol=ABS_TOL,
        ):
            raise CandidateRejected(
                f"{label}: wrong output at index={index}: expected {want}, got {got}"
            )


def validate_function(cuda: CudaDriver, function: ctypes.c_void_p) -> None:
    """Run hidden correctness, input-immutability, and guard checks."""
    for case_index, num_tokens in enumerate(CORRECTNESS_TOKENS):
        input_bits, expected = make_case(
            num_tokens,
            RANDOM_SEED + case_index * 0x101,
        )
        input_bytes = input_bits.tobytes()
        output_elements = num_tokens * OUTPUT_WIDTH
        output_bits = array("H", [OUTPUT_SENTINEL]) * output_elements
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
                f"Candidate launch failed for num_tokens={num_tokens}",
                cuda.launch,
                function,
                devices["input"] + GUARD_BYTES,
                devices["output"] + GUARD_BYTES,
                num_tokens,
            )
            candidate_call(
                f"Candidate synchronization failed for num_tokens={num_tokens}",
                cuda.synchronize,
            )
            for name, value in hosts.items():
                after[name] = candidate_call(
                    f"Candidate result copy failed for {name}, num_tokens={num_tokens}",
                    cuda.copy_from_device,
                    devices[name],
                    len(value),
                )
        except BaseException:
            # Preserve the primary Candidate failure. A bad kernel can poison
            # the CUDA context, so cleanup may fail as a consequence and must
            # not relabel the attempt as verifier infrastructure failure.
            for pointer in reversed(tuple(devices.values())):
                try:
                    cuda.free(pointer)
                except Exception:
                    pass
            raise
        else:
            for pointer in reversed(tuple(devices.values())):
                candidate_call(
                    f"Candidate cleanup failed for num_tokens={num_tokens}",
                    cuda.free,
                    pointer,
                )

        if after["input"] != bytes(hosts["input"]):
            raise CandidateRejected(f"input was modified for num_tokens={num_tokens}")
        assert_guards(after["output"], 0x5A, f"output tokens={num_tokens}")
        assert_bf16_output(
            after["output"][GUARD_BYTES : GUARD_BYTES + output_elements * 2],
            expected,
            f"output tokens={num_tokens}",
        )


def validate_performance_samples(
    cuda: CudaDriver,
    function: ctypes.c_void_p,
    input_pointer: int,
    output_pointer: int,
    num_tokens: int,
    expected_input: bytes,
    expected_output_row: list[float],
    label: str,
) -> None:
    """Check representative first/middle/last rows before timing a large shape."""
    cuda.memset_d16(
        output_pointer,
        OUTPUT_SENTINEL,
        num_tokens * OUTPUT_WIDTH,
    )
    cuda.launch(function, input_pointer, output_pointer, num_tokens)
    cuda.synchronize()
    input_after = cuda.copy_from_device(input_pointer, len(expected_input))
    if input_after != expected_input:
        raise CandidateRejected(
            f"{label}: performance input was modified for num_tokens={num_tokens}"
        )
    for token in sorted({0, num_tokens // 2, num_tokens - 1}):
        for column in (0, 1, 511, 512, OUTPUT_WIDTH - 1):
            index = token * OUTPUT_WIDTH + column
            bits = struct.unpack(
                "<H",
                cuda.copy_from_device(output_pointer + index * 2, 2),
            )[0]
            got = bf16_to_float(bits)
            expected = expected_output_row[column]
            if not math.isfinite(got) or not math.isclose(
                got,
                expected,
                rel_tol=REL_TOL,
                abs_tol=ABS_TOL,
            ):
                raise CandidateRejected(
                    f"{label}: performance-shape check failed at "
                    f"num_tokens={num_tokens}, token={token}, column={column}: "
                    f"expected {expected}, got {got}"
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
    num_tokens: int,
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
                    f"Candidate timing failed for num_tokens={num_tokens}",
                    cuda.measure_ms,
                    function,
                    input_pointer,
                    output_pointer,
                    num_tokens,
                    iterations,
                )
            else:
                elapsed = cuda.measure_ms(
                    function,
                    input_pointer,
                    output_pointer,
                    num_tokens,
                    iterations,
                )
            timings[name].append(elapsed)
    return timings["candidate"], timings["reference"]


def benchmark_shape(
    cuda: CudaDriver,
    candidate: ctypes.c_void_p,
    reference: ctypes.c_void_p,
    num_tokens: int,
    scored: bool,
) -> dict[str, object]:
    input_elements = num_tokens * INPUT_WIDTH
    output_elements = num_tokens * OUTPUT_WIDTH
    input_pointer = cuda.alloc(input_elements * 2)
    output_pointer = cuda.alloc(output_elements * 2)
    try:
        row_bits, expected_output_row = make_case(
            1,
            RANDOM_SEED ^ num_tokens,
        )
        expected_input = row_bits.tobytes() * num_tokens
        cuda.copy_to_device(input_pointer, expected_input)
        candidate_call(
            f"Candidate performance-shape validation failed for num_tokens={num_tokens}",
            validate_performance_samples,
            cuda,
            candidate,
            input_pointer,
            output_pointer,
            num_tokens,
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
                num_tokens,
                expected_input,
                expected_output_row,
                "reference",
            )
        except CandidateRejected as error:
            raise RuntimeError(f"Reference performance-shape validation failed: {error}") from error

        for _ in range(10):
            candidate_call(
                f"Candidate warmup failed for num_tokens={num_tokens}",
                cuda.launch,
                candidate,
                input_pointer,
                output_pointer,
                num_tokens,
            )
            cuda.launch(reference, input_pointer, output_pointer, num_tokens)
        cuda.synchronize()

        candidate_probe = candidate_call(
            f"Candidate timing probe failed for num_tokens={num_tokens}",
            cuda.measure_ms,
            candidate,
            input_pointer,
            output_pointer,
            num_tokens,
            10,
        )
        reference_probe = cuda.measure_ms(
            reference,
            input_pointer,
            output_pointer,
            num_tokens,
            10,
        )
        slower_probe = max(candidate_probe, reference_probe, 1e-6)
        iterations = max(10, min(20000, math.ceil(30.0 / slower_probe)))
        candidate_times, reference_times = benchmark_batches(
            cuda,
            candidate,
            reference,
            input_pointer,
            output_pointer,
            num_tokens,
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
                num_tokens,
                iterations,
            )
            repeated = True

        candidate_median = statistics.median(candidate_times)
        reference_median = statistics.median(reference_times)
        ratio = reference_median / candidate_median
        minimum_bytes = num_tokens * OUTPUT_WIDTH * 6
        result: dict[str, object] = {
            "num_tokens": num_tokens,
            "scored": scored,
            "iterations_per_batch": iterations,
            "candidate_ms_per_launch": candidate_times,
            "reference_ms_per_launch": reference_times,
            "candidate_median_ms": candidate_median,
            "reference_median_ms": reference_median,
            "speedup_vs_reference": ratio,
            "candidate_effective_bandwidth_gbps": minimum_bytes
            / (candidate_median * 1e6),
            "reference_effective_bandwidth_gbps": minimum_bytes
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
                f"Candidate benchmark cleanup failed for num_tokens={num_tokens}",
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
        benchmark_shape(cuda, candidate, reference, num_tokens, scored=False)
        for num_tokens in DIAGNOSTIC_TOKENS
    ]
    rows.extend(
        benchmark_shape(cuda, candidate, reference, num_tokens, scored=True)
        for num_tokens in PERFORMANCE_TOKENS
    )
    scored_ratios = [
        float(row["speedup_vs_reference"]) for row in rows if row["scored"]
    ]
    geometric_mean = math.exp(
        sum(math.log(value) for value in scored_ratios) / len(scored_ratios)
    )
    return rows, geometric_mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Validate Candidate and Reference correctness without timing.",
    )
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
    if size == 0:
        raise RuntimeError("fixed Reference PTX is empty")
    if size > MAX_PTX_BYTES:
        raise RuntimeError(f"fixed Reference PTX exceeds {MAX_PTX_BYTES} bytes")
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
            "Candidate module load failed",
            cuda.load_function,
            args.candidate,
        )
        candidate_call(
            "Candidate correctness validation failed",
            validate_function,
            cuda,
            candidate,
        )

        metrics: dict[str, object] = {
            "status": "correctness_passed",
            "validation_level": "correctness-only",
            "gpu": cuda.device_name(),
            "entry": ENTRY.decode(),
            "candidate_bytes": args.candidate.stat().st_size,
            "reference_bytes": args.reference.stat().st_size,
            "correctness_tokens": CORRECTNESS_TOKENS,
            "atol": ABS_TOL,
            "rtol": REL_TOL,
        }
        if args.correctness_only:
            reward: dict[str, float] = {
                "reward": 0.0,
                "correctness": 1.0,
                "performance": 0.0,
                "speedup_vs_reference": 0.0,
            }
        else:
            rows, speedup = benchmark(cuda, candidate, reference)
            performance = max(0.0, min(1.0, speedup))
            reward = {
                "reward": performance,
                "correctness": 1.0,
                "performance": performance,
                "speedup_vs_reference": speedup,
            }
            metrics.update(
                {
                    "status": "passed",
                    "validation_level": "correctness-and-performance",
                    "diagnostic_tokens": DIAGNOSTIC_TOKENS,
                    "performance_tokens": PERFORMANCE_TOKENS,
                    "performance_rows": rows,
                    "speedup_vs_reference": speedup,
                }
            )
        return metrics, reward
    finally:
        if reference_module.value:
            cuda.lib.cuModuleUnload(reference_module)
        if candidate_module.value:
            cuda.lib.cuModuleUnload(candidate_module)
        cuda.close()


def zero_reward() -> dict[str, float]:
    return {
        "reward": 0.0,
        "correctness": 0.0,
        "performance": 0.0,
        "speedup_vs_reference": 0.0,
    }


def main() -> int:
    args = parse_args()
    try:
        metrics, reward = run(args)
        write_json(args.log, metrics)
        write_json(args.reward, reward)
        print(
            "PTX_SILU_VERIFY_OK: "
            f"{len(CORRECTNESS_TOKENS)} correctness cases; "
            f"speedup_vs_reference={reward['speedup_vs_reference']:.6f}; "
            f"reward={reward['reward']:.6f}"
        )
        return 0
    except CandidateRejected as error:
        try:
            write_json(
                args.log,
                {
                    "status": "candidate_rejected",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            write_json(args.reward, zero_reward())
        except Exception:
            pass
        print(f"PTX_SILU_CANDIDATE_REJECTED: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        try:
            write_json(
                args.log,
                {
                    "status": "infrastructure_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            write_json(args.reward, zero_reward())
        except Exception:
            pass
        print(
            f"PTX_SILU_INFRASTRUCTURE_ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
