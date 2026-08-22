#!/usr/bin/env python3
"""B200 feasibility verifier for BF16 NeoX RoPE with head dimension 128."""

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


ENTRY = b"rope_forward_bf16_d128"
Q_HEADS = 32
K_HEADS = 8
HEAD_DIM = 128
HALF_DIM = 64
CACHE_ROWS = 8192
CACHE_ELEMENTS = CACHE_ROWS * HEAD_DIM
BLOCK_SIZE = 256
DYNAMIC_SHARED_BYTES = 512
MAX_PTX_BYTES = 1 << 20
GUARD_BYTES = 256
ABS_TOL = 1e-2
REL_TOL = 1e-2
RANDOM_SEED = 0x524F5045
CORRECTNESS_TOKENS = [0, 1, 2, 3, 7, 31, 257]
PERFORMANCE_TOKENS = [256, 4096, 32768]


class CudaError(RuntimeError):
    pass


class CandidateRejected(Exception):
    """The submitted PTX is invalid or produces a wrong result."""


def candidate_call(operation: str, function, *args, **kwargs):
    """Run Candidate-owned work while preserving infrastructure failures elsewhere."""
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
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int
        ]
        lib.cuCtxSynchronize.argtypes = []
        lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
        ]
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemcpyHtoD_v2.argtypes = [
            ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t
        ]
        lib.cuMemcpyDtoH_v2.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t
        ]
        lib.cuMemsetD16_v2.argtypes = [ctypes.c_uint64, ctypes.c_ushort, ctypes.c_size_t]
        lib.cuMemsetD32_v2.argtypes = [ctypes.c_uint64, ctypes.c_uint, ctypes.c_size_t]
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        lib.cuEventElapsedTime.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p
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
            self.lib.cuMemcpyHtoD_v2(destination, host, len(data)), "cuMemcpyHtoD"
        )

    def copy_from_device(self, source: int, size: int) -> bytes:
        host = (ctypes.c_ubyte * size)()
        self.check(self.lib.cuMemcpyDtoH_v2(host, source, size), "cuMemcpyDtoH")
        return bytes(host)

    def memset_d16(self, destination: int, value: int, count: int) -> None:
        self.check(self.lib.cuMemsetD16_v2(destination, value, count), "cuMemsetD16")

    def memset_d32(self, destination: int, value: int, count: int) -> None:
        self.check(self.lib.cuMemsetD32_v2(destination, value, count), "cuMemsetD32")

    def launch(
        self,
        function: ctypes.c_void_p,
        q: int,
        k: int,
        positions: int,
        cache: int,
        q_out: int,
        k_out: int,
        num_tokens: int,
    ) -> None:
        raw = [
            ctypes.c_uint64(q), ctypes.c_uint64(k), ctypes.c_uint64(positions),
            ctypes.c_uint64(cache), ctypes.c_uint64(q_out), ctypes.c_uint64(k_out),
            ctypes.c_uint32(num_tokens),
        ]
        arguments = (ctypes.c_void_p * len(raw))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in raw)
        )
        self.check(
            self.lib.cuLaunchKernel(
                function,
                max(1, num_tokens), 1, 1,
                BLOCK_SIZE, 1, 1,
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
        buffers: tuple[int, int, int, int, int, int],
        num_tokens: int,
        iterations: int,
    ) -> float:
        start = ctypes.c_void_p()
        end = ctypes.c_void_p()
        self.check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
        self.check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate")
        try:
            self.check(self.lib.cuEventRecord(start, None), "cuEventRecord(start)")
            for _ in range(iterations):
                self.launch(function, *buffers, num_tokens)
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


def guarded(payload: bytes, byte: int) -> bytearray:
    return bytearray([byte]) * GUARD_BYTES + payload + bytearray([byte]) * GUARD_BYTES


def make_positions(num_tokens: int) -> array:
    anchors = [0, 8191, 37, 37, 1, 4096, 7, 1024, 63]
    return array(
        "I",
        (
            anchors[i] if i < len(anchors) else (i * 997) % CACHE_ROWS
            for i in range(num_tokens)
        ),
    )


def make_cache(positions: array) -> array:
    cache = array("f", [0.0]) * CACHE_ELEMENTS
    for position in set(positions):
        base = position * HEAD_DIM
        for dim in range(HALF_DIM):
            angle = f32((position + 1) * (dim + 1) * 0.0007)
            cache[base + dim] = f32(math.cos(angle))
            cache[base + HALF_DIM + dim] = f32(math.sin(angle))
    return cache


def make_input(num_tokens: int, heads: int, seed: int) -> tuple[array, list[float]]:
    rng = random.Random(seed)
    bits = array("H")
    values: list[float] = []
    for token in range(num_tokens):
        for _ in range(heads * HEAD_DIM):
            source = 0.0 if token == 0 else 1.25 if token == 1 else rng.uniform(-2, 2)
            encoded = float_to_bf16(source)
            bits.append(encoded)
            values.append(bf16_to_float(encoded))
    return bits, values


def expected_output(
    values: list[float], positions: array, cache: array, heads: int
) -> list[float]:
    expected = [0.0] * len(values)
    token_stride = heads * HEAD_DIM
    for token, position in enumerate(positions):
        cache_base = position * HEAD_DIM
        token_base = token * token_stride
        for head in range(heads):
            head_base = token_base + head * HEAD_DIM
            for dim in range(HALF_DIM):
                first = values[head_base + dim]
                second = values[head_base + HALF_DIM + dim]
                cosine = cache[cache_base + dim]
                sine = cache[cache_base + HALF_DIM + dim]
                rotated_first = f32(f32(first * cosine) - f32(second * sine))
                rotated_second = f32(f32(second * cosine) + f32(first * sine))
                expected[head_base + dim] = bf16_to_float(float_to_bf16(rotated_first))
                expected[head_base + HALF_DIM + dim] = bf16_to_float(
                    float_to_bf16(rotated_second)
                )
    return expected


def assert_guards(payload: bytes, byte: int, label: str) -> None:
    guard = bytes([byte]) * GUARD_BYTES
    if payload[:GUARD_BYTES] != guard or payload[-GUARD_BYTES:] != guard:
        raise AssertionError(f"{label}: memory guard changed")


def assert_bf16_output(payload: bytes, expected: list[float], label: str) -> None:
    actual = array("H")
    actual.frombytes(payload)
    if len(actual) != len(expected):
        raise AssertionError(f"{label}: output length mismatch")
    for index, (bits, want) in enumerate(zip(actual, expected)):
        got = bf16_to_float(bits)
        if not math.isfinite(got) or not math.isclose(
            got, want, rel_tol=REL_TOL, abs_tol=ABS_TOL
        ):
            raise AssertionError(
                f"{label}: wrong output at index={index}: expected {want}, got {got}"
            )


def validate_function(cuda: CudaDriver, function: ctypes.c_void_p, label: str) -> None:
    for case_index, num_tokens in enumerate(CORRECTNESS_TOKENS):
        positions = make_positions(num_tokens)
        cache = make_cache(positions)
        q_bits, q_values = make_input(num_tokens, Q_HEADS, RANDOM_SEED + case_index * 2)
        k_bits, k_values = make_input(
            num_tokens, K_HEADS, RANDOM_SEED + case_index * 2 + 1
        )
        q_expected = expected_output(q_values, positions, cache, Q_HEADS)
        k_expected = expected_output(k_values, positions, cache, K_HEADS)
        q_bytes = q_bits.tobytes()
        k_bytes = k_bits.tobytes()
        hosts = {
            "q": guarded(q_bytes, 0xA1),
            "k": guarded(k_bytes, 0xB2),
            "positions": guarded(positions.tobytes(), 0xC3),
            "cache": guarded(cache.tobytes(), 0xD4),
            "q_out": guarded(bytes([0xCD]) * len(q_bytes), 0xE5),
            "k_out": guarded(bytes([0xCD]) * len(k_bytes), 0xF6),
        }
        devices = {name: cuda.alloc(len(value)) for name, value in hosts.items()}
        try:
            for name, value in hosts.items():
                cuda.copy_to_device(devices[name], value)
            cuda.launch(
                function,
                devices["q"] + GUARD_BYTES,
                devices["k"] + GUARD_BYTES,
                devices["positions"] + GUARD_BYTES,
                devices["cache"] + GUARD_BYTES,
                devices["q_out"] + GUARD_BYTES,
                devices["k_out"] + GUARD_BYTES,
                num_tokens,
            )
            cuda.synchronize()
            after = {
                name: cuda.copy_from_device(devices[name], len(value))
                for name, value in hosts.items()
            }
        finally:
            for pointer in reversed(tuple(devices.values())):
                cuda.free(pointer)
        for name in ("q", "k", "positions", "cache"):
            if after[name] != bytes(hosts[name]):
                raise AssertionError(f"{label}: input {name} was modified")
        assert_guards(after["q_out"], 0xE5, f"{label} q_out tokens={num_tokens}")
        assert_guards(after["k_out"], 0xF6, f"{label} k_out tokens={num_tokens}")
        assert_bf16_output(
            after["q_out"][GUARD_BYTES : GUARD_BYTES + len(q_bytes)],
            q_expected,
            f"{label} q_out tokens={num_tokens}",
        )
        assert_bf16_output(
            after["k_out"][GUARD_BYTES : GUARD_BYTES + len(k_bytes)],
            k_expected,
            f"{label} k_out tokens={num_tokens}",
        )


def validate_identity_samples(
    cuda: CudaDriver,
    function: ctypes.c_void_p,
    buffers: tuple[int, int, int, int, int, int],
    num_tokens: int,
    label: str,
) -> None:
    cuda.launch(function, *buffers, num_tokens)
    cuda.synchronize()
    for token in sorted({0, num_tokens // 2, num_tokens - 1}):
        for output, heads, output_label in (
            (buffers[4], Q_HEADS, "q"),
            (buffers[5], K_HEADS, "k"),
        ):
            for head in (0, heads - 1):
                for dim in (0, 63, 64, 127):
                    index = (token * heads + head) * HEAD_DIM + dim
                    got = struct.unpack(
                        "<H", cuda.copy_from_device(output + index * 2, 2)
                    )[0]
                    if got != 0x3F80:
                        raise AssertionError(
                            f"{label}: identity check failed for {output_label}, "
                            f"tokens={num_tokens}, token={token}, head={head}, dim={dim}"
                        )


def dispersion(values: list[float]) -> float:
    median = statistics.median(values)
    return math.inf if median == 0 else (max(values) - min(values)) / median


def benchmark_batches(
    cuda: CudaDriver,
    candidate: ctypes.c_void_p,
    reference: ctypes.c_void_p,
    buffers: tuple[int, int, int, int, int, int],
    num_tokens: int,
    iterations: int,
) -> tuple[list[float], list[float]]:
    timings: dict[str, list[float]] = {"candidate": [], "reference": []}
    for batch_index in range(3):
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
                    buffers,
                    num_tokens,
                    iterations,
                )
            else:
                elapsed = cuda.measure_ms(function, buffers, num_tokens, iterations)
            timings[name].append(elapsed)
    return timings["candidate"], timings["reference"]


def benchmark(
    cuda: CudaDriver,
    candidate: ctypes.c_void_p,
    reference: ctypes.c_void_p,
) -> tuple[list[dict[str, object]], float]:
    results: list[dict[str, object]] = []
    ratios: list[float] = []
    for num_tokens in PERFORMANCE_TOKENS:
        q_elements = num_tokens * Q_HEADS * HEAD_DIM
        k_elements = num_tokens * K_HEADS * HEAD_DIM
        buffers = (
            cuda.alloc(q_elements * 2),
            cuda.alloc(k_elements * 2),
            cuda.alloc(num_tokens * 4),
            cuda.alloc(CACHE_ELEMENTS * 4),
            cuda.alloc(q_elements * 2),
            cuda.alloc(k_elements * 2),
        )
        try:
            cuda.memset_d16(buffers[0], 0x3F80, q_elements)
            cuda.memset_d16(buffers[1], 0x3F80, k_elements)
            cuda.memset_d32(buffers[2], 0, num_tokens)
            cuda.memset_d32(buffers[3], 0, CACHE_ELEMENTS)
            cuda.copy_to_device(buffers[3], array("f", [1.0] * HALF_DIM).tobytes())
            cuda.memset_d16(buffers[4], 0, q_elements)
            cuda.memset_d16(buffers[5], 0, k_elements)
            candidate_call(
                f"Candidate identity validation failed for num_tokens={num_tokens}",
                validate_identity_samples,
                cuda,
                candidate,
                buffers,
                num_tokens,
                "candidate",
            )
            validate_identity_samples(cuda, reference, buffers, num_tokens, "reference")
            for _ in range(5):
                candidate_call(
                    f"Candidate warmup failed for num_tokens={num_tokens}",
                    cuda.launch,
                    candidate,
                    *buffers,
                    num_tokens,
                )
                cuda.launch(reference, *buffers, num_tokens)
            cuda.synchronize()
            candidate_probe = candidate_call(
                f"Candidate timing probe failed for num_tokens={num_tokens}",
                cuda.measure_ms,
                candidate,
                buffers,
                num_tokens,
                5,
            )
            reference_probe = cuda.measure_ms(reference, buffers, num_tokens, 5)
            iterations = max(
                5, min(2000, math.ceil(20.0 / max(candidate_probe, reference_probe, 1e-6)))
            )
            candidate_times, reference_times = benchmark_batches(
                cuda, candidate, reference, buffers, num_tokens, iterations
            )
            candidate_median = statistics.median(candidate_times)
            reference_median = statistics.median(reference_times)
            ratio = reference_median / candidate_median
            ratios.append(ratio)
            results.append(
                {
                    "num_tokens": num_tokens,
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
            for pointer in reversed(buffers):
                cuda.free(pointer)
    return results, math.exp(sum(math.log(value) for value in ratios) / len(ratios))


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


def validate_submission_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label} PTX: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} PTX is empty")
    if path.stat().st_size > MAX_PTX_BYTES:
        raise ValueError(f"{label} PTX exceeds {MAX_PTX_BYTES} bytes")
    source = path.read_text()
    if ".entry" not in source or ENTRY.decode() not in source:
        raise ValueError(f"{label} PTX is missing entry {ENTRY.decode()}")


def run(args: argparse.Namespace) -> int:
    candidate_call(
        "Candidate submission validation failed",
        validate_submission_file,
        args.candidate,
        "candidate",
    )
    validate_submission_file(args.reference, "reference")
    cuda = CudaDriver()
    candidate_module = ctypes.c_void_p()
    reference_module = ctypes.c_void_p()
    try:
        reference_module, reference = cuda.load_function(args.reference)
        validate_function(cuda, reference, "reference")
        candidate_module, candidate = candidate_call(
            "Candidate module load failed", cuda.load_function, args.candidate
        )
        candidate_call(
            "Candidate correctness validation failed",
            validate_function,
            cuda,
            candidate,
            "candidate",
        )
        if args.correctness_only:
            rows: list[dict[str, object]] = []
            speedup = 1.0
        else:
            rows, speedup = benchmark(cuda, candidate, reference)
        performance = max(0.0, min(1.0, speedup))
        write_json(
            args.log,
            {
                "status": "passed",
                "validation_level": "feasibility",
                "gpu": cuda.device_name(),
                "entry": ENTRY.decode(),
                "candidate_bytes": args.candidate.stat().st_size,
                "correctness_tokens": CORRECTNESS_TOKENS,
                "performance_rows": rows,
                "speedup_vs_reference": speedup,
            },
        )
        write_json(
            args.reward,
            {
                "reward": performance,
                "correctness": 1.0,
                "performance": performance,
                "speedup_vs_reference": speedup,
            },
        )
        print(
            "PTX_ROPE_FEASIBILITY_OK: "
            f"{len(CORRECTNESS_TOKENS)} correctness cases; "
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
            write_json(
                args.reward,
                {
                    "reward": 0.0,
                    "correctness": 0.0,
                    "performance": 0.0,
                    "speedup_vs_reference": 0.0,
                },
            )
        except Exception:
            pass
        print(f"PTX_ROPE_CANDIDATE_REJECTED: {error}", file=sys.stderr)
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
            write_json(
                args.reward,
                {
                    "reward": 0.0,
                    "correctness": 0.0,
                    "performance": 0.0,
                    "speedup_vs_reference": 0.0,
                },
            )
        except Exception:
            pass
        print(
            f"PTX_ROPE_INFRASTRUCTURE_ERROR: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
