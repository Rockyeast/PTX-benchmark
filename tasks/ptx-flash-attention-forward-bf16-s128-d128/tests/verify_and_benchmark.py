#!/usr/bin/env python3
"""Correctness-gated B200 verifier for causal BF16 attention S=D=128."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import random
import statistics
import struct
from array import array
from pathlib import Path


ENTRY = b"flash_attn_fwd_bf16_s128_d128"
S = 128
D = 128
BLOCK_SIZE = 128
MAX_PTX_BYTES = 1 << 20
GUARD_BYTES = 256
ABS_TOL = 2e-2
REL_TOL = 2e-2
RANDOM_SEED = 0x464C4153
OUTPUT_SENTINEL = 0x7FC0
CORRECTNESS_CASES = [
    (0, "empty"),
    (1, "prefix-average"),
    (1, "causal-future"),
    (3, "random-layout"),
    (8, "stable-logits"),
    (16, "dominant-key"),
]
PERFORMANCE_HEADS = [1, 8, 32]


class CudaError(RuntimeError):
    pass


class CandidateRejected(Exception):
    pass


def candidate_call(operation: str, function, *args):
    try:
        return function(*args)
    except CandidateRejected:
        raise
    except Exception as error:
        raise CandidateRejected(f"{operation}: {error}") from error


class CudaDriver:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._signatures()
        self.device = ctypes.c_int()
        self.context = ctypes.c_void_p()
        self.allocations: set[int] = set()
        self.check(self.lib.cuInit(0), "cuInit")
        self.check(self.lib.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")
        self.check(self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device), "cuCtxCreate")

    def _signatures(self) -> None:
        lib = self.lib
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        lib.cuCtxSynchronize.argtypes = []
        lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        lib.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        lib.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        lib.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
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
        self.check(self.lib.cuDeviceGetName(buffer, len(buffer), self.device), "cuDeviceGetName")
        return buffer.value.decode(errors="replace")

    def load_function(self, path: Path):
        module = ctypes.c_void_p()
        function = ctypes.c_void_p()
        self.check(self.lib.cuModuleLoad(ctypes.byref(module), str(path).encode()), f"cuModuleLoad({path})")
        try:
            self.check(self.lib.cuModuleGetFunction(ctypes.byref(function), module, ENTRY),
                       f"cuModuleGetFunction({ENTRY.decode()})")
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
        self.check(self.lib.cuMemcpyHtoD_v2(destination, host, len(data)), "cuMemcpyHtoD")

    def copy_from_device(self, source: int, size: int) -> bytes:
        host = (ctypes.c_ubyte * size)()
        self.check(self.lib.cuMemcpyDtoH_v2(host, source, size), "cuMemcpyDtoH")
        return bytes(host)

    def launch(self, function, q: int, k: int, v: int, output: int, heads: int) -> None:
        raw = [ctypes.c_uint64(q), ctypes.c_uint64(k), ctypes.c_uint64(v),
               ctypes.c_uint64(output), ctypes.c_uint32(heads)]
        arguments = (ctypes.c_void_p * len(raw))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in raw)
        )
        self.check(self.lib.cuLaunchKernel(function, max(1, heads * S), 1, 1,
                                           BLOCK_SIZE, 1, 1, 0, None, arguments, None),
                   "cuLaunchKernel")

    def synchronize(self) -> None:
        self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def measure_ms(self, function, pointers: tuple[int, int, int, int], heads: int, iterations: int) -> float:
        start, end = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate(start)")
        self.check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate(end)")
        try:
            self.check(self.lib.cuEventRecord(start, None), "cuEventRecord(start)")
            for _ in range(iterations):
                self.launch(function, *pointers, heads)
            self.check(self.lib.cuEventRecord(end, None), "cuEventRecord(end)")
            self.check(self.lib.cuEventSynchronize(end), "cuEventSynchronize")
            elapsed = ctypes.c_float()
            self.check(self.lib.cuEventElapsedTime(ctypes.byref(elapsed), start, end), "cuEventElapsedTime")
            return float(elapsed.value) / iterations
        finally:
            if end.value: self.lib.cuEventDestroy_v2(end)
            if start.value: self.lib.cuEventDestroy_v2(start)

    def close(self) -> None:
        for pointer in tuple(self.allocations):
            self.lib.cuMemFree_v2(pointer)
        self.allocations.clear()
        if self.context.value:
            self.lib.cuCtxDestroy_v2(self.context)


def float_to_bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF):
        return 0x7FC0
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def bf16_to_float(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def bf16(values) -> array:
    return array("H", (float_to_bf16(float(value)) for value in values))


class Truth:
    def __init__(self, path: Path) -> None:
        self.lib = ctypes.CDLL(str(path))
        self.lib.attention_truth.argtypes = [ctypes.POINTER(ctypes.c_uint16)] * 4 + [ctypes.c_uint32]
        self.lib.attention_truth.restype = ctypes.c_int

    def compute(self, q: array, k: array, v: array, heads: int) -> array:
        output = array("H", [0]) * (S * heads * D)
        if heads == 0:
            return output
        q_buf = (ctypes.c_uint16 * len(q)).from_buffer(q)
        k_buf = (ctypes.c_uint16 * len(k)).from_buffer(k)
        v_buf = (ctypes.c_uint16 * len(v)).from_buffer(v)
        o_buf = (ctypes.c_uint16 * len(output)).from_buffer(output)
        result = self.lib.attention_truth(q_buf, k_buf, v_buf, o_buf, heads)
        if result != 0:
            raise RuntimeError(f"CPU truth returned {result}")
        return output


def make_case(heads: int, kind: str, seed: int) -> tuple[array, array, array]:
    count = S * heads * D
    if heads == 0:
        return array("H"), array("H"), array("H")
    rng = random.Random(seed)
    q, k, v = array("H", [0]) * count, array("H", [0]) * count, array("H", [0]) * count
    for token in range(S):
        for head in range(heads):
            base = (token * heads + head) * D
            for feature in range(D):
                if kind == "prefix-average":
                    qv = kv = 0.0
                    vv = (token - 63.5) / 64.0 + ((feature % 9) - 4) / 32.0
                elif kind == "causal-future":
                    qv = kv = 0.0
                    vv = (0.03125 * feature if token < 64 else 24.0 + feature * 0.25)
                elif kind == "stable-logits":
                    sign = -1.0 if (token + feature + head) & 1 else 1.0
                    qv = sign * (6.0 if token % 7 == 0 else 1.25)
                    kv = sign * (5.0 if token == (head * 13 + 17) % S else 0.75)
                    vv = rng.uniform(-2.0, 2.0)
                elif kind == "dominant-key":
                    qv = rng.uniform(-0.75, 0.75)
                    kv = qv * (5.0 if token == (head * 5 + 3) % S else 0.5) + rng.uniform(-0.1, 0.1)
                    vv = rng.uniform(-3.0, 3.0) + head * 0.015625
                else:
                    qv, kv, vv = (rng.uniform(-1.5, 1.5) for _ in range(3))
                q[base + feature] = float_to_bf16(qv)
                k[base + feature] = float_to_bf16(kv)
                v[base + feature] = float_to_bf16(vv)
    return q, k, v


def guarded(payload: bytes, byte: int) -> bytearray:
    guard = bytearray([byte]) * GUARD_BYTES
    return guard + payload + guard


def assert_guards(payload: bytes, byte: int, label: str) -> None:
    guard = bytes([byte]) * GUARD_BYTES
    if payload[:GUARD_BYTES] != guard or payload[-GUARD_BYTES:] != guard:
        raise CandidateRejected(f"{label}: memory guard changed")


def assert_bf16_output(payload: bytes, expected: array, label: str) -> None:
    actual = array("H")
    actual.frombytes(payload)
    if len(actual) != len(expected):
        raise RuntimeError(f"{label}: verifier output length mismatch")
    for index, (got_bits, want_bits) in enumerate(zip(actual, expected)):
        got, want = bf16_to_float(got_bits), bf16_to_float(want_bits)
        if not math.isfinite(got) or not math.isclose(got, want, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            raise CandidateRejected(f"{label}: wrong output at index={index}: expected {want}, got {got}")


def run_case(cuda: CudaDriver, function, truth: Truth, heads: int, kind: str, seed: int, label: str) -> None:
    q, k, v = make_case(heads, kind, seed)
    expected = truth.compute(q, k, v, heads)
    payloads = {"q": q.tobytes(), "k": k.tobytes(), "v": v.tobytes(),
                "output": (array("H", [OUTPUT_SENTINEL]) * len(expected)).tobytes()}
    hosts = {name: guarded(data, byte) for (name, data), byte in zip(payloads.items(), (0xA1, 0xB2, 0xC3, 0xD4))}
    devices = {name: cuda.alloc(len(data)) for name, data in hosts.items()}
    after = {}
    try:
        for name, data in hosts.items(): cuda.copy_to_device(devices[name], data)
        candidate_call(f"{label} launch heads={heads} kind={kind}", cuda.launch, function,
                       devices["q"] + GUARD_BYTES, devices["k"] + GUARD_BYTES,
                       devices["v"] + GUARD_BYTES, devices["output"] + GUARD_BYTES, heads)
        candidate_call(f"{label} sync heads={heads} kind={kind}", cuda.synchronize)
        for name, data in hosts.items():
            after[name] = candidate_call(f"{label} copy {name}", cuda.copy_from_device, devices[name], len(data))
    finally:
        for pointer in reversed(tuple(devices.values())):
            try: cuda.free(pointer)
            except Exception: pass
    for name in ("q", "k", "v"):
        if after.get(name) != bytes(hosts[name]):
            raise CandidateRejected(f"{label}: {name} modified for heads={heads} kind={kind}")
    assert_guards(after["output"], 0xD4, f"{label} output heads={heads} kind={kind}")
    assert_bf16_output(after["output"][GUARD_BYTES:GUARD_BYTES + len(expected) * 2], expected,
                       f"{label} heads={heads} kind={kind}")


def validate_function(cuda: CudaDriver, function, truth: Truth, label: str) -> None:
    for index, (heads, kind) in enumerate(CORRECTNESS_CASES):
        run_case(cuda, function, truth, heads, kind, RANDOM_SEED + index * 0x101, label)


def make_benchmark_case(heads: int) -> tuple[array, array, array, array]:
    q = array("H", [0]) * (S * heads * D)
    k = array("H", [0]) * len(q)
    v = array("H", [0]) * len(q)
    expected = array("H", [0]) * len(q)
    running = [[0.0] * D for _ in range(heads)]
    for token in range(S):
        for head in range(heads):
            base = (token * heads + head) * D
            for d in range(D):
                value = ((token * 17 + head * 11 + d * 5) % 97 - 48) / 32.0
                v[base + d] = float_to_bf16(value)
                running[head][d] += bf16_to_float(v[base + d])
                expected[base + d] = float_to_bf16(running[head][d] / (token + 1))
    return q, k, v, expected


def validate_benchmark_case(cuda: CudaDriver, function, pointers, heads: int, expected: array, label: str) -> None:
    cuda.launch(function, *pointers, heads)
    cuda.synchronize()
    output = cuda.copy_from_device(pointers[3], len(expected) * 2)
    assert_bf16_output(output, expected, f"{label} performance heads={heads}")


def dispersion(values: list[float]) -> float:
    median = statistics.median(values)
    return math.inf if median == 0 else (max(values) - min(values)) / median


def benchmark_shape(cuda: CudaDriver, candidate, reference, heads: int) -> dict[str, object]:
    q, k, v, expected = make_benchmark_case(heads)
    buffers = [q.tobytes(), k.tobytes(), v.tobytes(), (array("H", [OUTPUT_SENTINEL]) * len(expected)).tobytes()]
    pointers = tuple(cuda.alloc(len(data)) for data in buffers)
    try:
        for pointer, data in zip(pointers, buffers): cuda.copy_to_device(pointer, data)
        candidate_call("Candidate performance validation", validate_benchmark_case,
                       cuda, candidate, pointers, heads, expected, "candidate")
        try:
            validate_benchmark_case(cuda, reference, pointers, heads, expected, "reference")
        except CandidateRejected as error:
            raise RuntimeError(f"Reference performance validation failed: {error}") from error
        for _ in range(5):
            cuda.launch(candidate, *pointers, heads)
            cuda.launch(reference, *pointers, heads)
        cuda.synchronize()
        candidate_probe = candidate_call("Candidate timing probe", cuda.measure_ms, candidate, pointers, heads, 3)
        reference_probe = cuda.measure_ms(reference, pointers, heads, 3)
        iterations = max(3, min(1000, math.ceil(50.0 / max(candidate_probe, reference_probe, 1e-6))))
        timings = {"candidate": [], "reference": []}
        for batch in range(5):
            order = (("candidate", candidate), ("reference", reference)) if batch % 2 == 0 else (("reference", reference), ("candidate", candidate))
            for name, function in order:
                call = candidate_call if name == "candidate" else lambda _op, fn, *a: fn(*a)
                timings[name].append(call("Candidate timing", cuda.measure_ms, function, pointers, heads, iterations))
        candidate_median = statistics.median(timings["candidate"])
        reference_median = statistics.median(timings["reference"])
        ratio = reference_median / candidate_median
        logical_flops = 4 * heads * (S * (S + 1) // 2) * D
        return {"num_heads": heads, "iterations_per_batch": iterations,
                "candidate_ms_per_launch": timings["candidate"], "reference_ms_per_launch": timings["reference"],
                "candidate_median_ms": candidate_median, "reference_median_ms": reference_median,
                "speedup_vs_reference": ratio,
                "candidate_logical_tflops": logical_flops / (candidate_median * 1e9),
                "reference_logical_tflops": logical_flops / (reference_median * 1e9),
                "candidate_dispersion": dispersion(timings["candidate"]),
                "reference_dispersion": dispersion(timings["reference"])}
    finally:
        for pointer in reversed(pointers):
            try: cuda.free(pointer)
            except Exception: pass


def validate_ptx(path: Path, candidate: bool) -> None:
    role = "Candidate" if candidate else "fixed Reference"
    if not path.is_file():
        error = CandidateRejected if candidate else FileNotFoundError
        raise error(f"missing {role} PTX: {path}")
    size = path.stat().st_size
    if size == 0 or size > MAX_PTX_BYTES:
        error = CandidateRejected if candidate else RuntimeError
        raise error(f"{role} PTX has invalid size")
    try: source = path.read_text()
    except UnicodeDecodeError as exc:
        error = CandidateRejected if candidate else RuntimeError
        raise error(f"{role} PTX is not UTF-8") from exc
    if ".entry" not in source or ENTRY.decode() not in source:
        error = CandidateRejected if candidate else RuntimeError
        raise error(f"{role} PTX is missing entry {ENTRY.decode()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--correctness-only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args):
    validate_ptx(args.reference, False)
    validate_ptx(args.candidate, True)
    truth = Truth(args.truth)
    cuda = CudaDriver()
    candidate_module, reference_module = ctypes.c_void_p(), ctypes.c_void_p()
    try:
        reference_module, reference = cuda.load_function(args.reference)
        try: validate_function(cuda, reference, truth, "reference")
        except CandidateRejected as error: raise RuntimeError(f"fixed Reference correctness failed: {error}") from error
        candidate_module, candidate = candidate_call("Candidate module load", cuda.load_function, args.candidate)
        candidate_call("Candidate correctness", validate_function, cuda, candidate, truth, "candidate")
        metrics = {"status": "correctness_passed", "gpu": cuda.device_name(), "entry": ENTRY.decode(),
                   "correctness_cases": CORRECTNESS_CASES, "atol": ABS_TOL, "rtol": REL_TOL}
        if args.correctness_only:
            reward = {"reward": 0.0, "correctness": 1.0, "performance": 0.0, "speedup_vs_reference": 0.0}
        else:
            timing = [benchmark_shape(cuda, candidate, reference, heads) for heads in PERFORMANCE_HEADS]
            ratios = [float(row["speedup_vs_reference"]) for row in timing]
            raw = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            performance = min(1.0, max(0.0, raw))
            metrics.update(status="passed", timing=timing, raw_speedup_vs_reference=raw)
            reward = {"reward": performance, "correctness": 1.0,
                      "performance": performance, "speedup_vs_reference": raw}
        return metrics, reward
    finally:
        if candidate_module.value: cuda.lib.cuModuleUnload(candidate_module)
        if reference_module.value: cuda.lib.cuModuleUnload(reference_module)
        cuda.close()


def main() -> int:
    args = parse_args()
    reward = {"reward": 0.0, "correctness": 0.0, "performance": 0.0, "speedup_vs_reference": 0.0}
    metrics = {"status": "failed", "entry": ENTRY.decode()}
    try:
        metrics, reward = run(args)
    except CandidateRejected as error:
        metrics.update(status="candidate_rejected", error_type=type(error).__name__, error=str(error))
        print(f"PTX verification failed: {type(error).__name__}: {error}")
        write_json(args.log, metrics); write_json(args.reward, reward)
        return 1
    except Exception as error:
        metrics.update(status="infrastructure_error", error_type=type(error).__name__, error=str(error))
        print(f"PTX verifier infrastructure failure: {type(error).__name__}: {error}")
        write_json(args.log, metrics); write_json(args.reward, reward)
        return 2
    write_json(args.log, metrics); write_json(args.reward, reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
