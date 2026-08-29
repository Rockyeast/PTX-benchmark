#!/usr/bin/env python3
"""Correctness-gated B200 verifier for BF16 attention backward S=D=128."""

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


ENTRY = b"flash_attn_bwd_bf16_s128_d128"
S = 128
D = 128
BLOCK_SIZE = 128
MAX_PTX_BYTES = 1 << 20
GUARD_BYTES = 256
ABS_TOL = 2e-2
REL_TOL = 3e-2
RANDOM_SEED = 0x42574431
OUTPUT_SENTINEL = 0x7FC00000
CORRECTNESS_CASES = ["zeros", "random", "structured", "dominant-key"]


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

    def launch(self, function, pointers: tuple[int, ...]) -> None:
        if len(pointers) != 7:
            raise RuntimeError("internal ABI error: expected seven pointers")
        raw = [ctypes.c_uint64(pointer) for pointer in pointers]
        arguments = (ctypes.c_void_p * len(raw))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in raw)
        )
        self.check(self.lib.cuLaunchKernel(function, 1, 1, 1,
                                           BLOCK_SIZE, 1, 1, 0, None, arguments, None),
                   "cuLaunchKernel")

    def synchronize(self) -> None:
        self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def measure_ms(self, function, pointers: tuple[int, ...], iterations: int) -> float:
        start, end = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate(start)")
        self.check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate(end)")
        try:
            self.check(self.lib.cuEventRecord(start, None), "cuEventRecord(start)")
            for _ in range(iterations):
                self.launch(function, pointers)
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
        p16 = ctypes.POINTER(ctypes.c_uint16)
        p32 = ctypes.POINTER(ctypes.c_float)
        self.lib.attention_bwd_truth.argtypes = [p16, p16, p16, p16, p32, p32, p32]
        self.lib.attention_bwd_truth.restype = ctypes.c_int

    def compute(self, q: array, k: array, v: array, grad_output: array) -> tuple[array, array, array]:
        outputs = tuple(array("f", [0.0]) * (S * D) for _ in range(3))
        inputs = [
            (ctypes.c_uint16 * len(values)).from_buffer(values)
            for values in (q, k, v, grad_output)
        ]
        output_buffers = [
            (ctypes.c_float * len(values)).from_buffer(values) for values in outputs
        ]
        result = self.lib.attention_bwd_truth(*inputs, *output_buffers)
        if result != 0:
            raise RuntimeError(f"CPU truth returned {result}")
        return outputs


def make_case(kind: str, seed: int) -> tuple[array, array, array, array]:
    count = S * D
    rng = random.Random(seed)
    q = array("H", [0]) * count
    k = array("H", [0]) * count
    v = array("H", [0]) * count
    grad_output = array("H", [0]) * count
    for token in range(S):
        base = token * D
        for feature in range(D):
            if kind == "zeros":
                qv = kv = 0.0
                vv = ((token * 3 + feature * 5) % 31 - 15) / 32.0
                dov = ((token * 7 + feature * 2) % 29 - 14) / 32.0
            elif kind == "structured":
                qv = ((token + feature * 3) % 17 - 8) / 16.0
                kv = ((token * 5 - feature) % 19 - 9) / 16.0
                vv = ((token * 2 + feature * 7) % 23 - 11) / 16.0
                dov = ((token * 11 + feature) % 27 - 13) / 16.0
            elif kind == "dominant-key":
                # Every query row is aligned with key row 7. The old form
                # compared the current row index with a row-dependent target,
                # which did not actually construct a dominant key.
                pattern = ((feature * 5) % 17 - 8) / 16.0
                qv = pattern + ((token % 3) - 1) / 64.0
                kv = 3.0 * pattern if token == 7 else rng.uniform(-0.05, 0.05)
                vv = rng.uniform(-0.75, 0.75)
                dov = rng.uniform(-0.75, 0.75)
            else:
                qv, kv, vv, dov = (rng.uniform(-0.75, 0.75) for _ in range(4))
            q[base + feature] = float_to_bf16(qv)
            k[base + feature] = float_to_bf16(kv)
            v[base + feature] = float_to_bf16(vv)
            grad_output[base + feature] = float_to_bf16(dov)
    return q, k, v, grad_output


def guarded(payload: bytes, byte: int) -> bytearray:
    guard = bytearray([byte]) * GUARD_BYTES
    return guard + payload + guard


def assert_guards(payload: bytes, byte: int, label: str) -> None:
    guard = bytes([byte]) * GUARD_BYTES
    if payload[:GUARD_BYTES] != guard or payload[-GUARD_BYTES:] != guard:
        raise CandidateRejected(f"{label}: memory guard changed")


def assert_float_output(payload: bytes, expected: array, label: str) -> None:
    actual = array("f")
    actual.frombytes(payload)
    if len(actual) != len(expected):
        raise RuntimeError(f"{label}: verifier output length mismatch")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if not math.isfinite(got) or not math.isclose(got, want, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            raise CandidateRejected(f"{label}: wrong output at index={index}: expected {want}, got {got}")


def run_case(cuda: CudaDriver, function, truth: Truth, kind: str, seed: int, label: str) -> None:
    q, k, v, grad_output = make_case(kind, seed)
    expected = truth.compute(q, k, v, grad_output)
    sentinel = (array("I", [OUTPUT_SENTINEL]) * (S * D)).tobytes()
    payloads = {
        "q": q.tobytes(), "k": k.tobytes(), "v": v.tobytes(),
        "grad_output": grad_output.tobytes(),
        "grad_q": sentinel, "grad_k": sentinel, "grad_v": sentinel,
    }
    guard_bytes = (0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x97)
    hosts = {
        name: guarded(data, byte)
        for (name, data), byte in zip(payloads.items(), guard_bytes)
    }
    devices = {name: cuda.alloc(len(data)) for name, data in hosts.items()}
    after = {}
    try:
        for name, data in hosts.items(): cuda.copy_to_device(devices[name], data)
        pointers = tuple(devices[name] + GUARD_BYTES for name in payloads)
        candidate_call(f"{label} launch kind={kind}", cuda.launch, function, pointers)
        candidate_call(f"{label} sync kind={kind}", cuda.synchronize)
        for name, data in hosts.items():
            after[name] = candidate_call(f"{label} copy {name}", cuda.copy_from_device, devices[name], len(data))
    finally:
        for pointer in reversed(tuple(devices.values())):
            try: cuda.free(pointer)
            except Exception: pass
    for name in ("q", "k", "v", "grad_output"):
        if after.get(name) != bytes(hosts[name]):
            raise CandidateRejected(f"{label}: {name} modified for kind={kind}")
    for index, name in enumerate(("grad_q", "grad_k", "grad_v")):
        byte = guard_bytes[4 + index]
        assert_guards(after[name], byte, f"{label} {name} kind={kind}")
        payload = after[name][GUARD_BYTES:GUARD_BYTES + S * D * 4]
        assert_float_output(payload, expected[index], f"{label} {name} kind={kind}")


def validate_function(cuda: CudaDriver, function, truth: Truth, label: str) -> None:
    for index, kind in enumerate(CORRECTNESS_CASES):
        run_case(cuda, function, truth, kind, RANDOM_SEED + index * 0x101, label)


def validate_benchmark_case(cuda: CudaDriver, function, pointers, expected, label: str) -> None:
    cuda.launch(function, pointers)
    cuda.synchronize()
    for index, output_pointer in enumerate(pointers[4:]):
        output = cuda.copy_from_device(output_pointer, S * D * 4)
        assert_float_output(output, expected[index], f"{label} performance output={index}")


def dispersion(values: list[float]) -> float:
    median = statistics.median(values)
    return math.inf if median == 0 else (max(values) - min(values)) / median


def benchmark_shape(cuda: CudaDriver, candidate, reference, truth: Truth) -> dict[str, object]:
    q, k, v, grad_output = make_case("random", RANDOM_SEED + 0x5151)
    expected = truth.compute(q, k, v, grad_output)
    sentinel = (array("I", [OUTPUT_SENTINEL]) * (S * D)).tobytes()
    buffers = [q.tobytes(), k.tobytes(), v.tobytes(), grad_output.tobytes(),
               sentinel, sentinel, sentinel]
    pointers = tuple(cuda.alloc(len(data)) for data in buffers)
    try:
        for pointer, data in zip(pointers, buffers): cuda.copy_to_device(pointer, data)
        candidate_call("Candidate performance validation", validate_benchmark_case,
                       cuda, candidate, pointers, expected, "candidate")
        try:
            validate_benchmark_case(cuda, reference, pointers, expected, "reference")
        except CandidateRejected as error:
            raise RuntimeError(f"Reference performance validation failed: {error}") from error
        for _ in range(5):
            cuda.launch(candidate, pointers)
            cuda.launch(reference, pointers)
        cuda.synchronize()
        candidate_probe = candidate_call("Candidate timing probe", cuda.measure_ms, candidate, pointers, 3)
        reference_probe = cuda.measure_ms(reference, pointers, 3)
        iterations = max(3, min(1000, math.ceil(50.0 / max(candidate_probe, reference_probe, 1e-6))))
        timings = {"candidate": [], "reference": []}
        for batch in range(5):
            order = (("candidate", candidate), ("reference", reference)) if batch % 2 == 0 else (("reference", reference), ("candidate", candidate))
            for name, function in order:
                call = candidate_call if name == "candidate" else lambda _op, fn, *a: fn(*a)
                timings[name].append(call("Candidate timing", cuda.measure_ms, function, pointers, iterations))
        candidate_median = statistics.median(timings["candidate"])
        reference_median = statistics.median(timings["reference"])
        ratio = reference_median / candidate_median
        logical_flops = 10 * S * S * D
        return {"shape": {"S": S, "D": D}, "iterations_per_batch": iterations,
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
            timing = [benchmark_shape(cuda, candidate, reference, truth)]
            raw = float(timing[0]["speedup_vs_reference"])
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
