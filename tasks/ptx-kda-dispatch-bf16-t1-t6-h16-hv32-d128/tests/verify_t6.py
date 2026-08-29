#!/usr/bin/env python3
"""Correctness-gated verifier for fixed T6 speculative KDA decode on B200."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
from pathlib import Path


ENTRY = b"flash_kda_decode_sm100a_t6_split1"
T, N, H, HV, D = 6, 8, 16, 32, 128
SLOTS = N * T + 6
QK_ELEMENTS = N * T * H * D
VG_ELEMENTS = N * T * HV * D
BETA_ELEMENTS = N * T * HV
STATE_ELEMENTS = SLOTS * HV * D * D
OUTPUT_ELEMENTS = N * T * HV * D
INDEX_ELEMENTS = N * T
ACCEPTED_ELEMENTS = N
GRID_X, GRID_Y, BLOCK_X = 32, N, 256
DYNAMIC_SHARED_BYTES = 50560
GUARD_BYTES = 256
MAX_PTX_BYTES = 4 << 20
ABS_TOL, REL_TOL = 1.0e-2, 1.0e-2
SCALE = 1.0 / math.sqrt(D)
OUTPUT_SENTINEL = 0x7FC0
CORRECTNESS_CASES = [
    (0, "random-mixed-accepted"),
    (1, "permuted-checkpoints"),
    (2, "zero-state"),
    (3, "zero-beta"),
    (4, "structured-head-map"),
    (5, "decay-extremes"),
    (6, "state-rounding"),
    (7, "accepted-boundaries"),
]
PERFORMANCE_CASES = [
    (0, "random-mixed-accepted"),
    (1, "permuted-checkpoints"),
    (4, "structured-head-map"),
    (7, "accepted-boundaries"),
]


class CudaError(RuntimeError):
    pass


class CandidateRejected(Exception):
    pass


def candidate_call(operation, function, *args):
    try:
        return function(*args)
    except CandidateRejected:
        raise
    except Exception as error:
        raise CandidateRejected(f"{operation}: {error}") from error


class CudaDriver:
    def __init__(self):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._signatures()
        self.device = ctypes.c_int()
        self.context = ctypes.c_void_p()
        self.allocations: set[int] = set()
        self.check(self.lib.cuInit(0), "cuInit")
        self.check(self.lib.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")
        self.check(
            self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device),
            "cuCtxCreate",
        )

    def _signatures(self):
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
        lib.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        lib.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
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
            return name.value.decode()
        return "UNKNOWN"

    def check(self, result: int, operation: str):
        if result:
            raise CudaError(f"{operation} failed: {self.error_name(result)} ({result})")

    def device_name(self) -> str:
        buffer = ctypes.create_string_buffer(256)
        self.check(self.lib.cuDeviceGetName(buffer, len(buffer), self.device), "cuDeviceGetName")
        return buffer.value.decode()

    def load_function(self, path: Path):
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cuModuleLoad(ctypes.byref(module), str(path).encode()), f"cuModuleLoad({path})")
        try:
            self.check(
                self.lib.cuModuleGetFunction(ctypes.byref(function), module, ENTRY),
                f"cuModuleGetFunction({ENTRY.decode()})",
            )
            self.check(
                self.lib.cuFuncSetAttribute(function, 8, DYNAMIC_SHARED_BYTES),
                "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)",
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

    def free(self, pointer: int):
        if pointer in self.allocations:
            self.check(self.lib.cuMemFree_v2(pointer), "cuMemFree")
            self.allocations.remove(pointer)

    def copy_to_device(self, pointer: int, data: bytes):
        host = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        self.check(self.lib.cuMemcpyHtoD_v2(pointer, host, len(data)), "cuMemcpyHtoD")

    def copy_from_device(self, pointer: int, size: int) -> bytes:
        host = (ctypes.c_ubyte * size)()
        self.check(self.lib.cuMemcpyDtoH_v2(host, pointer, size), "cuMemcpyDtoH")
        return bytes(host)

    def launch(self, function, pointers: tuple[int, ...]):
        if len(pointers) != 9:
            raise RuntimeError("internal ABI error: expected nine pointers")
        raw = [ctypes.c_uint64(pointer) for pointer in pointers]
        raw.append(ctypes.c_float(SCALE))
        arguments = (ctypes.c_void_p * len(raw))(
            *(ctypes.cast(ctypes.byref(item), ctypes.c_void_p) for item in raw)
        )
        self.check(
            self.lib.cuLaunchKernel(
                function, GRID_X, GRID_Y, 1, BLOCK_X, 1, 1,
                DYNAMIC_SHARED_BYTES, None, arguments, None
            ),
            "cuLaunchKernel",
        )

    def synchronize(self):
        self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def measure_ms(self, function, pointers, state_pointer, initial_state, iterations):
        self.copy_to_device(state_pointer, initial_state)
        start, end = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
        self.check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate")
        try:
            self.check(self.lib.cuEventRecord(start, None), "cuEventRecord")
            for _ in range(iterations):
                self.launch(function, pointers)
            self.check(self.lib.cuEventRecord(end, None), "cuEventRecord")
            self.check(self.lib.cuEventSynchronize(end), "cuEventSynchronize")
            elapsed = ctypes.c_float()
            self.check(self.lib.cuEventElapsedTime(ctypes.byref(elapsed), start, end), "cuEventElapsedTime")
            return float(elapsed.value) / iterations
        finally:
            if end.value:
                self.lib.cuEventDestroy_v2(end)
            if start.value:
                self.lib.cuEventDestroy_v2(start)

    def close(self):
        for pointer in tuple(self.allocations):
            self.lib.cuMemFree_v2(pointer)
        if self.context.value:
            self.lib.cuCtxDestroy_v2(self.context)


class Truth:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        p16 = ctypes.POINTER(ctypes.c_uint16)
        p32 = ctypes.POINTER(ctypes.c_int32)
        self.lib.kda_t6_make_case.argtypes = [
            p16, p16, p16, p16, p16, p16, p32, p32, ctypes.c_uint32
        ]
        self.lib.kda_t6_make_case.restype = ctypes.c_int
        self.lib.kda_t6_truth.argtypes = [
            p16, p16, p16, p16, p16, p16, p16, p32, p32, ctypes.c_float
        ]
        self.lib.kda_t6_truth.restype = ctypes.c_int
        self.lib.kda_t6_compare.argtypes = [
            p16, p16, ctypes.c_uint64, ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.kda_t6_compare.restype = ctypes.c_int

    def make_case(self, mode: int):
        q = (ctypes.c_uint16 * QK_ELEMENTS)()
        k = (ctypes.c_uint16 * QK_ELEMENTS)()
        v = (ctypes.c_uint16 * VG_ELEMENTS)()
        log_decay = (ctypes.c_uint16 * VG_ELEMENTS)()
        beta = (ctypes.c_uint16 * BETA_ELEMENTS)()
        state = (ctypes.c_uint16 * STATE_ELEMENTS)()
        state_indices = (ctypes.c_int32 * INDEX_ELEMENTS)()
        accepted = (ctypes.c_int32 * ACCEPTED_ELEMENTS)()
        result = self.lib.kda_t6_make_case(
            q, k, v, log_decay, beta, state, state_indices, accepted, mode
        )
        if result:
            raise RuntimeError(f"CPU case generator returned {result} for mode {mode}")
        return q, k, v, log_decay, beta, state, state_indices, accepted

    def compute(self, q, k, v, log_decay, beta, initial_state, state_indices, accepted):
        expected_state = (ctypes.c_uint16 * STATE_ELEMENTS)()
        ctypes.memmove(expected_state, initial_state, ctypes.sizeof(expected_state))
        expected_output = (ctypes.c_uint16 * OUTPUT_ELEMENTS)()
        result = self.lib.kda_t6_truth(
            q, k, v, log_decay, beta, expected_state, expected_output,
            state_indices, accepted, ctypes.c_float(SCALE)
        )
        if result:
            raise RuntimeError(f"CPU truth returned {result}")
        return expected_state, expected_output

    def compare(self, got: bytes, expected, count: int, label: str):
        got_array = (ctypes.c_uint16 * count).from_buffer_copy(got)
        bad_index, bad_got, bad_expected = ctypes.c_uint64(), ctypes.c_float(), ctypes.c_float()
        max_abs, max_rel = ctypes.c_float(), ctypes.c_float()
        result = self.lib.kda_t6_compare(
            got_array, expected, count, ctypes.c_float(ABS_TOL), ctypes.c_float(REL_TOL),
            ctypes.byref(bad_index), ctypes.byref(bad_got), ctypes.byref(bad_expected),
            ctypes.byref(max_abs), ctypes.byref(max_rel),
        )
        metrics = {"max_abs": float(max_abs.value), "max_rel": float(max_rel.value)}
        if result == 1:
            raise CandidateRejected(
                f"{label}: mismatch at index {bad_index.value}: expected {bad_expected.value}, "
                f"got {bad_got.value}; max_abs_so_far={max_abs.value}"
            )
        if result:
            raise RuntimeError(f"CPU comparison helper returned {result}")
        return metrics


def array_bytes(value) -> bytes:
    return ctypes.string_at(ctypes.addressof(value), ctypes.sizeof(value))


def guarded(payload: bytes, guard: int) -> bytes:
    marker = bytes([guard]) * GUARD_BYTES
    return marker + payload + marker


def assert_guards(payload: bytes, guard: int, label: str):
    marker = bytes([guard]) * GUARD_BYTES
    if payload[:GUARD_BYTES] != marker or payload[-GUARD_BYTES:] != marker:
        raise CandidateRejected(f"{label}: memory guard changed")


def run_case(cuda: CudaDriver, function, truth: Truth, mode: int, label: str):
    q, k, v, log_decay, beta, initial_state, state_indices, accepted = truth.make_case(mode)
    expected_state, expected_output = truth.compute(
        q, k, v, log_decay, beta, initial_state, state_indices, accepted
    )
    payloads = {
        "q": array_bytes(q), "k": array_bytes(k), "v": array_bytes(v),
        "log_decay": array_bytes(log_decay), "beta": array_bytes(beta),
        "state": array_bytes(initial_state),
        "output": bytes((ctypes.c_uint16 * OUTPUT_ELEMENTS)(
            *([OUTPUT_SENTINEL] * OUTPUT_ELEMENTS)
        )),
        "state_indices": array_bytes(state_indices),
        "accepted": array_bytes(accepted),
    }
    guard_values = {
        "q": 0xA1, "k": 0xB2, "v": 0xC3, "log_decay": 0xD4,
        "beta": 0xE5, "state": 0x96, "output": 0x78,
        "state_indices": 0x69, "accepted": 0x5A,
    }
    hosts = {name: guarded(data, guard_values[name]) for name, data in payloads.items()}
    devices = {name: cuda.alloc(len(data)) for name, data in hosts.items()}
    after: dict[str, bytes] = {}
    try:
        for name, data in hosts.items():
            cuda.copy_to_device(devices[name], data)
        pointers = tuple(
            devices[name] + GUARD_BYTES
            for name in (
                "q", "k", "v", "log_decay", "beta", "state", "output",
                "state_indices", "accepted",
            )
        )
        candidate_call(f"{label} launch", cuda.launch, function, pointers)
        candidate_call(f"{label} synchronize", cuda.synchronize)
        for name, data in hosts.items():
            after[name] = candidate_call(f"{label} copy {name}", cuda.copy_from_device, devices[name], len(data))
    finally:
        for pointer in reversed(tuple(devices.values())):
            try:
                cuda.free(pointer)
            except Exception:
                pass
    for name in ("q", "k", "v", "log_decay", "beta", "state_indices", "accepted"):
        if after[name] != hosts[name]:
            raise CandidateRejected(f"{label}: read-only input {name} was modified")
    assert_guards(after["state"], guard_values["state"], f"{label} state")
    assert_guards(after["output"], guard_values["output"], f"{label} output")
    return {
        "mode": mode,
        "name": label,
        "state": truth.compare(after["state"][GUARD_BYTES:-GUARD_BYTES], expected_state, STATE_ELEMENTS, f"{label} state"),
        "output": truth.compare(
            after["output"][GUARD_BYTES:-GUARD_BYTES], expected_output,
            OUTPUT_ELEMENTS, f"{label} output"
        ),
    }


def validate_function(cuda: CudaDriver, function, truth: Truth, role: str):
    return [run_case(cuda, function, truth, mode, f"{role} {name}") for mode, name in CORRECTNESS_CASES]


def dispersion(values: list[float]) -> float:
    median = statistics.median(values)
    return math.inf if median == 0.0 else (max(values) - min(values)) / median


def benchmark_case(cuda, candidate, reference, truth, mode, name):
    q, k, v, log_decay, beta, initial_state, state_indices, accepted = truth.make_case(mode)
    expected_state, expected_output = truth.compute(
        q, k, v, log_decay, beta, initial_state, state_indices, accepted
    )
    tensor_inputs = [
        array_bytes(q), array_bytes(k), array_bytes(v), array_bytes(log_decay),
        array_bytes(beta),
    ]
    metadata_inputs = [array_bytes(state_indices), array_bytes(accepted)]
    initial_state_bytes, output_bytes = array_bytes(initial_state), bytes(ctypes.sizeof(expected_output))
    tensor_pointers = tuple(cuda.alloc(len(data)) for data in tensor_inputs)
    metadata_pointers = tuple(cuda.alloc(len(data)) for data in metadata_inputs)
    candidate_state, candidate_output = cuda.alloc(len(initial_state_bytes)), cuda.alloc(len(output_bytes))
    reference_state, reference_output = cuda.alloc(len(initial_state_bytes)), cuda.alloc(len(output_bytes))
    candidate_pointers = tensor_pointers + (candidate_state, candidate_output) + metadata_pointers
    reference_pointers = tensor_pointers + (reference_state, reference_output) + metadata_pointers
    all_pointers = (
        tensor_pointers + metadata_pointers
        + (candidate_state, candidate_output, reference_state, reference_output)
    )
    try:
        for pointer, data in zip(tensor_pointers, tensor_inputs):
            cuda.copy_to_device(pointer, data)
        for pointer, data in zip(metadata_pointers, metadata_inputs):
            cuda.copy_to_device(pointer, data)
        validation = {}
        for role, function, pointers, state_pointer, output_pointer in (
            ("candidate", candidate, candidate_pointers, candidate_state, candidate_output),
            ("reference", reference, reference_pointers, reference_state, reference_output),
        ):
            cuda.copy_to_device(state_pointer, initial_state_bytes)
            cuda.copy_to_device(output_pointer, output_bytes)
            try:
                cuda.launch(function, pointers)
                cuda.synchronize()
                validation[role] = {
                    "state": truth.compare(cuda.copy_from_device(state_pointer, len(initial_state_bytes)), expected_state, STATE_ELEMENTS, f"{role} performance {name} state"),
                    "output": truth.compare(
                        cuda.copy_from_device(output_pointer, len(output_bytes)),
                        expected_output, OUTPUT_ELEMENTS,
                        f"{role} performance {name} output"
                    ),
                }
            except CandidateRejected:
                if role == "reference":
                    raise RuntimeError("fixed Reference performance correctness failed")
                raise
        for function, pointers, state_pointer in (
            (candidate, candidate_pointers, candidate_state),
            (reference, reference_pointers, reference_state),
        ):
            cuda.measure_ms(function, pointers, state_pointer, initial_state_bytes, 3)
        candidate_probe = candidate_call(
            "Candidate timing probe", cuda.measure_ms, candidate, candidate_pointers,
            candidate_state, initial_state_bytes, 3,
        )
        reference_probe = cuda.measure_ms(reference, reference_pointers, reference_state, initial_state_bytes, 3)
        iterations = max(3, min(2000, math.ceil(30.0 / max(candidate_probe, reference_probe, 1.0e-6))))
        times = {"candidate": [], "reference": []}
        for batch in range(5):
            order = (
                (("candidate", candidate, candidate_pointers, candidate_state), ("reference", reference, reference_pointers, reference_state))
                if batch % 2 == 0
                else (("reference", reference, reference_pointers, reference_state), ("candidate", candidate, candidate_pointers, candidate_state))
            )
            for role, function, pointers, state_pointer in order:
                if role == "candidate":
                    elapsed = candidate_call("Candidate timing", cuda.measure_ms, function, pointers, state_pointer, initial_state_bytes, iterations)
                else:
                    elapsed = cuda.measure_ms(function, pointers, state_pointer, initial_state_bytes, iterations)
                times[role].append(elapsed)
        candidate_median, reference_median = statistics.median(times["candidate"]), statistics.median(times["reference"])
        ratio = reference_median / candidate_median
        logical_flops = N * T * HV * D * D * 6
        return {
            "mode": mode, "name": name, "iterations_per_batch": iterations,
            "candidate_ms_per_launch": times["candidate"], "reference_ms_per_launch": times["reference"],
            "candidate_median_ms": candidate_median, "reference_median_ms": reference_median,
            "speedup_vs_reference": ratio,
            "candidate_logical_tflops": logical_flops / (candidate_median * 1.0e9),
            "reference_logical_tflops": logical_flops / (reference_median * 1.0e9),
            "candidate_dispersion": dispersion(times["candidate"]),
            "reference_dispersion": dispersion(times["reference"]),
            "validation": validation,
        }
    finally:
        for pointer in reversed(all_pointers):
            try:
                cuda.free(pointer)
            except Exception:
                pass


def validate_ptx(path: Path, candidate: bool):
    role = "Candidate" if candidate else "fixed Reference"
    error_type = CandidateRejected if candidate else RuntimeError
    if not path.is_file():
        raise error_type(f"missing {role} PTX: {path}")
    size = path.stat().st_size
    if size == 0 or size > MAX_PTX_BYTES:
        raise error_type(f"{role} PTX has invalid size {size}")
    try:
        source = path.read_text()
    except UnicodeDecodeError as error:
        raise error_type(f"{role} PTX is not UTF-8") from error
    if ".entry" not in source or ENTRY.decode() not in source:
        raise error_type(f"{role} PTX missing entry {ENTRY.decode()}")


def validate_image(path: Path, candidate: bool):
    role = "Candidate" if candidate else "fixed Reference"
    error_type = CandidateRejected if candidate else RuntimeError
    if not path.is_file() or path.stat().st_size == 0:
        raise error_type(f"missing assembled {role} image: {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    for name in ("candidate", "reference", "truth", "log", "reward"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--candidate-image", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--correctness-only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args):
    validate_ptx(args.reference, False)
    validate_ptx(args.candidate, True)
    validate_image(args.reference_image, False)
    validate_image(args.candidate_image, True)
    truth, cuda = Truth(args.truth), CudaDriver()
    candidate_module, reference_module = ctypes.c_void_p(), ctypes.c_void_p()
    try:
        reference_module, reference = cuda.load_function(args.reference_image)
        try:
            reference_correctness = validate_function(cuda, reference, truth, "reference")
        except CandidateRejected as error:
            raise RuntimeError(f"fixed Reference correctness failed: {error}") from error
        candidate_module, candidate = candidate_call(
            "Candidate module load", cuda.load_function, args.candidate_image
        )
        candidate_correctness = candidate_call("Candidate correctness", validate_function, cuda, candidate, truth, "candidate")
        metrics = {
            "status": "correctness_passed", "gpu": cuda.device_name(), "entry": ENTRY.decode(),
            "shape": {"T": T, "N": N, "H": H, "HV": HV, "D": D},
            "launch": {
                "grid": [GRID_X, GRID_Y, 1],
                "block": [BLOCK_X, 1, 1],
                "dynamic_shared_bytes": DYNAMIC_SHARED_BYTES,
            },
            "atol": ABS_TOL, "rtol": REL_TOL,
            "reference_correctness": reference_correctness,
            "candidate_correctness": candidate_correctness,
        }
        if args.correctness_only:
            reward = {"reward": 0.0, "correctness": 1.0, "performance": 0.0, "speedup_vs_reference": 0.0}
        else:
            timing = [benchmark_case(cuda, candidate, reference, truth, mode, name) for mode, name in PERFORMANCE_CASES]
            ratios = [item["speedup_vs_reference"] for item in timing]
            raw_speedup = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
            performance = min(1.0, max(0.0, raw_speedup))
            metrics.update(status="passed", timing=timing, raw_speedup_vs_reference=raw_speedup)
            reward = {"reward": performance, "correctness": 1.0, "performance": performance, "speedup_vs_reference": raw_speedup}
        return metrics, reward
    finally:
        if candidate_module.value:
            cuda.lib.cuModuleUnload(candidate_module)
        if reference_module.value:
            cuda.lib.cuModuleUnload(reference_module)
        cuda.close()


def main():
    args = parse_args()
    reward = {"reward": 0.0, "correctness": 0.0, "performance": 0.0, "speedup_vs_reference": 0.0}
    metrics = {"status": "failed", "entry": ENTRY.decode()}
    try:
        metrics, reward = run(args)
    except CandidateRejected as error:
        metrics.update(status="candidate_rejected", error_type=type(error).__name__, error=str(error))
        print(f"PTX verification failed: {error}")
        write_json(args.log, metrics)
        write_json(args.reward, reward)
        return 1
    except Exception as error:
        metrics.update(status="infrastructure_error", error_type=type(error).__name__, error=str(error))
        print(f"PTX verifier infrastructure failure: {error}")
        write_json(args.log, metrics)
        write_json(args.reward, reward)
        return 2
    write_json(args.log, metrics)
    write_json(args.reward, reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
