#!/usr/bin/env python3
"""Hidden correctness verifier for paged sparse GQA attention backward."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import struct
from array import array
from pathlib import Path

ZERO_ENTRY = b"paged_sparse_attn_bwd_zero_bf16_q128_hq8_hkv2_d128_p16"
MAIN_ENTRY = b"paged_sparse_attn_bwd_main_bf16_q128_hq8_hkv2_d128_p16"
QLEN, HQ, HKV, D = 128, 8, 2, 128
PAGE, PAGES = 16, 32
Q_ELEMENTS = QLEN * HQ * D
CACHE_ELEMENTS = PAGES * PAGE * HKV * D
GUARD_BYTES = 256
MAX_PTX_BYTES = 2 << 20
ABS_TOL, REL_TOL = 3e-2, 5e-2
CASE_SEEDS = (3, 11, 19, 27)


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
        self.device, self.context = ctypes.c_int(), ctypes.c_void_p()
        self.allocations = set()
        self.check(self.lib.cuInit(0), "cuInit")
        self.check(self.lib.cuDeviceGet(ctypes.byref(self.device), 0), "cuDeviceGet")
        self.check(self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, self.device), "cuCtxCreate")

    def _signatures(self):
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
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def error_name(self, result):
        name = ctypes.c_char_p()
        if self.lib.cuGetErrorName(result, ctypes.byref(name)) == 0 and name.value:
            return name.value.decode()
        return "UNKNOWN"

    def check(self, result, operation):
        if result:
            raise CudaError(f"{operation} failed: {self.error_name(result)} ({result})")

    def device_name(self):
        buffer = ctypes.create_string_buffer(256)
        self.check(self.lib.cuDeviceGetName(buffer, len(buffer), self.device), "cuDeviceGetName")
        return buffer.value.decode()

    def load_functions(self, path):
        module = ctypes.c_void_p()
        self.check(self.lib.cuModuleLoad(ctypes.byref(module), str(path).encode()), f"cuModuleLoad({path})")
        functions = []
        try:
            for entry in (ZERO_ENTRY, MAIN_ENTRY):
                function = ctypes.c_void_p()
                self.check(self.lib.cuModuleGetFunction(ctypes.byref(function), module, entry),
                           f"cuModuleGetFunction({entry.decode()})")
                functions.append(function)
        except Exception:
            self.lib.cuModuleUnload(module)
            raise
        return module, tuple(functions)

    def alloc(self, size):
        pointer = ctypes.c_uint64()
        self.check(self.lib.cuMemAlloc_v2(ctypes.byref(pointer), size), "cuMemAlloc")
        self.allocations.add(pointer.value)
        return pointer.value

    def free(self, pointer):
        if pointer in self.allocations:
            self.check(self.lib.cuMemFree_v2(pointer), "cuMemFree")
            self.allocations.remove(pointer)

    def copy_to_device(self, pointer, data):
        host = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        self.check(self.lib.cuMemcpyHtoD_v2(pointer, host, len(data)), "cuMemcpyHtoD")

    def copy_from_device(self, pointer, size):
        host = (ctypes.c_ubyte * size)()
        self.check(self.lib.cuMemcpyDtoH_v2(host, pointer, size), "cuMemcpyDtoH")
        return bytes(host)

    def _launch(self, function, grid, raw):
        args = (ctypes.c_void_p * len(raw))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in raw)
        )
        self.check(self.lib.cuLaunchKernel(function, grid, 1, 1, 128, 1, 1, 0, None, args, None),
                   "cuLaunchKernel")

    def launch_zero(self, function, dq, dk, dv):
        self._launch(function, 3072, [ctypes.c_uint64(dq), ctypes.c_uint64(dk), ctypes.c_uint64(dv)])

    def launch_main(self, function, pointers):
        order = ("q", "k", "v", "output", "grad_output", "lse", "page_table",
                 "row_offsets", "page_indices", "grad_q", "grad_k", "grad_v")
        self._launch(function, QLEN * HQ,
                     [ctypes.c_uint64(pointers[name]) for name in order])

    def synchronize(self):
        self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self):
        for pointer in tuple(self.allocations):
            self.lib.cuMemFree_v2(pointer)
        if self.context.value:
            self.lib.cuCtxDestroy_v2(self.context)


def float_to_bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if (bits & 0x7F800000) == 0x7F800000 and bits & 0x007FFFFF:
        return 0x7FC0
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


class Truth:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        p16 = ctypes.POINTER(ctypes.c_uint16)
        pi32 = ctypes.POINTER(ctypes.c_int32)
        pf = ctypes.POINTER(ctypes.c_float)
        self.lib.paged_sparse_forward_truth.argtypes = [p16, p16, p16, pi32, pi32, pi32, p16, pf]
        self.lib.paged_sparse_forward_truth.restype = ctypes.c_int
        self.lib.paged_sparse_backward_truth.argtypes = [
            p16, p16, p16, p16, p16, pf, pi32, pi32, pi32, pf, pf, pf,
        ]
        self.lib.paged_sparse_backward_truth.restype = ctypes.c_int

    @staticmethod
    def _p16(values):
        return (ctypes.c_uint16 * len(values)).from_buffer(values)

    @staticmethod
    def _pi32(values):
        return (ctypes.c_int32 * len(values)).from_buffer(values)

    @staticmethod
    def _pf(values):
        return (ctypes.c_float * len(values)).from_buffer(values)

    def compute(self, q, k, v, grad_output, page_table, row_offsets, page_indices):
        output = array("H", [0]) * Q_ELEMENTS
        lse = array("f", [0.0]) * (QLEN * HQ)
        forward = self.lib.paged_sparse_forward_truth(
            self._p16(q), self._p16(k), self._p16(v), self._pi32(page_table),
            self._pi32(row_offsets), self._pi32(page_indices), self._p16(output), self._pf(lse),
        )
        if forward:
            raise RuntimeError(f"CPU forward truth returned {forward}")
        dq = array("f", [0.0]) * Q_ELEMENTS
        dk = array("f", [0.0]) * CACHE_ELEMENTS
        dv = array("f", [0.0]) * CACHE_ELEMENTS
        backward = self.lib.paged_sparse_backward_truth(
            self._p16(q), self._p16(k), self._p16(v), self._p16(output),
            self._p16(grad_output), self._pf(lse), self._pi32(page_table),
            self._pi32(row_offsets), self._pi32(page_indices), self._pf(dq),
            self._pf(dk), self._pf(dv),
        )
        if backward:
            raise RuntimeError(f"CPU backward truth returned {backward}")
        return output, lse, (dq, dk, dv)


def make_metadata(seed):
    page_table = array("i", (((logical * 13) + seed) % PAGES for logical in range(PAGES)))
    row_offsets = array("i", [0])
    page_indices = array("i")
    for row in range(8):
        count = 2 + ((seed + row * 3) % 7)
        chosen = [0]
        cursor = (seed * 5 + row * 11 + 25) % PAGES
        while len(chosen) < count:
            if cursor not in chosen:
                chosen.append(cursor)
            cursor = (cursor + 7) % PAGES
        if (seed + row) % 2:
            chosen.reverse()
        else:
            chosen = chosen[1:] + chosen[:1]
        page_indices.extend(chosen)
        row_offsets.append(len(page_indices))
    return page_table, row_offsets, page_indices


def make_case(seed):
    q = array("H", [0]) * Q_ELEMENTS
    k = array("H", [0]) * CACHE_ELEMENTS
    v = array("H", [0]) * CACHE_ELEMENTS
    grad_output = array("H", [0]) * Q_ELEMENTS
    for i in range(Q_ELEMENTS):
        q[i] = float_to_bf16(((i * 17 + seed * 13) % 43 - 21) / 24.0)
        grad_output[i] = float_to_bf16(((i * 11 + seed * 7) % 37 - 18) / 20.0)
    for i in range(CACHE_ELEMENTS):
        kval = ((i * 7 + seed * 19) % 47 - 23) / 22.0
        vval = ((i * 13 + seed * 23) % 41 - 20) / 18.0
        k[i] = float_to_bf16(kval)
        v[i] = float_to_bf16(vval)
    # Force a different dominant logical token/head in each case.
    for d in range(D):
        physical = ((seed * 13) % PAGES) * PAGE
        k[(physical * HKV) * D + d] = float_to_bf16(2.5 if d % 2 == 0 else -2.5)
    return (q, k, v, grad_output, *make_metadata(seed))


def guarded(payload, byte):
    return bytearray([byte]) * GUARD_BYTES + payload + bytearray([byte]) * GUARD_BYTES


def assert_guards(payload, byte, label):
    guard = bytes([byte]) * GUARD_BYTES
    if payload[:GUARD_BYTES] != guard or payload[-GUARD_BYTES:] != guard:
        raise CandidateRejected(f"{label}: memory guard changed")


def assert_float_output(payload, expected, label):
    actual = array("f")
    actual.frombytes(payload)
    if len(actual) != len(expected):
        raise RuntimeError(f"{label}: output size mismatch")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if not math.isfinite(got) or not math.isclose(got, want, abs_tol=ABS_TOL, rel_tol=REL_TOL):
            raise CandidateRejected(f"{label}: wrong output index={index}: expected {want}, got {got}")


def run_case(cuda, functions, truth, seed, label):
    q, k, v, grad_output, page_table, row_offsets, page_indices = make_case(seed)
    output, lse, expected = truth.compute(q, k, v, grad_output, page_table, row_offsets, page_indices)
    payloads = {
        "q": q.tobytes(), "k": k.tobytes(), "v": v.tobytes(),
        "output": output.tobytes(), "grad_output": grad_output.tobytes(),
        "lse": lse.tobytes(), "page_table": page_table.tobytes(),
        "row_offsets": row_offsets.tobytes(), "page_indices": page_indices.tobytes(),
        "grad_q": (array("f", [1.25]) * Q_ELEMENTS).tobytes(),
        "grad_k": (array("f", [-2.5]) * CACHE_ELEMENTS).tobytes(),
        "grad_v": (array("f", [3.75]) * CACHE_ELEMENTS).tobytes(),
    }
    guard_values = {name: (0x91 + index * 7) & 0xFF for index, name in enumerate(payloads)}
    hosts = {name: guarded(data, guard_values[name]) for name, data in payloads.items()}
    devices = {name: cuda.alloc(len(data)) for name, data in hosts.items()}
    after = {}
    try:
        for name, data in hosts.items():
            cuda.copy_to_device(devices[name], data)
        pointers = {name: pointer + GUARD_BYTES for name, pointer in devices.items()}
        zero, main = functions
        candidate_call(f"{label} zero launch seed={seed}", cuda.launch_zero, zero,
                       pointers["grad_q"], pointers["grad_k"], pointers["grad_v"])
        candidate_call(f"{label} zero sync seed={seed}", cuda.synchronize)
        candidate_call(f"{label} main launch seed={seed}", cuda.launch_main, main, pointers)
        candidate_call(f"{label} main sync seed={seed}", cuda.synchronize)
        for name, data in hosts.items():
            after[name] = candidate_call(f"{label} copy {name}", cuda.copy_from_device,
                                         devices[name], len(data))
    finally:
        for pointer in reversed(tuple(devices.values())):
            try:
                cuda.free(pointer)
            except Exception:
                pass
    for name in ("q", "k", "v", "output", "grad_output", "lse", "page_table", "row_offsets", "page_indices"):
        if after[name] != bytes(hosts[name]):
            raise CandidateRejected(f"{label}: input {name} modified for seed={seed}")
    for index, name in enumerate(("grad_q", "grad_k", "grad_v")):
        assert_guards(after[name], guard_values[name], f"{label} {name} seed={seed}")
        body = after[name][GUARD_BYTES:GUARD_BYTES + len(expected[index]) * 4]
        assert_float_output(body, expected[index], f"{label} {name} seed={seed}")


def validate_functions(cuda, functions, truth, label):
    for seed in CASE_SEEDS:
        run_case(cuda, functions, truth, seed, label)


def validate_ptx(path, candidate):
    role = "Candidate" if candidate else "fixed Reference"
    error = CandidateRejected if candidate else RuntimeError
    if not path.is_file():
        raise error(f"missing {role} PTX: {path}")
    if path.stat().st_size == 0 or path.stat().st_size > MAX_PTX_BYTES:
        raise error(f"{role} PTX has invalid size")
    try:
        source = path.read_text()
    except UnicodeDecodeError as exception:
        raise error(f"{role} PTX is not UTF-8") from exception
    for entry in (ZERO_ENTRY, MAIN_ENTRY):
        if entry.decode() not in source:
            raise error(f"{role} PTX missing entry {entry.decode()}")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser()
    for name in ("candidate", "reference", "truth", "log", "reward"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    return parser.parse_args()


def run(args):
    validate_ptx(args.reference, False)
    validate_ptx(args.candidate, True)
    truth = Truth(args.truth)
    cuda = CudaDriver()
    reference_module = candidate_module = ctypes.c_void_p()
    try:
        reference_module, reference = cuda.load_functions(args.reference)
        try:
            validate_functions(cuda, reference, truth, "reference")
        except CandidateRejected as error:
            raise RuntimeError(f"fixed Reference correctness failed: {error}") from error
        candidate_module, candidate = candidate_call("Candidate module load", cuda.load_functions, args.candidate)
        candidate_call("Candidate correctness", validate_functions, cuda, candidate, truth, "candidate")
        metrics = {
            "status": "passed", "gpu": cuda.device_name(),
            "entries": [ZERO_ENTRY.decode(), MAIN_ENTRY.decode()],
            "correctness_cases": list(CASE_SEEDS), "atol": ABS_TOL, "rtol": REL_TOL,
            "scoring": "binary-correctness-upper-probe",
        }
        reward = {"reward": 1.0, "correctness": 1.0, "performance": 0.0}
        return metrics, reward
    finally:
        if candidate_module.value:
            cuda.lib.cuModuleUnload(candidate_module)
        if reference_module.value:
            cuda.lib.cuModuleUnload(reference_module)
        cuda.close()


def main():
    args = parse_args()
    reward = {"reward": 0.0, "correctness": 0.0, "performance": 0.0}
    metrics = {"status": "failed", "entries": [ZERO_ENTRY.decode(), MAIN_ENTRY.decode()]}
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
