#!/usr/bin/env python3
"""Correctness-only verifier for the complete seven-route KDA PTX family."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
from dataclasses import dataclass
from pathlib import Path


H, D = 16, 128
SCALE = 1.0 / math.sqrt(D)
LOWER_BOUND = -5.0
GUARD_BYTES = 256
MAX_PTX_BYTES = 64 << 20
ABS_TOL = REL_TOL = 1.0e-2
OUTPUT_SENTINEL = 0x7FC0


@dataclass(frozen=True)
class Route:
    name: str
    tokens: int
    sequences: int
    value_heads: int
    grid_x: int
    block_x: int
    dynamic_shared: int
    mode: int

    @property
    def slots(self) -> int:
        return self.sequences if self.tokens == 1 else self.sequences * self.tokens + 6

    @property
    def total_tokens(self) -> int:
        return self.sequences * self.tokens


ROUTES = [
    Route("flash_kda_decode_sm100a_t1_split16", 1, 8, 32, 512, 32, 0, 0),
    Route("flash_kda_decode_sm100a_t1_split16", 1, 16, 32, 512, 32, 0, 1),
    Route("flash_kda_decode_sm100a_t1_split8", 1, 32, 32, 256, 32, 0, 0),
    Route("flash_kda_decode_sm100a_t1_split8", 1, 64, 32, 256, 32, 0, 1),
    Route("flash_kda_decode_sm100a_t2_split4", 2, 8, 32, 128, 64, 14720, 0),
    Route("flash_kda_decode_sm100a_t2_split4", 2, 16, 32, 128, 64, 14720, 1),
    Route("flash_kda_decode_sm100a_t3_split4", 3, 2, 16, 64, 96, 15872, 0),
    Route("flash_kda_decode_sm100a_t3_split4", 3, 8, 16, 64, 96, 15872, 1),
    Route("flash_kda_decode_sm100a_t4_split2", 4, 8, 32, 64, 128, 25856, 0),
    Route("flash_kda_decode_sm100a_t4_split2", 4, 16, 32, 64, 128, 25856, 1),
    Route("flash_kda_decode_sm100a_t5_split1", 5, 8, 32, 32, 256, 49024, 0),
    Route("flash_kda_decode_sm100a_t5_split1", 5, 16, 32, 32, 256, 49024, 1),
    Route("flash_kda_decode_sm100a_t6_split1", 6, 8, 32, 32, 256, 50560, 0),
    Route("flash_kda_decode_sm100a_t6_split1", 6, 16, 32, 32, 256, 50560, 1),
]
REQUIRED_ENTRIES = tuple(dict.fromkeys(route.name for route in ROUTES))


class CudaError(RuntimeError):
    pass


class CandidateRejected(Exception):
    pass


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
        self.check(
            self.lib.cuDeviceGetName(buffer, len(buffer), self.device),
            "cuDeviceGetName",
        )
        return buffer.value.decode()

    def load_module(self, path: Path):
        module = ctypes.c_void_p()
        self.check(
            self.lib.cuModuleLoad(ctypes.byref(module), str(path).encode()),
            f"cuModuleLoad({path})",
        )
        functions = {}
        try:
            for entry in REQUIRED_ENTRIES:
                function = ctypes.c_void_p()
                self.check(
                    self.lib.cuModuleGetFunction(
                        ctypes.byref(function), module, entry.encode()
                    ),
                    f"cuModuleGetFunction({entry})",
                )
                functions[entry] = function
            for route in ROUTES:
                if route.dynamic_shared:
                    self.check(
                        self.lib.cuFuncSetAttribute(
                            functions[route.name], 8, route.dynamic_shared
                        ),
                        f"cuFuncSetAttribute({route.name})",
                    )
        except Exception:
            self.lib.cuModuleUnload(module)
            raise
        return module, functions

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

    def launch(self, function, route: Route, pointers: dict[str, int]):
        names = ["q", "k", "v", "g", "beta"]
        if route.tokens == 3:
            names.extend(["a_log", "dt_bias"])
        names.extend(["state", "output"])
        if route.tokens > 1:
            names.extend(["indices", "accepted"])
        raw: list[ctypes._SimpleCData] = [
            ctypes.c_uint64(pointers[name] + GUARD_BYTES) for name in names
        ]
        raw.append(ctypes.c_float(SCALE))
        if route.tokens == 3:
            raw.append(ctypes.c_float(LOWER_BOUND))
        arguments = (ctypes.c_void_p * len(raw))(
            *(ctypes.cast(ctypes.byref(item), ctypes.c_void_p) for item in raw)
        )
        self.check(
            self.lib.cuLaunchKernel(
                function,
                route.grid_x, route.sequences, 1,
                route.block_x, 1, 1,
                route.dynamic_shared, None, arguments, None,
            ),
            f"cuLaunchKernel({route.name},T{route.tokens},N{route.sequences})",
        )

    def synchronize(self):
        self.check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self):
        for pointer in tuple(self.allocations):
            self.lib.cuMemFree_v2(pointer)
        if self.context.value:
            self.lib.cuCtxDestroy_v2(self.context)


class Truth:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path))
        p16 = ctypes.POINTER(ctypes.c_uint16)
        pf = ctypes.POINTER(ctypes.c_float)
        p32 = ctypes.POINTER(ctypes.c_int32)
        self.lib.kda_full_make_case.argtypes = [
            p16, p16, p16, p16, p16, pf, pf, p16, p32, p32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
        ]
        self.lib.kda_full_make_case.restype = ctypes.c_int
        self.lib.kda_full_truth.argtypes = [
            p16, p16, p16, p16, p16, pf, pf, p16, p16, p32, p32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float,
        ]
        self.lib.kda_full_truth.restype = ctypes.c_int
        self.lib.kda_full_compare.argtypes = [
            p16, p16, ctypes.c_uint64, ctypes.c_float, ctypes.c_float,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.kda_full_compare.restype = ctypes.c_int

    def make_case(self, route: Route):
        total = route.total_tokens
        qk_count = total * H * D
        vg_count = total * route.value_heads * D
        beta_count = total * route.value_heads
        state_count = route.slots * route.value_heads * D * D
        index_count = max(1, route.sequences * route.tokens)
        accepted_count = route.sequences
        q = (ctypes.c_uint16 * qk_count)()
        k = (ctypes.c_uint16 * qk_count)()
        v = (ctypes.c_uint16 * vg_count)()
        g = (ctypes.c_uint16 * vg_count)()
        beta = (ctypes.c_uint16 * beta_count)()
        a_log = (ctypes.c_float * H)()
        dt_bias = (ctypes.c_float * (H * D))()
        state = (ctypes.c_uint16 * state_count)()
        indices = (ctypes.c_int32 * index_count)()
        accepted = (ctypes.c_int32 * accepted_count)()
        result = self.lib.kda_full_make_case(
            q, k, v, g, beta, a_log, dt_bias, state, indices, accepted,
            route.tokens, route.sequences, route.value_heads, route.slots,
            route.mode,
        )
        if result:
            raise RuntimeError(f"CPU case generator returned {result} for {route}")
        return {
            "q": q, "k": k, "v": v, "g": g, "beta": beta,
            "a_log": a_log, "dt_bias": dt_bias, "state": state,
            "indices": indices, "accepted": accepted,
        }

    def compute(self, route: Route, case):
        state_count = route.slots * route.value_heads * D * D
        output_count = route.total_tokens * route.value_heads * D
        expected_state = (ctypes.c_uint16 * state_count)()
        ctypes.memmove(expected_state, case["state"], ctypes.sizeof(expected_state))
        expected_output = (ctypes.c_uint16 * output_count)()
        result = self.lib.kda_full_truth(
            case["q"], case["k"], case["v"], case["g"], case["beta"],
            case["a_log"], case["dt_bias"], expected_state, expected_output,
            case["indices"], case["accepted"], route.tokens, route.sequences,
            route.value_heads, route.slots, ctypes.c_float(SCALE),
            ctypes.c_float(LOWER_BOUND),
        )
        if result:
            raise RuntimeError(f"CPU truth returned {result} for {route}")
        return expected_state, expected_output

    def compare(self, got: bytes, expected, count: int, label: str):
        got_array = (ctypes.c_uint16 * count).from_buffer_copy(got)
        bad_index = ctypes.c_uint64()
        bad_got = ctypes.c_float()
        bad_expected = ctypes.c_float()
        max_abs = ctypes.c_float()
        max_rel = ctypes.c_float()
        result = self.lib.kda_full_compare(
            got_array, expected, count, ctypes.c_float(ABS_TOL),
            ctypes.c_float(REL_TOL), ctypes.byref(bad_index),
            ctypes.byref(bad_got), ctypes.byref(bad_expected),
            ctypes.byref(max_abs), ctypes.byref(max_rel),
        )
        metrics = {"max_abs": float(max_abs.value), "max_rel": float(max_rel.value)}
        if result == 1:
            raise CandidateRejected(
                f"{label}: mismatch at index {bad_index.value}: "
                f"expected {bad_expected.value}, got {bad_got.value}"
            )
        if result:
            raise RuntimeError(f"CPU comparison helper returned {result}")
        return metrics


def array_bytes(value) -> bytes:
    return ctypes.string_at(ctypes.addressof(value), ctypes.sizeof(value))


def guarded(payload: bytes, marker: int) -> bytes:
    guard = bytes([marker]) * GUARD_BYTES
    return guard + payload + guard


def assert_guards(payload: bytes, marker: int, label: str):
    guard = bytes([marker]) * GUARD_BYTES
    if payload[:GUARD_BYTES] != guard or payload[-GUARD_BYTES:] != guard:
        raise CandidateRejected(f"{label}: memory guard changed")


def run_case(cuda: CudaDriver, function, truth: Truth, route: Route):
    label = f"{route.name}/T{route.tokens}/N{route.sequences}/mode{route.mode}"
    case = truth.make_case(route)
    expected_state, expected_output = truth.compute(route, case)
    output_count = route.total_tokens * route.value_heads * D
    payloads = {name: array_bytes(value) for name, value in case.items()}
    payloads["output"] = bytes(
        (ctypes.c_uint16 * output_count)(*([OUTPUT_SENTINEL] * output_count))
    )
    markers = {
        "q": 0xA1, "k": 0xB2, "v": 0xC3, "g": 0xD4,
        "beta": 0xE5, "a_log": 0xF6, "dt_bias": 0x87,
        "state": 0x96, "indices": 0x69, "accepted": 0x5A,
        "output": 0x78,
    }
    hosts = {name: guarded(data, markers[name]) for name, data in payloads.items()}
    devices = {name: cuda.alloc(len(data)) for name, data in hosts.items()}
    after: dict[str, bytes] = {}
    try:
        for name, data in hosts.items():
            cuda.copy_to_device(devices[name], data)
        cuda.launch(function, route, devices)
        cuda.synchronize()
        for name, data in hosts.items():
            after[name] = cuda.copy_from_device(devices[name], len(data))
    except Exception as error:
        if isinstance(error, CandidateRejected):
            raise
        raise CandidateRejected(f"{label}: {error}") from error
    finally:
        for pointer in reversed(tuple(devices.values())):
            try:
                cuda.free(pointer)
            except Exception:
                pass

    for name in ("q", "k", "v", "g", "beta", "a_log", "dt_bias", "indices", "accepted"):
        if after[name] != hosts[name]:
            raise CandidateRejected(f"{label}: read-only input {name} was modified")
    for name in ("state", "output"):
        assert_guards(after[name], markers[name], f"{label} {name}")

    state_count = route.slots * route.value_heads * D * D
    return {
        "entry": route.name,
        "tokens": route.tokens,
        "sequences": route.sequences,
        "value_heads": route.value_heads,
        "mode": route.mode,
        "state": truth.compare(
            after["state"][GUARD_BYTES:-GUARD_BYTES], expected_state,
            state_count, f"{label} state",
        ),
        "output": truth.compare(
            after["output"][GUARD_BYTES:-GUARD_BYTES], expected_output,
            output_count, f"{label} output",
        ),
    }


def validate_ptx(path: Path):
    if not path.is_file():
        raise CandidateRejected(f"missing Candidate PTX: {path}")
    size = path.stat().st_size
    if size == 0 or size > MAX_PTX_BYTES:
        raise CandidateRejected(f"Candidate PTX has invalid size {size}")
    try:
        source = path.read_text()
    except UnicodeDecodeError as error:
        raise CandidateRejected("Candidate PTX is not UTF-8") from error
    missing = [entry for entry in REQUIRED_ENTRIES if entry not in source]
    if missing:
        raise CandidateRejected(f"Candidate PTX missing required entries: {missing}")


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-image", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    reward = {"reward": 0.0, "correctness": 0.0, "performance": 0.0}
    metrics = {"status": "failed", "required_entries": REQUIRED_ENTRIES}
    cuda = None
    module = ctypes.c_void_p()
    try:
        validate_ptx(args.candidate)
        if not args.candidate_image.is_file() or not args.candidate_image.stat().st_size:
            raise CandidateRejected("missing assembled Candidate image")
        truth = Truth(args.truth)
        cuda = CudaDriver()
        try:
            module, functions = cuda.load_module(args.candidate_image)
        except Exception as error:
            raise CandidateRejected(f"Candidate module/entry load failed: {error}") from error
        cases = [run_case(cuda, functions[route.name], truth, route) for route in ROUTES]
        metrics = {
            "status": "passed",
            "gpu": cuda.device_name(),
            "required_entries": REQUIRED_ENTRIES,
            "supported_shape_count": 30,
            "verified_cases": cases,
            "atol": ABS_TOL,
            "rtol": REL_TOL,
        }
        reward = {"reward": 1.0, "correctness": 1.0, "performance": 0.0}
    except CandidateRejected as error:
        metrics.update(
            status="candidate_rejected",
            error_type=type(error).__name__,
            error=str(error),
        )
        print(f"PTX verification failed: {error}")
        write_json(args.log, metrics)
        write_json(args.reward, reward)
        return 1
    except Exception as error:
        metrics.update(
            status="infrastructure_error",
            error_type=type(error).__name__,
            error=str(error),
        )
        print(f"PTX verifier infrastructure failure: {error}")
        write_json(args.log, metrics)
        write_json(args.reward, reward)
        return 2
    finally:
        if cuda is not None:
            if module.value:
                cuda.lib.cuModuleUnload(module)
            cuda.close()
    write_json(args.log, metrics)
    write_json(args.reward, reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
