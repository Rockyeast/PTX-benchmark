#!/usr/bin/env python3
"""Public Candidate-only timing probe for fixed block-sparse attention."""

from __future__ import annotations

import ctypes
import statistics
import struct
import sys
from pathlib import Path


ENTRY = b"block_sparse_attn_fwd_bf16_s1024_d128_b16"
TENSOR_BYTES = 1024 * 128 * 2
OFFSETS_BYTES = 65 * 4
INDICES_BYTES = 512 * 4
OUTPUT_BYTES = TENSOR_BYTES


class Driver:
    def __init__(self, ptx: Path) -> None:
        self.lib = ctypes.CDLL("libcuda.so.1")
        self._signatures()
        self.context = ctypes.c_void_p()
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        self.allocations: list[int] = []
        device = ctypes.c_int()
        self.check(self.lib.cuInit(0), "cuInit")
        self.check(self.lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
        self.check(self.lib.cuCtxCreate_v2(ctypes.byref(self.context), 0, device), "cuCtxCreate")
        self.check(self.lib.cuModuleLoad(ctypes.byref(self.module), str(ptx).encode()), "cuModuleLoad")
        self.check(
            self.lib.cuModuleGetFunction(ctypes.byref(self.function), self.module, ENTRY),
            "cuModuleGetFunction",
        )

    def _signatures(self) -> None:
        lib = self.lib
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuCtxCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
        lib.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemsetD8_v2.argtypes = [ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t]
        lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        lib.cuLaunchKernel.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        lib.cuEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        lib.cuEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.cuEventSynchronize.argtypes = [ctypes.c_void_p]
        lib.cuEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
        lib.cuEventDestroy_v2.argtypes = [ctypes.c_void_p]
        lib.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def check(self, code: int, operation: str) -> None:
        if code == 0:
            return
        name = ctypes.c_char_p()
        self.lib.cuGetErrorName(code, ctypes.byref(name))
        detail = name.value.decode() if name.value else "UNKNOWN"
        raise RuntimeError(f"{operation}: {detail} ({code})")

    def alloc(self, size: int) -> int:
        pointer = ctypes.c_uint64()
        self.check(self.lib.cuMemAlloc_v2(ctypes.byref(pointer), size), "cuMemAlloc")
        self.allocations.append(pointer.value)
        self.check(self.lib.cuMemsetD8_v2(pointer.value, 0, size), "cuMemsetD8")
        return pointer.value

    def launch(self, pointers: tuple[int, ...]) -> None:
        raw = [ctypes.c_uint64(value) for value in pointers]
        params = (ctypes.c_void_p * len(raw))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in raw)
        )
        self.check(
            self.lib.cuLaunchKernel(self.function, 1024, 1, 1, 128, 1, 1, 0, None, params, None),
            "cuLaunchKernel",
        )

    def copy_to_device(self, pointer: int, payload: bytes) -> None:
        host = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        self.check(self.lib.cuMemcpyHtoD_v2(pointer, host, len(payload)), "cuMemcpyHtoD")

    def measure(self, pointers: tuple[int, ...], iterations: int) -> float:
        start, end = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
        self.check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate")
        try:
            self.check(self.lib.cuEventRecord(start, None), "cuEventRecord")
            for _ in range(iterations):
                self.launch(pointers)
            self.check(self.lib.cuEventRecord(end, None), "cuEventRecord")
            self.check(self.lib.cuEventSynchronize(end), "cuEventSynchronize")
            elapsed = ctypes.c_float()
            self.check(self.lib.cuEventElapsedTime(ctypes.byref(elapsed), start, end), "cuEventElapsedTime")
            return float(elapsed.value) / iterations
        finally:
            self.lib.cuEventDestroy_v2(end)
            self.lib.cuEventDestroy_v2(start)

    def close(self) -> None:
        for pointer in reversed(self.allocations):
            self.lib.cuMemFree_v2(pointer)
        if self.module.value:
            self.lib.cuModuleUnload(self.module)
        if self.context.value:
            self.lib.cuCtxDestroy_v2(self.context)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /app/kernel.ptx", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file() or path.stat().st_size == 0:
        print(f"missing Candidate PTX: {path}", file=sys.stderr)
        return 2
    driver = Driver(path)
    try:
        pointers = tuple(driver.alloc(TENSOR_BYTES) for _ in range(3)) + (
            driver.alloc(OFFSETS_BYTES), driver.alloc(INDICES_BYTES),
            driver.alloc(OUTPUT_BYTES),
        )
        offsets = [row * 8 for row in range(65)]
        indices = [((row * 5) + slot * 7) % 64 for row in range(64) for slot in range(8)]
        driver.copy_to_device(pointers[3], struct.pack("<65i", *offsets))
        driver.copy_to_device(pointers[4], struct.pack("<512i", *indices))
        for _ in range(3):
            driver.launch(pointers)
        probe = driver.measure(pointers, 2)
        iterations = max(2, min(100, int(100.0 / max(probe, 1e-6))))
        samples = [driver.measure(pointers, iterations) for _ in range(5)]
        print(f"candidate_median_ms={statistics.median(samples):.6f}")
        print("warning=Candidate-only development input; hidden correctness not evaluated")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
