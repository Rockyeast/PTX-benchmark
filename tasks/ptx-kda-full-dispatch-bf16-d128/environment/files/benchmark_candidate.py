#!/usr/bin/env python3
"""Public Candidate-only latency probe for the fixed T6 KDA task.

This is Agent feedback, not a trusted grader. It deliberately contains no
fixed Reference image, hidden verifier cases, CPU truth, expected outputs, or
reward calculation.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import struct
import subprocess
import tempfile
from array import array
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
GRID_X, GRID_Y, BLOCK_X = 32, N, 256
DYNAMIC_SHARED_BYTES = 50560
SCALE = 1.0 / math.sqrt(D)
MAX_PTX_BYTES = 4 << 20
LOGICAL_FLOPS = N * T * HV * D * D * 6


class CudaError(RuntimeError):
    pass


def bf16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def repeated_u16(values: list[int], count: int) -> bytes:
    pattern = array("H", values)
    repeated = pattern * math.ceil(count / len(pattern))
    del repeated[count:]
    return repeated.tobytes()


def public_inputs() -> tuple[list[bytes], bytes, list[bytes]]:
    """Create a disclosed deterministic workload unused by the hidden grader."""
    q_pattern = [bf16(((index % 13) - 6) / 32.0) for index in range(32)]
    k_pattern = [bf16(((index * 5 % 17) - 8) / 40.0) for index in range(32)]
    v_pattern = [bf16(((index * 7 % 19) - 9) / 24.0) for index in range(32)]
    decay_pattern = [bf16(-0.0078125 * (1 + index % 4)) for index in range(32)]
    beta_pattern = [bf16(0.0625 * (1 + index % 4)) for index in range(16)]
    tensor_inputs = [
        repeated_u16(q_pattern, QK_ELEMENTS),
        repeated_u16(k_pattern, QK_ELEMENTS),
        repeated_u16(v_pattern, VG_ELEMENTS),
        repeated_u16(decay_pattern, VG_ELEMENTS),
        repeated_u16(beta_pattern, BETA_ELEMENTS),
    ]
    # The first update makes this initially zero state non-zero. This keeps the
    # public workload deterministic without embedding a correctness oracle.
    initial_state = bytes(STATE_ELEMENTS * 2)
    indices = array(
        "i", ((position * 17 + 5) % (N * T) for position in range(INDEX_ELEMENTS))
    )
    accepted = array("i", (1 + sequence % T for sequence in range(N)))
    return tensor_inputs, initial_state, [indices.tobytes(), accepted.tobytes()]


class CudaDriver:
    def __init__(self) -> None:
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
        lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p
        ]
        lib.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        lib.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        lib.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
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

    def check(self, result: int, operation: str) -> None:
        if result:
            raise CudaError(f"{operation} failed: {self.error_name(result)} ({result})")

    def device_name(self) -> str:
        buffer = ctypes.create_string_buffer(256)
        self.check(self.lib.cuDeviceGetName(buffer, len(buffer), self.device), "cuDeviceGetName")
        return buffer.value.decode()

    def load_function(self, path: Path) -> tuple[ctypes.c_void_p, ctypes.c_void_p]:
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cuModuleLoad(ctypes.byref(module), str(path).encode()), "cuModuleLoad")
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

    def copy_to_device(self, pointer: int, data: bytes) -> None:
        host = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        self.check(self.lib.cuMemcpyHtoD_v2(pointer, host, len(data)), "cuMemcpyHtoD")

    def launch(self, function: ctypes.c_void_p, pointers: tuple[int, ...]) -> None:
        raw = [ctypes.c_uint64(pointer) for pointer in pointers]
        raw.append(ctypes.c_float(SCALE))
        arguments = (ctypes.c_void_p * len(raw))(
            *(ctypes.cast(ctypes.byref(item), ctypes.c_void_p) for item in raw)
        )
        self.check(
            self.lib.cuLaunchKernel(
                function,
                GRID_X, GRID_Y, 1,
                BLOCK_X, 1, 1,
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
        pointers: tuple[int, ...],
        state_pointer: int,
        initial_state: bytes,
        iterations: int,
    ) -> float:
        self.copy_to_device(state_pointer, initial_state)
        start, end = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cuEventCreate(ctypes.byref(start), 0), "cuEventCreate")
        self.check(self.lib.cuEventCreate(ctypes.byref(end), 0), "cuEventCreate")
        try:
            self.check(self.lib.cuEventRecord(start, None), "cuEventRecord(start)")
            for _ in range(iterations):
                self.launch(function, pointers)
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
        if self.context.value:
            self.lib.cuCtxDestroy_v2(self.context)


def validate_candidate(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing Candidate PTX: {path}")
    size = path.stat().st_size
    if size == 0 or size > MAX_PTX_BYTES:
        raise ValueError(f"Candidate PTX has invalid size {size}")
    source = path.read_text()
    if ".entry" not in source or ENTRY.decode() not in source:
        raise ValueError(f"Candidate PTX missing entry {ENTRY.decode()}")


def dispersion(values: list[float]) -> float:
    median = statistics.median(values)
    return math.inf if median == 0.0 else (max(values) - min(values)) / median


def run(candidate: Path) -> dict[str, object]:
    validate_candidate(candidate)
    with tempfile.TemporaryDirectory(prefix="direct-ptx-public-") as temporary:
        image = Path(temporary) / "candidate.cubin"
        assembly = subprocess.run(
            [
                "/usr/local/cuda/bin/ptxas",
                "-arch=sm_100a",
                "-v",
                "-o",
                str(image),
                str(candidate),
            ],
            capture_output=True,
            text=True,
        )
        if assembly.returncode:
            detail = (assembly.stderr or assembly.stdout).strip()
            raise ValueError(f"ptxas rejected Candidate:\n{detail}")

        tensor_inputs, initial_state, metadata_inputs = public_inputs()
        output = bytes(OUTPUT_ELEMENTS * 2)
        cuda = CudaDriver()
        module = ctypes.c_void_p()
        try:
            module, function = cuda.load_function(image)
            tensor_pointers = tuple(cuda.alloc(len(data)) for data in tensor_inputs)
            state_pointer = cuda.alloc(len(initial_state))
            output_pointer = cuda.alloc(len(output))
            metadata_pointers = tuple(cuda.alloc(len(data)) for data in metadata_inputs)
            pointers = tensor_pointers + (state_pointer, output_pointer) + metadata_pointers
            for pointer, data in zip(tensor_pointers, tensor_inputs):
                cuda.copy_to_device(pointer, data)
            cuda.copy_to_device(output_pointer, output)
            for pointer, data in zip(metadata_pointers, metadata_inputs):
                cuda.copy_to_device(pointer, data)

            # This catches load/launch/runtime faults, not semantic errors.
            cuda.copy_to_device(state_pointer, initial_state)
            cuda.launch(function, pointers)
            cuda.synchronize()

            cuda.measure_ms(function, pointers, state_pointer, initial_state, 3)
            probe = cuda.measure_ms(function, pointers, state_pointer, initial_state, 3)
            iterations = max(3, min(2000, math.ceil(30.0 / max(probe, 1.0e-6))))
            samples = [
                cuda.measure_ms(function, pointers, state_pointer, initial_state, iterations)
                for _ in range(5)
            ]
            median = statistics.median(samples)
            return {
                "status": "candidate_timed",
                "warning": "public Candidate timing is not hidden correctness or final reward",
                "gpu": cuda.device_name(),
                "entry": ENTRY.decode(),
                "public_case": "deterministic-public-v1",
                "shape": {"T": T, "N": N, "H": H, "HV": HV, "D": D},
                "launch": {
                    "grid": [GRID_X, GRID_Y, 1],
                    "block": [BLOCK_X, 1, 1],
                    "dynamic_shared_bytes": DYNAMIC_SHARED_BYTES,
                },
                "iterations_per_batch": iterations,
                "candidate_ms_per_launch": samples,
                "candidate_median_ms": median,
                "candidate_dispersion": dispersion(samples),
                "candidate_logical_tflops": LOGICAL_FLOPS / (median * 1.0e9),
                "ptxas_info": (assembly.stderr or assembly.stdout).strip(),
            }
        finally:
            if module.value:
                cuda.lib.cuModuleUnload(module)
            cuda.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Time one Candidate PTX on a public KDA development input."
    )
    parser.add_argument("candidate", type=Path, nargs="?", default=Path("/app/kernel.ptx"))
    args = parser.parse_args()
    try:
        metrics = run(args.candidate.resolve())
    except Exception as error:
        print(json.dumps({"status": "candidate_benchmark_failed", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
