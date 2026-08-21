#!/usr/bin/env python3

import ctypes
import sys
from pathlib import Path


CUDA_SUCCESS = 0
CC_MAJOR = 75
CC_MINOR = 76
THREADS_PER_BLOCK = 128
CANARY = -123456789


def check(result: int, operation: str) -> None:
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed with CUDA error {result}")


def configure_cuda(cuda: ctypes.CDLL) -> None:
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    cuda.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    cuda.cuCtxCreate_v2.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
        ctypes.c_int,
    ]
    cuda.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    cuda.cuModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    cuda.cuMemcpyHtoD_v2.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    cuda.cuLaunchKernel.argtypes = [
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
    cuda.cuCtxSynchronize.argtypes = []
    cuda.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
    cuda.cuModuleUnload.argtypes = [ctypes.c_void_p]
    cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]


def device_identity(cuda: ctypes.CDLL, device: ctypes.c_int) -> tuple[str, int, int]:
    name_buffer = ctypes.create_string_buffer(128)
    major = ctypes.c_int()
    minor = ctypes.c_int()
    check(cuda.cuDeviceGetName(name_buffer, len(name_buffer), device), "cuDeviceGetName")
    check(
        cuda.cuDeviceGetAttribute(ctypes.byref(major), CC_MAJOR, device),
        "cuDeviceGetAttribute(major)",
    )
    check(
        cuda.cuDeviceGetAttribute(ctypes.byref(minor), CC_MINOR, device),
        "cuDeviceGetAttribute(minor)",
    )
    return name_buffer.value.decode(), major.value, minor.value


def run_case(cuda: ctypes.CDLL, function: ctypes.c_void_p, n_value: int) -> None:
    input_values = [((index * 37) % 2001) - 1000 for index in range(n_value)]
    if n_value >= 2:
        input_values[0] = -2147483647
        input_values[-1] = 2147483646
    expected = [value + 1 for value in input_values]

    host_input = (ctypes.c_int32 * n_value)(*input_values)
    host_output = (ctypes.c_int32 * (n_value + 2))(
        CANARY, *([0] * n_value), CANARY
    )
    input_bytes = ctypes.sizeof(host_input)
    output_bytes = ctypes.sizeof(host_output)
    device_input = ctypes.c_uint64()
    device_output = ctypes.c_uint64()

    check(
        cuda.cuMemAlloc_v2(ctypes.byref(device_input), input_bytes),
        "cuMemAlloc(input)",
    )
    check(
        cuda.cuMemAlloc_v2(ctypes.byref(device_output), output_bytes),
        "cuMemAlloc(output)",
    )
    try:
        check(
            cuda.cuMemcpyHtoD_v2(device_input, host_input, input_bytes),
            "cuMemcpyHtoD(input)",
        )
        check(
            cuda.cuMemcpyHtoD_v2(device_output, host_output, output_bytes),
            "cuMemcpyHtoD(output)",
        )

        output_payload = ctypes.c_uint64(device_output.value + ctypes.sizeof(ctypes.c_int32))
        n = ctypes.c_uint32(n_value)
        kernel_args = (ctypes.c_void_p * 3)(
            ctypes.cast(ctypes.byref(device_input), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(output_payload), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(n), ctypes.c_void_p),
        )
        blocks = (n_value + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        check(
            cuda.cuLaunchKernel(
                function,
                blocks,
                1,
                1,
                THREADS_PER_BLOCK,
                1,
                1,
                0,
                None,
                kernel_args,
                None,
            ),
            f"cuLaunchKernel(n={n_value})",
        )
        check(cuda.cuCtxSynchronize(), f"cuCtxSynchronize(n={n_value})")
        check(
            cuda.cuMemcpyDtoH_v2(host_output, device_output, output_bytes),
            "cuMemcpyDtoH(output)",
        )
    finally:
        if device_output.value:
            cuda.cuMemFree_v2(device_output)
        if device_input.value:
            cuda.cuMemFree_v2(device_input)

    if host_output[0] != CANARY or host_output[-1] != CANARY:
        raise AssertionError(f"output boundary overwritten for n={n_value}")
    actual = list(host_output)[1:-1]
    if actual != expected:
        raise AssertionError(
            f"wrong output for n={n_value}: expected {expected[:8]}, got {actual[:8]}"
        )


def main() -> None:
    ptx_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/kernel.ptx")
    if not ptx_path.is_file():
        raise FileNotFoundError(f"missing PTX output: {ptx_path}")

    cuda = ctypes.CDLL("libcuda.so.1")
    configure_cuda(cuda)
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()

    check(cuda.cuInit(0), "cuInit")
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    name, major, minor = device_identity(cuda, device)
    if (major, minor) != (8, 9) or "L40" not in name:
        raise RuntimeError(
            f"expected NVIDIA L40 compute capability 8.9, got {name} ({major}.{minor})"
        )
    check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")

    try:
        check(
            cuda.cuModuleLoad(ctypes.byref(module), str(ptx_path).encode()),
            "cuModuleLoad",
        )
        check(
            cuda.cuModuleGetFunction(ctypes.byref(function), module, b"add_one"),
            "cuModuleGetFunction(add_one)",
        )
        for n_value in (1, 8, 257, 1025):
            run_case(cuda, function, n_value)
    finally:
        if module.value:
            cuda.cuModuleUnload(module)
        if context.value:
            cuda.cuCtxDestroy_v2(context)

    print(f"PTX_VERIFY_OK: {name} sm_{major}{minor}; 4 cases passed")


if __name__ == "__main__":
    main()
