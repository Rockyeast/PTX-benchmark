#!/usr/bin/env python3

import ctypes
import sys
from pathlib import Path


def check(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {result}")


def main() -> None:
    ptx_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/app/kernel.ptx")
    if not ptx_path.is_file():
        raise FileNotFoundError(f"missing PTX output: {ptx_path}")

    cuda = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    module = ctypes.c_void_p()
    function = ctypes.c_void_p()
    device_input = ctypes.c_uint64()
    device_output = ctypes.c_uint64()

    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
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

    check(cuda.cuInit(0), "cuInit")
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
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

        input_values = (ctypes.c_int32 * 8)(
            -4, -1, 0, 1, 7, 42, 1000, 2147483646
        )
        output_values = (ctypes.c_int32 * 8)()
        expected = [-3, 0, 1, 2, 8, 43, 1001, 2147483647]
        byte_count = ctypes.sizeof(input_values)

        check(
            cuda.cuMemAlloc_v2(ctypes.byref(device_input), byte_count),
            "cuMemAlloc(input)",
        )
        check(
            cuda.cuMemAlloc_v2(ctypes.byref(device_output), byte_count),
            "cuMemAlloc(output)",
        )
        try:
            check(
                cuda.cuMemcpyHtoD_v2(device_input, input_values, byte_count),
                "cuMemcpyHtoD",
            )
            n = ctypes.c_uint32(len(input_values))
            kernel_args = (ctypes.c_void_p * 3)(
                ctypes.cast(ctypes.byref(device_input), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(device_output), ctypes.c_void_p),
                ctypes.cast(ctypes.byref(n), ctypes.c_void_p),
            )
            check(
                cuda.cuLaunchKernel(
                    function,
                    1,
                    1,
                    1,
                    32,
                    1,
                    1,
                    0,
                    None,
                    kernel_args,
                    None,
                ),
                "cuLaunchKernel",
            )
            check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
            check(
                cuda.cuMemcpyDtoH_v2(output_values, device_output, byte_count),
                "cuMemcpyDtoH",
            )
        finally:
            if device_output.value:
                cuda.cuMemFree_v2(device_output)
            if device_input.value:
                cuda.cuMemFree_v2(device_input)

        actual = list(output_values)
        if actual != expected:
            raise AssertionError(f"wrong output: expected {expected}, got {actual}")
        print(f"PTX_VERIFY_OK: {len(actual)} outputs matched")
    finally:
        if module.value:
            cuda.cuModuleUnload(module)
        if context.value:
            cuda.cuCtxDestroy_v2(context)


if __name__ == "__main__":
    main()
